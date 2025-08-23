import ccxt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import warnings
import numba
import optuna
from datetime import datetime

# ==============================================================================
# ⚙️ 1. 사용자 설정
# ==============================================================================
OPTIMIZATION_SETTINGS = {
    'n_trials': 300,
    'search_space': {
        'timeframe_pairs': [('1m', '15m'), ('3m', '30m'), ('5m', '30m'), ('1m', '30m')],
        'rr_ratio': (3.0, 10.0),
        'atr_period': (3, 20),
        'atr_multiplier': (0.2, 2.0),
        'htf_swing_lookback': (25, 80),
    },
    'targets': {
        'daily_trades_target': 4.0,
        'daily_trades_tolerance': 1.0,
        'win_rate_target_min': 40.0,
        'win_rate_target_max': 50.0,
    }
}
BACKTEST_SETTINGS = {
    'period': {
        'start_date': "2024-07-01T00:00:00Z",
        'end_date': "2025-08-15T23:59:59Z",
    },
    'symbol': 'ETH/USDT', 
}

# ==============================================================================
# 2. 데이터 처리 및 전략 로직 함수
# ==============================================================================
def fetch_binance_data(symbol, timeframe, since, to):
    binance = ccxt.binance()
    since_timestamp, to_timestamp = binance.parse8601(since), binance.parse8601(to)
    all_ohlcv = []
    while since_timestamp < to_timestamp:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=1000)
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        since_timestamp = ohlcv[-1][0] + 1
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    df = df[df['timestamp'] <= pd.to_datetime(to_timestamp, unit='ms')]
    df.set_index('timestamp', inplace=True)
    return df[~df.index.duplicated(keep='first')]

def calculate_atr(df, period):
    tr = pd.concat([df['high'] - df['low'], np.abs(df['high'] - df['close'].shift()), np.abs(df['low'] - df['close'].shift())], axis=1).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

# <--- [수정] get_market_structure_trend 함수 안정성 강화 ---
def get_market_structure_trend(df_slice):
    if not isinstance(df_slice, pd.DataFrame) or df_slice.empty or len(df_slice) < 20:
        return 'SIDEWAYS'
    try:
        price_std = df_slice['high'].std()
        if pd.isna(price_std) or price_std == 0: return 'SIDEWAYS'
        
        high_peaks_idx, _ = find_peaks(df_slice['high'], prominence=price_std*0.7, distance=5)
        low_peaks_idx, _ = find_peaks(-df_slice['low'], prominence=price_std*0.7, distance=5)
        
        if len(high_peaks_idx) < 2 or len(low_peaks_idx) < 2: return 'SIDEWAYS'
        
        last_high, prev_high = df_slice['high'].iloc[high_peaks_idx[-1]], df_slice['high'].iloc[high_peaks_idx[-2]]
        last_low, prev_low = df_slice['low'].iloc[low_peaks_idx[-1]], df_slice['low'].iloc[low_peaks_idx[-2]]
        
        if (last_high > prev_high) and (last_low > prev_low): return 'UPTREND'
        elif (last_low < prev_low) and (last_high < prev_high): return 'DOWNTREND'
        else: return 'SIDEWAYS'
    except Exception: return 'SIDEWAYS'

# <--- [수정] 추세 계산 함수 로직 변경 ---
def precalculate_trend(df_trend, lookback):
    print(f"{lookback} 윈도우로 추세 사전 계산 중...")
    trends = [get_market_structure_trend(df_trend.iloc[i-lookback:i]) for i in range(lookback, len(df_trend))]
    # 앞부분은 계산이 불가능하므로 빈 값으로 채움
    trend_series = pd.Series([None]*lookback + trends, index=df_trend.index)
    print("추세 계산 완료.")
    return trend_series

@numba.jit(nopython=True)
def numba_signal_calculator(high, low, close, atr_threshold):
    max_fvgs = 200000 
    bull_fvgs, bear_fvgs = np.zeros((max_fvgs, 3)), np.zeros((max_fvgs, 3))
    bull_inv_fvgs, bear_inv_fvgs = np.zeros((max_fvgs, 2)), np.zeros((max_fvgs, 2))
    b_fvg_idx, br_fvg_idx, b_ifvg_idx, br_ifvg_idx = 0, 0, 0, 0
    n = len(high)
    signal_types, sl_prices = np.full(n, -1), np.full(n, np.nan)
    for i in range(2, n):
        if low[i] > high[i-2] and (low[i] - high[i-2]) > atr_threshold[i]:
            if b_fvg_idx < max_fvgs: bull_fvgs[b_fvg_idx], b_fvg_idx = [low[i], high[i-2], 0], b_fvg_idx + 1
        if high[i] < low[i-2] and (low[i-2] - high[i]) > atr_threshold[i]:
            if br_fvg_idx < max_fvgs: bear_fvgs[br_fvg_idx], br_fvg_idx = [low[i-2], high[i], 0], br_fvg_idx + 1
        for j in range(b_fvg_idx):
            if bull_fvgs[j, 2] == 0 and close[i] < bull_fvgs[j, 1]:
                bull_fvgs[j, 2] = 1
                if br_ifvg_idx < max_fvgs: bear_inv_fvgs[br_ifvg_idx], br_ifvg_idx = [bull_fvgs[j, 0], bull_fvgs[j, 1]], br_ifvg_idx + 1
        for j in range(br_fvg_idx):
            if bear_fvgs[j, 2] == 0 and close[i] > bear_fvgs[j, 0]:
                bear_fvgs[j, 2] = 1
                if b_ifvg_idx < max_fvgs: bull_inv_fvgs[b_ifvg_idx], b_ifvg_idx = [bear_fvgs[j, 0], bear_fvgs[j, 1]], b_ifvg_idx + 1
        sl, sh, sc = low[i-1], high[i-1], close[i-1]
        if b_ifvg_idx > 0:
            for j in range(b_ifvg_idx - 1, -1, -1):
                mid = (bull_inv_fvgs[j, 0] + bull_inv_fvgs[j, 1]) / 2
                if sl <= mid and sc > mid: signal_types[i-1], sl_prices[i-1] = 1, sl; break
        if signal_types[i-1] == -1 and br_ifvg_idx > 0:
            for j in range(br_ifvg_idx - 1, -1, -1):
                mid = (bear_inv_fvgs[j, 0] + bear_inv_fvgs[j, 1]) / 2
                if sh >= mid and sc < mid: signal_types[i-1], sl_prices[i-1] = 0, sh; break
    return signal_types, sl_prices

# ==============================================================================
# 3. 백테스팅 실행 및 Optuna Objective 함수
# ==============================================================================

def run_backtest(params, df_main, df_trend):
    df_trend_resampled = df_trend.reindex(df_main.index, method='ffill')
    df_main['atr'] = calculate_atr(df_main, params['atr_period'])
    atr_threshold = (df_main['atr'] * params['atr_multiplier']).fillna(0).to_numpy()
    signal_types_int, sl_prices = numba_signal_calculator(
        df_main['high'].to_numpy(), df_main['low'].to_numpy(), df_main['close'].to_numpy(), atr_threshold)
    df_main['signal_type'] = pd.Series(signal_types_int, index=df_main.index).map({-1: None, 0: 'bearish', 1: 'bullish'})
    df_main['sl_price'] = sl_prices

    trend_series = precalculate_trend(df_trend_resampled, params['htf_swing_lookback'])
    df_main = df_main.join(trend_series.rename('htf_trend'))
    df_main['htf_trend'].fillna(method='ffill', inplace=True)

    trades = []
    position = None
    entry_price, sl_price, tp_price = 0, 0, 0

    for i in range(params['htf_swing_lookback'], len(df_main)):
        current_candle = df_main.iloc[i]
        if position is None:
            signal_candle = df_main.iloc[i-1]
            signal_type, sl_price_val = signal_candle['signal_type'], signal_candle['sl_price']
            htf_trend = signal_candle['htf_trend']
            
            if signal_type and htf_trend:
                if (signal_type == 'bullish' and htf_trend == 'UPTREND') or (signal_type == 'bearish' and htf_trend == 'DOWNTREND'):
                    position = 'long' if signal_type == 'bullish' else 'short'
                    entry_price = current_candle['open']
                    risk_dist = abs(entry_price - sl_price_val)
                    if risk_dist <= 0: position = None; continue
                    sl_price = sl_price_val
                    tp_price = entry_price + risk_dist * params['rr_ratio'] if position == 'long' else entry_price - risk_dist * params['rr_ratio']
                    trades.append({'entry_time': df_main.index[i], 'side': position})
        elif position is not None:
            exit_price = 0
            if position == 'long':
                if current_candle['low'] <= sl_price: exit_price = sl_price
                elif current_candle['high'] >= tp_price: exit_price = tp_price
            elif position == 'short':
                if current_candle['high'] >= sl_price: exit_price = sl_price
                elif current_candle['low'] <= tp_price: exit_price = tp_price
            if exit_price > 0:
                last_trade = trades[-1]
                last_trade.update({'is_win': 1 if (position == 'long' and exit_price > entry_price) or (position == 'short' and exit_price < entry_price) else 0})
                position = None
    
    if not trades or 'is_win' not in trades[-1]: return 0, 0, 0
    trades_df = pd.DataFrame(trades)
    total_days = (df_main.index[-1] - df_main.index[0]).days + 1
    trades_per_day = len(trades_df) / total_days if total_days > 0 else 0
    win_rate = trades_df['is_win'].sum() / len(trades_df) * 100
    return trades_per_day, win_rate, params['rr_ratio']

def objective(trial, data_cache):
    space = OPTIMIZATION_SETTINGS['search_space']
    timeframe_pair_str = trial.suggest_categorical('timeframe_pair', [f"{p[0]}/{p[1]}" for p in space['timeframe_pairs']])
    main_tf, trend_tf = timeframe_pair_str.split('/')
    params = {
        'rr_ratio': trial.suggest_float('rr_ratio', *space['rr_ratio']),
        'atr_period': trial.suggest_int('atr_period', *space['atr_period']),
        'atr_multiplier': trial.suggest_float('atr_multiplier', *space['atr_multiplier']),
        'htf_swing_lookback': trial.suggest_int('htf_swing_lookback', *space['htf_swing_lookback']),
    }
    
    df_main = data_cache[main_tf]
    df_trend = data_cache[trend_tf]
    trades_per_day, win_rate, rr_ratio = run_backtest(params, df_main.copy(), df_trend.copy())
    
    targets = OPTIMIZATION_SETTINGS['targets']
    freq_target, freq_tol = targets['daily_trades_target'], targets['daily_trades_tolerance']
    wr_min, wr_max = targets['win_rate_target_min'], targets['win_rate_target_max']
    
    freq_penalty = max(0, abs(trades_per_day - freq_target) - freq_tol)
    win_rate_penalty = max(0, wr_min - win_rate) + max(0, win_rate - wr_max)
    
    trial.set_user_attr("trades_per_day", trades_per_day)
    trial.set_user_attr("win_rate", win_rate)

    if freq_penalty > 0 or win_rate_penalty > 0:
        return 1000 + freq_penalty * 10 + win_rate_penalty
    else:
        return -rr_ratio

# ==============================================================================
# 4. 메인 실행 블록
# ==============================================================================
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    data_cache = {}
    unique_timeframes = set()
    for main_tf, trend_tf in OPTIMIZATION_SETTINGS['search_space']['timeframe_pairs']:
        unique_timeframes.add(main_tf)
        unique_timeframes.add(trend_tf)
    
    print("최적화를 위해 필요한 모든 데이터를 미리 다운로드합니다...")
    for tf in unique_timeframes:
        print(f"'{tf}' 데이터 다운로드 중...")
        data_cache[tf] = fetch_binance_data(BACKTEST_SETTINGS['symbol'], tf, BACKTEST_SETTINGS['period']['start_date'], BACKTEST_SETTINGS['period']['end_date'])
    print("데이터 캐싱 완료.")

    study = optuna.create_study(direction='minimize')
    print(f"\n{OPTIMIZATION_SETTINGS['n_trials']}회의 시도를 통해 최적화를 시작합니다...")
    study.optimize(lambda trial: objective(trial, data_cache), n_trials=OPTIMIZATION_SETTINGS['n_trials'])

    print("\n" + "="*50)
    print("🚀 최적화 완료!")
    
    successful_trials = [t for t in study.trials if t.value < 0]
    
    if not successful_trials:
        print("\n아쉽게도 설정된 제약 조건(거래 빈도, 승률)을 만족하는 파라미터 조합을 찾지 못했습니다.")
        print("n_trials를 늘리거나, 탐색 범위(search_space) 또는 목표(targets)를 완화하여 다시 시도해보세요.")
        best_trial = study.best_trial
        print("\n가장 점수가 좋았던 trial 정보:")
        print(f"  - Score: {best_trial.value:.4f}")
        print(f"  - 일일 거래: {best_trial.user_attrs.get('trades_per_day', 'N/A'):.2f}회, 승률: {best_trial.user_attrs.get('win_rate', 'N/A'):.2f}%")
        print("  - Parameters:")
        for key, value in best_trial.params.items(): print(f"    - {key}: {value}")
            
    else:
        best_trial = min(successful_trials, key=lambda t: t.value)
        
        print("\n목표 제약 조건을 만족하는 최적의 파라미터 조합을 찾았습니다!")
        print(f"  - 목표: 하루 약 {OPTIMIZATION_SETTINGS['targets']['daily_trades_target']}회 거래, 승률 {OPTIMIZATION_SETTINGS['targets']['win_rate_target_min']}-{OPTIMIZATION_SETTINGS['targets']['win_rate_target_max']}%")
        print(f"  - Best Found RR Ratio: {-best_trial.value:.2f}")
        print(f"  - 일일 거래: {best_trial.user_attrs.get('trades_per_day', 'N/A'):.2f}회, 승률: {best_trial.user_attrs.get('win_rate', 'N/A'):.2f}%")
        print("  - Best Hyperparameters:")
        for key, value in best_trial.params.items():
            print(f"    - {key}: {value}")
