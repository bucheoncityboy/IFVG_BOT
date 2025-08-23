import ccxt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from datetime import datetime
import warnings
import numba

# ==============================================================================
# ⚙️ 1. 사용자 설정 (이곳에서 모든 것을 제어하세요)
# ==============================================================================
BACKTEST_SETTINGS = {
    # --- 기간 및 심볼 설정 ---
    'period': {
        'start_date': "2024-07-01T00:00:00Z",
        'end_date': "2024-07-31T23:59:59Z", # 종료일 설정
    },
    'symbol_info': {
        'contract': 'ETH/USDT', 
        'timeframe': '5m',
        'trend_timeframe': '30m',
    },
    
    # --- 리스크 관리 및 자금 설정 ---
    'risk_management': {
        'initial_capital': 1000.0,            # 초기 시드머니 (USD)
        'risk_per_trade_usd': 5.0,            # 거래당 감수할 손실액 (USD)
        'taker_fee_percent': 0.05 / 100,      # 시장가(진입/손절) 수수료 (Taker 0.05%)
        'maker_fee_percent': 0.02 / 100,      # 지정가(익절) 수수료 (Maker 0.02%)
    },
    
    # --- 전략 파라미터 ---
    'strategy_params': {
        'rr_ratio': 4.0,                      # 손익비 (Risk/Reward Ratio)
        'atr_period': 5,
        'atr_multiplier': 1.0,
        'htf_swing_lookback': 75,
    },

    # --- 재투자(복리) 설정 ---
    'reinvestment': {
        'use_reinvestment': True,             # 재투자 로직 사용 여부
        'reinvestment_percent': 0.3,          # 수익금의 재투자 비율 (30%)
        'capital_threshold_multiplier': 2.0,  # 복리 시작을 위한 자본 배수 (초기 자본의 2배)
        'win_streak_to_reset': 2,             # 연속 승리 시 복리 모드 초기화
    }
}

# ==============================================================================
# 2. 데이터 처리 및 전략 로직 함수
# ==============================================================================

def fetch_binance_data(symbol, timeframe, since, to):
    """지정한 시작-종료 기간의 바이낸스 OHLCV 데이터를 가져옵니다."""
    binance = ccxt.binance()
    since_timestamp = binance.parse8601(since)
    to_timestamp = binance.parse8601(to)
    
    all_ohlcv = []
    limit = 1000
    
    while since_timestamp < to_timestamp:
        ohlcv = binance.fetch_ohlcv(symbol, timeframe, since=since_timestamp, limit=limit)
        if not ohlcv: break
        all_ohlcv.extend(ohlcv)
        since_timestamp = ohlcv[-1][0] + 1
        
    df = pd.DataFrame(all_ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms')
    
    end_datetime = pd.to_datetime(to_timestamp, unit='ms')
    df = df[df['timestamp'] <= end_datetime]
    
    df.set_index('timestamp', inplace=True)
    df = df[~df.index.duplicated(keep='first')]
    return df

def calculate_atr(df, period):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    atr = tr.ewm(alpha=1/period, adjust=False).mean()
    return atr

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

def precalculate_trend(df_trend, lookback):
    print(f"{lookback} 윈도우로 추세 사전 계산 중...")
    trends = [get_market_structure_trend(df_trend.iloc[i-lookback:i]) for i in range(lookback, len(df_trend))]
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

def precalculate_signals_and_trend(df_main, df_trend, params):
    """신호와 추세를 모두 미리 계산하여 메인 데이터프레임에 추가합니다."""
    print("Numba로 진입 신호 초고속 계산 중...")
    df_main['atr'] = calculate_atr(df_main, params['strategy_params']['atr_period'])
    atr_threshold = (df_main['atr'] * params['strategy_params']['atr_multiplier']).fillna(0).to_numpy()
    signal_types_int, sl_prices = numba_signal_calculator(
        df_main['high'].to_numpy(), df_main['low'].to_numpy(), df_main['close'].to_numpy(), atr_threshold)
    df_main['signal_type'] = pd.Series(signal_types_int, index=df_main.index).map({-1: None, 0: 'bearish', 1: 'bullish'})
    df_main['sl_price'] = sl_prices
    print("신호 계산 완료.")
    
    df_trend_resampled = df_trend.reindex(df_main.index, method='ffill')
    trend_series = precalculate_trend(df_trend_resampled, params['strategy_params']['htf_swing_lookback'])
    df_main = df_main.join(trend_series.rename('htf_trend'))
    df_main['htf_trend'].fillna(method='ffill', inplace=True)
    return df_main

# ==============================================================================
# 3. 메인 실행 블록
# ==============================================================================
if __name__ == '__main__':
    warnings.filterwarnings('ignore')
    
    # --- 데이터 로딩 ---
    print("데이터 로딩 중...")
    df_main_data = fetch_binance_data(BACKTEST_SETTINGS['symbol_info']['contract'], BACKTEST_SETTINGS['symbol_info']['timeframe'], BACKTEST_SETTINGS['period']['start_date'], BACKTEST_SETTINGS['period']['end_date'])
    df_trend_data = fetch_binance_data(BACKTEST_SETTINGS['symbol_info']['contract'], BACKTEST_SETTINGS['symbol_info']['trend_timeframe'], BACKTEST_SETTINGS['period']['start_date'], BACKTEST_SETTINGS['period']['end_date'])
    print("데이터 로딩 완료.")
    
    # --- 신호 및 추세 사전 계산 ---
    df_precalculated = precalculate_signals_and_trend(df_main_data.copy(), df_trend_data.copy(), BACKTEST_SETTINGS)

    # --- 백테스팅 변수 초기화 ---
    trades = []
    balance = BACKTEST_SETTINGS['risk_management']['initial_capital']
    position, entry_price, sl_price, tp_price, position_size = None, 0, 0, 0, 0
    
    reinvestment_mode = False
    reinvestment_amount = 0
    reinvestment_win_streak = 0
    
    print("\n백테스팅 시뮬레이션 시작...")
    start_index = BACKTEST_SETTINGS['strategy_params']['htf_swing_lookback']
    for i in range(start_index, len(df_precalculated)):
        
        current_time = df_precalculated.index[i]
        current_candle = df_precalculated.iloc[i]
        
        # 1. 진입 로직
        if position is None:
            signal_candle = df_precalculated.iloc[i-1]
            signal_type = signal_candle['signal_type']
            htf_trend = signal_candle['htf_trend']
            
            if signal_type and htf_trend:
                if (signal_type == 'bullish' and htf_trend == 'UPTREND') or (signal_type == 'bearish' and htf_trend == 'DOWNTREND'):
                    risk_dist = abs(current_candle['open'] - signal_candle['sl_price'])
                    if risk_dist <= 0: continue
                        
                    current_risk_usd = reinvestment_amount if reinvestment_mode else BACKTEST_SETTINGS['risk_management']['risk_per_trade_usd']
                    position_size = current_risk_usd / risk_dist
                    
                    position = 'long' if signal_type == 'bullish' else 'short'
                    entry_price = current_candle['open']
                    sl_price = signal_candle['sl_price']
                    tp_price = entry_price + risk_dist * BACKTEST_SETTINGS['strategy_params']['rr_ratio'] if position == 'long' else entry_price - risk_dist * BACKTEST_SETTINGS['strategy_params']['rr_ratio']
                    
                    trades.append({'entry_time': current_time, 'entry_price': entry_price, 'side': position, 'sl': sl_price, 'tp': tp_price, 'pos_size': position_size, 'risk_usd': current_risk_usd})
                    print(f"{current_time} | {position.upper()} 진입 | Size: {position_size:.4f} | Risk: ${current_risk_usd:.2f}")

        # 2. 청산 로직
        if position is not None:
            exit_price, result = 0, ''
            if position == 'long':
                if current_candle['low'] <= sl_price: exit_price, result = sl_price, 'SL'
                elif current_candle['high'] >= tp_price: exit_price, result = tp_price, 'TP'
            elif position == 'short':
                if current_candle['high'] >= sl_price: exit_price, result = sl_price, 'SL'
                elif current_candle['low'] <= tp_price: exit_price, result = tp_price, 'TP'
            
            if exit_price > 0:
                last_trade = trades[-1]
                pnl_usd = (exit_price - entry_price) * position_size if position == 'long' else (entry_price - exit_price) * position_size
                
                entry_fee = entry_price * position_size * BACKTEST_SETTINGS['risk_management']['taker_fee_percent']
                exit_fee_rate = BACKTEST_SETTINGS['risk_management']['maker_fee_percent'] if result == 'TP' else BACKTEST_SETTINGS['risk_management']['taker_fee_percent']
                exit_fee = exit_price * position_size * exit_fee_rate
                total_fee = entry_fee + exit_fee
                net_pnl = pnl_usd - total_fee
                balance += net_pnl
                
                last_trade.update({'exit_time': current_time, 'exit_price': exit_price, 'result': result, 'pnl_usd': net_pnl, 'fees': total_fee, 'balance': balance})
                print(f"{current_time} | {position.upper()} 청산 | {result} | Net PnL: ${net_pnl:.2f} | Fees: ${total_fee:.2f} | Balance: ${balance:.2f}")
                
                if BACKTEST_SETTINGS['reinvestment']['use_reinvestment']:
                    capital_threshold = BACKTEST_SETTINGS['risk_management']['initial_capital'] * BACKTEST_SETTINGS['reinvestment']['capital_threshold_multiplier']
                    if not (balance >= capital_threshold) and reinvestment_mode:
                        reinvestment_mode, reinvestment_win_streak, reinvestment_amount = False, 0, 0
                    if net_pnl > 0:
                        if reinvestment_mode:
                            reinvestment_win_streak += 1
                            if reinvestment_win_streak >= BACKTEST_SETTINGS['reinvestment']['win_streak_to_reset']:
                                reinvestment_mode, reinvestment_win_streak, reinvestment_amount = False, 0, 0
                            else: reinvestment_amount = net_pnl * BACKTEST_SETTINGS['reinvestment']['reinvestment_percent']
                        elif balance >= capital_threshold:
                            reinvestment_mode, reinvestment_win_streak = True, 1
                            reinvestment_amount = net_pnl * BACKTEST_SETTINGS['reinvestment']['reinvestment_percent']
                    else: 
                        if reinvestment_mode: reinvestment_mode, reinvestment_win_streak, reinvestment_amount = False, 0, 0
                position = None
    
    print("\n백테스팅 완료.")

    # --- 결과 분석 및 시각화 ---
    if not trades:
        print("분석할 거래 내역이 없습니다.")
    else:
        trades_df = pd.DataFrame(trades).dropna()
        if trades_df.empty:
            print("완료된 거래가 없습니다.")
        else:
            # 최종 결과 리포트
            trades_df['is_win'] = np.where(trades_df['pnl_usd'] > 0, 1, 0)
            initial_capital = BACKTEST_SETTINGS['risk_management']['initial_capital']
            
            total_trades = len(trades_df)
            win_rate = trades_df['is_win'].sum() / total_trades * 100
            net_pnl_total = trades_df['pnl_usd'].sum()
            pnl_percent_total = (net_pnl_total / initial_capital) * 100
            total_fees = trades_df['fees'].sum()
            
            wins = trades_df[trades_df['is_win'] == 1]
            losses = trades_df[trades_df['is_win'] == 0]
            avg_win = wins['pnl_usd'].mean() if not wins.empty else 0
            avg_loss = abs(losses['pnl_usd'].mean()) if not losses.empty else 0
            profit_factor = wins['pnl_usd'].sum() / abs(losses['pnl_usd'].sum()) if not losses.empty and not losses['pnl_usd'].sum() == 0 else float('inf')
            avg_rr = avg_win / avg_loss if avg_loss > 0 else float('inf')
            
            trades_df['cum_max_balance'] = trades_df['balance'].cummax()
            trades_df['drawdown'] = trades_df['cum_max_balance'] - trades_df['balance']
            max_drawdown = trades_df['drawdown'].max()
            mdd_percent = (max_drawdown / trades_df['cum_max_balance'].max()) * 100 if not trades_df['cum_max_balance'].empty and trades_df['cum_max_balance'].max() > 0 else 0

            print("\n" + "="*50)
            print("📈 전체 백테스팅 결과")
            print("="*50)
            print(f" 기간: {BACKTEST_SETTINGS['period']['start_date']} ~ {BACKTEST_SETTINGS['period']['end_date']}")
            print(f" 최종 자산: ${balance:.2f}")
            print(f" 총 순손익: ${net_pnl_total:.2f} ({pnl_percent_total:.2f}%)")
            print(f" 총 거래 횟수: {total_trades}회")
            print(f" 승률: {win_rate:.2f}%")
            print(f" 평균 손익비 (실현 기준): {avg_rr:.2f} R")
            print(f" 수익 팩터 (Profit Factor): {profit_factor:.2f}")
            print(f" 최대 자본 하락 (MDD): -${max_drawdown:.2f} (-{mdd_percent:.2f}%)")
            print(f" 총 지불 수수료: ${total_fees:.2f}")
            
            trades_df['month'] = trades_df['entry_time'].dt.to_period('M')
            monthly_summary = trades_df.groupby('month').agg(
                trade_count=('entry_time', 'size'),
                win_count=('is_win', 'sum'),
                net_pnl_usd=('pnl_usd', 'sum')
            ).round(2)
            monthly_summary['win_rate'] = (monthly_summary['win_count'] / monthly_summary['trade_count'] * 100).round(2)

            print("\n" + "="*50)
            print("📅 월별 거래 분석")
            print("="*50)
            print(monthly_summary)
            
            # --- 시각화 ---
            from plotly.subplots import make_subplots
            import plotly.graph_objects as go

            fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.05, row_heights=[0.7, 0.3])
            
            fig.add_trace(go.Candlestick(x=df_precalculated.index, open=df_precalculated['open'], high=df_precalculated['high'], low=df_precalculated['low'], close=df_precalculated['close'], name='Price'), row=1, col=1)
            fig.add_trace(go.Scatter(x=trades_df.query("side=='long'")['entry_time'], y=trades_df.query("side=='long'")['entry_price'], mode='markers', marker=dict(color='green', symbol='triangle-up', size=10), name='Long Entry'), row=1, col=1)
            fig.add_trace(go.Scatter(x=trades_df.query("side=='short'")['entry_time'], y=trades_df.query("side=='short'")['entry_price'], mode='markers', marker=dict(color='red', symbol='triangle-down', size=10), name='Short Entry'), row=1, col=1)
            fig.add_trace(go.Scatter(x=trades_df['exit_time'], y=trades_df['exit_price'], mode='markers', marker=dict(color='blue', symbol='x', size=8), name='Exit'), row=1, col=1)

            for _, trade in trades_df.iterrows():
                fig.add_shape(type="line", x0=trade['entry_time'], y0=trade['sl'], x1=trade['exit_time'], y1=trade['sl'], line=dict(color="red", width=2, dash="dash"), row=1, col=1)
                fig.add_shape(type="line", x0=trade['entry_time'], y0=trade['tp'], x1=trade['exit_time'], y1=trade['tp'], line=dict(color="green", width=2, dash="dash"), row=1, col=1)
            
            fig.add_trace(go.Scatter(x=trades_df['exit_time'], y=trades_df['balance'], mode='lines', name='Balance', line=dict(color='orange', width=2)), row=2, col=1)
            
            fig.update_layout(title_text=f"{BACKTEST_SETTINGS['symbol_info']['contract']} 백테스팅 결과", xaxis_rangeslider_visible=False)
            fig.update_yaxes(title_text="Price (USDT)", row=1, col=1)
            fig.update_yaxes(title_text="Balance (USDT)", row=2, col=1)
            fig.show()
