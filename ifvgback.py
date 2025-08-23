# 필요 라이브러리 설치
# !pip install ccxt pandas numpy quantstats scipy tqdm --upgrade

import ccxt
import pandas as pd
import numpy as np
from scipy.signal import find_peaks
import quantstats as qs
from datetime import datetime
import logging
from tqdm import tqdm
import re

# 로깅 설정 (진행 상황 확인용)
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# =================================================================================
# ⚙️ 1. 백테스팅 설정
# =================================================================================
# --- 데이터 및 기간 설정 ---
SYMBOL = 'ETH/USDT'
TIMEFRAME = '5m'
TREND_TIMEFRAME = '30m'
BACKTEST_START_DATE = '2025-07-01T00:00:00Z' # 백테스트 시작일
BACKTEST_END_DATE = '2025-08-15T00:00:00Z'   # 백테스트 종료일

# --- 시뮬레이션 설정 ---
INITIAL_CAPITAL = 1000.0
DETAILED_TRADE_LOGGING = True   # 상세 거래 로그 출력 기능 ON/OFF
MAKER_FEE_RATE = 0.00004        # 진입, 익절 수수료 (0.004%)
TAKER_FEE_RATE = 0.0001         # 손절 수수료 (0.01%)
SLIPPAGE_RATE = 0.0005          # 시장가 체결 오차 범위 (0.05%)

# --- IFVG 전략 파라미터 ---
STRATEGY_PARAMS = {
    'rr_ratio': 5.0,
    'risk_per_trade_usd': 20,
    'ifvg_entry_level': 0.75,
    'atr_period': 5,
    'atr_multiplier': 1.0,
    'data_fetch_limit': 200,
    'htf_swing_lookback': 75,
}

# =================================================================================
# 🛠️ 2. 데이터 로드 함수
# =================================================================================
def fetch_binance_data(symbol, timeframe, since, until):
    binance = ccxt.binance()
    ohlcv = []
    since_timestamp = binance.parse8601(since)
    until_timestamp = binance.parse8601(until)
    
    if since_timestamp >= until_timestamp:
        logging.error("Error: Start date must be before end date.")
        return pd.DataFrame()

    logging.info(f"Fetching {symbol} {timeframe} data from Binance...")
    with tqdm(total=(until_timestamp - since_timestamp), unit='ms', desc=f"Fetching {timeframe} data", leave=False) as pbar:
        while since_timestamp < until_timestamp:
            try:
                new_data = binance.fetch_ohlcv(symbol, timeframe, since_timestamp, limit=1000)
                if not new_data: break
                fetched_duration = new_data[-1][0] - new_data[0][0] if len(new_data) > 0 else 0
                pbar.update(fetched_duration)
                since_timestamp = new_data[-1][0] + 1
                ohlcv.extend(new_data)
            except Exception as e:
                logging.error(f"Error fetching data: {e}"); break
    df = pd.DataFrame(ohlcv, columns=['timestamp', 'open', 'high', 'low', 'close', 'volume'])
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    df.set_index('timestamp', inplace=True)
    # Ensure data is within the exact requested range
    df = df.loc[since:until]
    logging.info(f"Data fetching for {timeframe} complete. Total {len(df)} candles.")
    return df

# =================================================================================
# 🔬 3. 전략 로직
# =================================================================================
def calculate_atr(df, period=14):
    high_low = df['high'] - df['low']
    high_close = np.abs(df['high'] - df['close'].shift())
    low_close = np.abs(df['low'] - df['close'].shift())
    tr = pd.DataFrame({'hl': high_low, 'hc': high_close, 'lc': low_close}).max(axis=1)
    return tr.ewm(alpha=1/period, adjust=False).mean()

def get_market_structure_trend(df_slice):
    if df_slice.empty or len(df_slice) < 20: return 'SIDEWAYS'
    try:
        price_std = df_slice['high'].std()
        high_peaks_idx, _ = find_peaks(df_slice['high'], prominence=price_std * 0.7, distance=5)
        low_peaks_idx, _ = find_peaks(-df_slice['low'], prominence=price_std * 0.7, distance=5)
        if len(high_peaks_idx) < 2 or len(low_peaks_idx) < 2: return 'SIDEWAYS'
        last_high, prev_high = df_slice['high'].iloc[high_peaks_idx[-1]], df_slice['high'].iloc[high_peaks_idx[-2]]
        last_low, prev_low = df_slice['low'].iloc[low_peaks_idx[-1]], df_slice['low'].iloc[low_peaks_idx[-2]]
        is_uptrend = (last_high > prev_high) and (last_low > prev_low)
        is_downtrend = (last_low < prev_low) and (last_high < prev_high)
        if is_uptrend: return 'UPTREND'
        elif is_downtrend: return 'DOWNTREND'
        else: return 'SIDEWAYS'
    except Exception: return 'SIDEWAYS'

def find_ifvg_arrow_signal(df, params):
    df = df.copy()
    if len(df) < params['atr_period'] + 5: return None
    df['atr'] = calculate_atr(df, period=params['atr_period'])
    atr_threshold = df['atr'] * params['atr_multiplier']
    bull_fvgs, bear_fvgs, bull_inv_fvgs, bear_inv_fvgs = [], [], [], []

    for i in range(2, len(df)):
        if df['low'].iloc[i] > df['high'].iloc[i-2] and (df['low'].iloc[i] - df['high'].iloc[i-2]) > atr_threshold.iloc[i]:
            bull_fvgs.append({'top': df['low'].iloc[i], 'bot': df['high'].iloc[i-2], 'inverted': False})
        if df['high'].iloc[i] < df['low'].iloc[i-2] and (df['low'].iloc[i-2] - df['high'].iloc[i]) > atr_threshold.iloc[i]:
            bear_fvgs.append({'top': df['low'].iloc[i-2], 'bot': df['high'].iloc[i], 'inverted': False})
        for fvg in bull_fvgs:
            if not fvg['inverted'] and df['close'].iloc[i] < fvg['bot']: fvg['inverted'] = True; bear_inv_fvgs.append(fvg)
        for fvg in bear_fvgs:
            if not fvg['inverted'] and df['close'].iloc[i] > fvg['top']: fvg['inverted'] = True; bull_inv_fvgs.append(fvg)
    
    if len(df) < 2: return None
    latest_candle, prev_candle = df.iloc[-1], df.iloc[-2]
    
    for fvg in reversed(bull_inv_fvgs):
        if prev_candle['low'] <= fvg['top'] and prev_candle['low'] > fvg['bot'] and latest_candle['close'] > fvg['top']:
            body_high, body_low = max(prev_candle['open'], prev_candle['close']), min(prev_candle['open'], prev_candle['close'])
            entry = body_high - (body_high - body_low) * (1 - params['ifvg_entry_level'])
            return {'type': 'bullish', 'entry_price': entry, 'sl_price': prev_candle['low'], 'sl_trigger_price': fvg['bot']}
    for fvg in reversed(bear_inv_fvgs):
        if prev_candle['high'] >= fvg['bot'] and prev_candle['high'] < fvg['top'] and latest_candle['close'] < fvg['bot']:
            body_high, body_low = max(prev_candle['open'], prev_candle['close']), min(prev_candle['open'], prev_candle['close'])
            entry = body_low + (body_high - body_low) * (1 - params['ifvg_entry_level'])
            return {'type': 'bearish', 'entry_price': entry, 'sl_price': prev_candle['high'], 'sl_trigger_price': fvg['top']}
    return None

def precalculate_signals(df, params):
    df['type'], df['entry_price'], df['sl_price'], df['sl_trigger_price'] = [None, np.nan, np.nan, np.nan]
    min_len = params['data_fetch_limit']
    for i in tqdm(range(min_len, len(df)), desc=f"Pre-calculating Signals ({params.get('timeframe', 'N/A')})", leave=False):
        history_df = df.iloc[i-min_len:i+1]
        setup = find_ifvg_arrow_signal(history_df, params)
        if setup:
            df.loc[df.index[i], ['type', 'entry_price', 'sl_price', 'sl_trigger_price']] = \
                [setup['type'], setup['entry_price'], setup['sl_price'], setup['sl_trigger_price']]
    return df

# =================================================================================
# 🚀 4. 백테스팅 엔진
# =================================================================================
class Backtester:
    def __init__(self, ltf_df, htf_df, strategy_params, capital, maker_fee_rate, taker_fee_rate, slippage_rate, detailed_logging):
        self.ltf_df, self.htf_df, self.params = ltf_df, htf_df, strategy_params
        self.initial_capital, self.balance = capital, capital
        self.maker_fee_rate, self.taker_fee_rate, self.slippage_rate = maker_fee_rate, taker_fee_rate, slippage_rate
        self.detailed_logging = detailed_logging
        self.trades, self.position, self.pending_order = [], {}, {}
        self.signal_count = 0

    def run(self):
        logging.info("Backtest starting...")
        trends = []
        htf_lookback = self.params['htf_swing_lookback']
        for i in tqdm(range(len(self.htf_df)), desc="Calculating HTF Trend", leave=False):
            if i < htf_lookback: trends.append(np.nan)
            else: trends.append(get_market_structure_trend(self.htf_df.iloc[i-htf_lookback:i]))
        self.htf_df['trend'] = trends

        self.ltf_df = pd.merge_asof(self.ltf_df.sort_index(), self.htf_df[['trend']].sort_index(), left_index=True, right_index=True, direction='forward')
        self.ltf_df.dropna(inplace=True)
        self.ltf_df = precalculate_signals(self.ltf_df, self.params)

        for i in tqdm(range(1, len(self.ltf_df)), desc="Running Backtest", leave=False):
            current_candle, prev_candle = self.ltf_df.iloc[i], self.ltf_df.iloc[i-1]
            if self.position:
                exit_price, exit_type = None, None
                if self.position['side'] == 'long':
                    if current_candle['high'] >= self.position['tp']: exit_price, exit_type = self.position['tp'], 'TP'
                    elif prev_candle['close'] < self.position['sl_trigger_price']: exit_price, exit_type = current_candle['open'], 'SL'
                elif self.position['side'] == 'short':
                    if current_candle['low'] <= self.position['tp']: exit_price, exit_type = self.position['tp'], 'TP'
                    elif prev_candle['close'] > self.position['sl_trigger_price']: exit_price, exit_type = current_candle['open'], 'SL'
                if exit_price: self._close_position(exit_price, current_candle.name, exit_type); continue
            
            current_signal = self.ltf_df.iloc[i]
            if pd.notna(current_signal['type']):
                setup = {k: current_signal[k] for k in ['type', 'entry_price', 'sl_price', 'sl_trigger_price']}
                htf_trend = current_signal['trend']
                if (setup['type'] == 'bullish' and htf_trend == 'UPTREND') or (setup['type'] == 'bearish' and htf_trend == 'DOWNTREND'):
                    if not self.pending_order or setup['entry_price'] != self.pending_order.get('entry_price'):
                        self.signal_count += 1
                        if self.detailed_logging: print(f"\n📩 NEW/UPDATED PENDING ORDER at {current_candle.name}: {setup['type']} @ {setup['entry_price']:.2f}")
                        self.pending_order = setup
            
            if self.pending_order:
                order, is_triggered = self.pending_order, False
                if order['type'] == 'bullish' and current_candle['low'] <= order['entry_price']: is_triggered = True
                elif order['type'] == 'bearish' and current_candle['high'] >= order['entry_price']: is_triggered = True
                if is_triggered: 
                    if self.detailed_logging: print(f"🎉 TRADE TRIGGERED at {current_candle.name}")
                    self._open_position(order, current_candle.name)
                    self.pending_order = {}
        
        logging.info("Backtest finished.")
        return pd.DataFrame(self.trades) if self.trades else None

    def _open_position(self, setup, entry_time):
        e, sl = setup['entry_price'], setup['sl_price']
        risk_dist = abs(e - sl)
        if risk_dist == 0: return
        size = self.params['risk_per_trade_usd'] / risk_dist
        side = 'long' if setup['type'] == 'bullish' else 'short'
        tp = e + risk_dist * self.params['rr_ratio'] if side == 'long' else e - risk_dist * self.params['rr_ratio']
        self.balance -= size * e * self.maker_fee_rate
        self.position = {'side': side, 'size': size, 'entry_price': e, 'sl': sl, 'tp': tp, 'sl_trigger_price': setup['sl_trigger_price'], 'entry_time': entry_time}

    def _close_position(self, exit_price, exit_time, exit_type):
        pnl = (exit_price - self.position['entry_price']) * self.position['size']
        if self.position['side'] == 'short': pnl *= -1
        fee_rate = self.maker_fee_rate if exit_type == 'TP' else self.taker_fee_rate
        if exit_type == 'SL': pnl -= self.position['size'] * exit_price * self.slippage_rate
        net_pnl = pnl - (self.position['size'] * exit_price * fee_rate)
        self.balance += net_pnl
        self.trades.append({'entry_time': self.position['entry_time'], 'exit_time': exit_time, 'side': self.position['side'], 'pnl': net_pnl, 'balance': self.balance, 'exit_type': exit_type})
        self.position = {}

# =================================================================================
# 🏁 5. 백테스트 실행
# =================================================================================
if __name__ == '__main__':
    ltf_data = fetch_binance_data(SYMBOL, TIMEFRAME, BACKTEST_START_DATE, BACKTEST_END_DATE)
    htf_data = fetch_binance_data(SYMBOL, TREND_TIMEFRAME, BACKTEST_START_DATE, BACKTEST_END_DATE)
    
    if not ltf_data.empty and not htf_data.empty:
        backtester = Backtester(ltf_df=ltf_data, htf_df=htf_data, strategy_params=STRATEGY_PARAMS, 
                                capital=INITIAL_CAPITAL, maker_fee_rate=MAKER_FEE_RATE, 
                                taker_fee_rate=TAKER_FEE_RATE, slippage_rate=SLIPPAGE_RATE, 
                                detailed_logging=True)
        results = backtester.run()
        
        if results is not None and not results.empty:
            results.set_index('exit_time', inplace=True)
            results.index = pd.to_datetime(results.index) # 인덱스를 datetime으로 변환
            
            # --- 상세 거래 로그 출력 (추가된 부분) ---
            print("\n--- Detailed Trade Log ---")
            log_df = results.copy()
            log_df['entry_time'] = pd.to_datetime(log_df['entry_time'])
            for index, row in log_df.iterrows():
                print(f"[{row['entry_time'].strftime('%Y-%m-%d %H:%M')}] -> [{index.strftime('%Y-%m-%d %H:%M')}] | "
                      f"Side: {row['side']:<5} | PnL: ${row['pnl']:>8.2f} | Exit: {row['exit_type']:<3} | Balance: ${row['balance']:>10,.2f}")

            # --- 백테스트 요약 (수정된 부분) ---
            print("\n--- Backtest Summary ---")
            print(f"Initial Capital: ${INITIAL_CAPITAL:,.2f}"); print(f"Final Capital:   ${results['balance'].iloc[-1]:,.2f}")
            total_pnl = results['balance'].iloc[-1] - INITIAL_CAPITAL
            print(f"Total PnL:       ${total_pnl:,.2f} ({total_pnl/INITIAL_CAPITAL:.2%})")
            print(f"Total Signals:   {backtester.signal_count}")
            print(f"Total Trades:    {len(results)}")
            win_rate = (results['pnl'] > 0).mean(); print(f"Win Rate:        {win_rate:.2%}")
            wins = results[results['pnl'] > 0]['pnl']; losses = results[results['pnl'] <= 0]['pnl']
            avg_win = wins.mean() if not wins.empty else 0; avg_loss = losses.mean() if not losses.empty else 0
            print(f"Average Win:     ${avg_win:,.2f}"); print(f"Average Loss:    ${avg_loss:,.2f}")
            if losses.sum() != 0: print(f"Profit Factor:   {abs(wins.sum() / losses.sum()):.2f}")
            
            # --- 월별 승률 출력 (추가된 부분) ---
            print("\n--- Monthly Win Rate ---")
            monthly_win_rate = results.groupby(results.index.to_period('M')).apply(lambda x: (x['pnl'] > 0).mean())
            if not monthly_win_rate.empty:
                for month, rate in monthly_win_rate.items():
                    print(f"{month}: {rate:.2%}")
            else:
                print("Not enough data for monthly analysis.")

            # --- QuantStats 리포트 생성 ---
            returns = results['pnl'] / INITIAL_CAPITAL
            qs.reports.html(returns, output='backtest_report_ifvg.html', title=f'{SYMBOL} IFVG Strategy Backtest')
            logging.info("Backtest report saved as 'backtest_report_ifvg.html'")
        else:
            logging.warning("No trades were made during the backtest period.")
