# -*- coding: utf-8 -*-
import time
import pandas as pd
import pandas_ta as ta
import numpy as np
from scipy.signal import find_peaks
import logging
import math
import json
from gate_api import ApiClient, Configuration, FuturesOrder, FuturesApi, FuturesPriceTriggeredOrder, FuturesPriceTrigger
from gate_api.exceptions import ApiException, GateApiException
from datetime import datetime, timedelta

# =================================================================================
# ⚙️ 최종 설정 (IFVG 전략 파라미터 적용)
# =================================================================================
API_KEY = "YOUR_API_KEY"      # 실제 API 키로 변경하세요
API_SECRET = "YOUR_API_SECRET"  # 실제 API 시크릿으로 변경하세요

# --- 새로운 IFVG 전략 파라미터 ---
DEFAULT_CONFIG = {
    'contract': 'ETH_USDT',
    'timeframe': '1m',
    'trend_timeframe': '15m',
    'rr_ratio': 10.0,
    'risk_per_trade_usd': 0.5,      # 거래당 리스크 (달러)
    'leverage': '100',
    
    'atr_period': 14,
    'atr_multiplier': 0.25,

    'data_fetch_limit': 200,
    'htf_swing_lookback': 75,

    'use_reinvestment': True,
    'reinvestment_percent': 0.3,
    'initial_capital': 10.0,
}

# =================================================================================
# 🤖 BOT LOGIC & FUNCTIONS (최종 버전)
# =================================================================================
class TradingBot:
    def __init__(self, api_key, api_secret, config):
        self.params = config
        self.contract = config['contract']
        self.configuration = Configuration(key=api_key, secret=api_secret)
        self.api_client = ApiClient(self.configuration)
        self.futures_api = FuturesApi(self.api_client)
        self.settle = "usdt"
        self.active_order = None
        self.position_details = {}
        self.last_position_size = 0
        self.price_precision = 8
        self.quanto_multiplier = 1
        self.order_creation_time = None
        
        self.last_candle_timestamp = None

        self.state_file = "bot_state.json"
        self.current_capital = self.params['initial_capital']
        self.reinvestment_mode_activated = False
        self.reinvestment_amount = 0
        self.reinvestment_win_streak = 0
        self._load_state()

        self.pre_flight_checks()
        self.set_leverage()

    def _save_state(self):
        try:
            state = {
                'current_capital': self.current_capital,
                'reinvestment_mode_activated': self.reinvestment_mode_activated,
                'reinvestment_amount': self.reinvestment_amount,
                'reinvestment_win_streak': self.reinvestment_win_streak
            }
            with open(self.state_file, 'w') as f:
                json.dump(state, f, indent=4)
            logging.info(f"💾 상태 저장 완료. 현재 자본: ${self.current_capital:.2f}, 복리모드: {self.reinvestment_mode_activated}")
        except Exception as e:
            logging.error(f"상태 저장 실패: {e}")

    def _load_state(self):
        try:
            with open(self.state_file, 'r') as f:
                state = json.load(f)
                self.current_capital = state.get('current_capital', self.params['initial_capital'])
                self.reinvestment_mode_activated = state.get('reinvestment_mode_activated', False)
                self.reinvestment_amount = state.get('reinvestment_amount', 0)
                self.reinvestment_win_streak = state.get('reinvestment_win_streak', 0)
                logging.info(f"📂 상태 불러오기 완료. 복원된 자본: ${self.current_capital:.2f}, 복리모드: {self.reinvestment_mode_activated}")
        except FileNotFoundError:
            logging.warning("저장된 상태 파일 없음. 초기 설정으로 시작합니다.")
        except Exception as e:
            logging.error(f"상태 불러오기 실패: {e}")

    def handle_api_exception(self, e, context=""):
        error_context = f"오류 발생 지점: {context}"
        if isinstance(e, GateApiException):
            logging.error(f"{error_context}\nGate.io 서버 응답: [Label: {e.label}, Message: {e.message}]")
        else:
            logging.error(f"{error_context}\n전체 오류 내용: {e}", exc_info=True)
        return None

    def pre_flight_checks(self):
        logging.info("--- 시작 전 자가진단 시작 ---")
        try:
            logging.info("1. API 키 유효성 검사 중...")
            self.futures_api.list_futures_accounts(settle=self.settle)
            logging.info(" -> API 키가 유효합니다.")
            
            logging.info(f"2. {self.contract} 계약 정보 조회 중...")
            market_info = self.futures_api.get_futures_contract(settle=self.settle, contract=self.contract)
            self.price_precision = abs(int(math.log10(float(market_info.order_price_round))))
            self.quanto_multiplier = float(market_info.quanto_multiplier) if float(market_info.quanto_multiplier) != 0 else 1
            logging.info(f" -> 계약 정보 로드 완료: 가격 정밀도={self.price_precision}, 승수={self.quanto_multiplier}")
            
            logging.info("--- 자가진단 통과 ---")
        except Exception as e:
            self.handle_api_exception(e, "시작 전 자가진단")
            raise

    def set_leverage(self):
        try:
            logging.info(f"{self.contract}의 레버리지를 {self.params['leverage']}배로 설정합니다...")
            self.futures_api.update_position_leverage(settle=self.settle, contract=self.contract, leverage=str(self.params['leverage']))
            logging.info("레버리지 설정 완료.")
        except GateApiException as e:
            if "leverage not changed" in str(e.body):
                logging.warning("레버리지가 이미 설정되어 있습니다.")
            else:
                self.handle_api_exception(e, "레버리지 설정")
                raise

    def place_order(self, size, price, reduce_only=False):
        formatted_price = self.format_price(price)
        if formatted_price is None and price != '0': return None
        client_order_id = f"t-{int(time.time() * 1000)}"
        order = FuturesOrder(
            contract=self.contract, size=size, price=str(formatted_price if price != '0' else '0'),
            tif='gtc', text=client_order_id, reduce_only=reduce_only
        )
        try:
            created_order = self.futures_api.create_futures_order(settle=self.settle, futures_order=order)
            side = "매수(롱)" if size > 0 else "매도(숏)"
            order_type = "지정가" if price != '0' else "시장가"
            if reduce_only: side = "포지션 종료"
            logging.info(f"✅ {order_type} 주문 제출 성공: {side} {abs(size)}계약 @ {formatted_price if price != '0' else 'Market'}")
            return created_order
        except GateApiException as e:
            return self.handle_api_exception(e, "주문 제출")

    def place_tp_sl_orders(self, size, side, sl_price, tp_price):
        try:
            close_size = -size if side == 'long' else size
            
            tp_order = FuturesOrder(
                contract=self.contract, size=close_size, price=str(self.format_price(tp_price)),
                tif='gtc', reduce_only=True, text='t-tp'
            )
            self.futures_api.create_futures_order(settle=self.settle, futures_order=tp_order)
            logging.info(f"✅ 익절(지정가) 주문 제출 성공: 포지션 종료 {abs(close_size)}계약 @ {self.format_price(tp_price)}")

            trigger = FuturesPriceTrigger(
                price=str(self.format_price(sl_price)),
                rule=1 if side == 'long' else 2
            )
            
            sl_order = FuturesPriceTriggeredOrder(
                initial=FuturesOrder(
                    contract=self.contract, size=close_size, price='0',
                    tif='ioc', reduce_only=True, text='t-sl-market-safety'
                ),
                trigger=trigger
            )
            
            self.futures_api.create_price_triggered_order(settle=self.settle, futures_price_triggered_order=sl_order)
            logging.info(f"✅ 조건부 시장가 손절(안전장치) 주문 제출 성공 (트리거 @ {self.format_price(sl_price)})")
            return True
            
        except GateApiException as e:
            self.handle_api_exception(e, "TP/SL 주문 제출")
            logging.error("❌ TP/SL 주문 제출 실패. 위험 관리를 위해 즉시 포지션을 시장가로 종료합니다.")
            
            try:
                position = self.futures_api.get_position(settle=self.settle, contract=self.contract)
                position_size = int(position.size or 0)
                if position_size != 0:
                    self.force_close_position_market(position_size)
            except Exception as close_e:
                logging.critical(f"🚨🚨🚨 비상! TP/SL 실패 후 포지션 강제 종료마저 실패했습니다. 즉시 수동 개입이 필요합니다! 오류: {close_e}")
            
            return False

    def force_close_position_market(self, position_size):
        logging.warning(f"🚨 포지션을 시장가로 강제 청산합니다... (수량: {position_size})")
        try:
            self.futures_api.cancel_futures_orders(settle=self.settle, contract=self.contract)
            logging.info("강제 청산을 위해 모든 대기 주문을 취소했습니다.")
            
            close_size = -position_size if position_size > 0 else abs(position_size)
            self.place_order(size=close_size, price='0', reduce_only=True)
            self.position_details = {}
        except GateApiException as e:
            self.handle_api_exception(e, "시장가 강제 청산")

    def run(self):
        logging.info("🚀 IFVG 전략 트레이딩 봇을 시작합니다...")
        logging.info(f"전략 파라미터: {self.params}")
        while True:
            try:
                self.check_and_execute_trade()
                time.sleep(5) 
            except Exception as e:
                logging.error(f"메인 루프에서 예상치 못한 오류 발생: {e}", exc_info=True)
                time.sleep(60)

    def check_and_execute_trade(self):
        try:
            position = self.futures_api.get_position(settle=self.settle, contract=self.contract)
            position_size = int(position.size or 0)
        except GateApiException as e:
            if "position not found" in str(e.body):
                position_size = 0; position = None
            else:
                self.handle_api_exception(e, "포지션 조회"); return

        if position_size == 0 and self.last_position_size != 0:
            self.handle_closed_position(position)
        
        self.last_position_size = position_size
        
        if position_size != 0:
            self.monitor_open_position(position_size)
            return

        logging.info("="*50)
        logging.info("📈 새로운 거래 기회를 탐색합니다...")
        df = self.get_historical_data(self.contract, self.params['timeframe'], self.params['data_fetch_limit'])
        trend_df = self.get_historical_data(self.contract, self.params['trend_timeframe'], self.params['htf_swing_lookback'] + 50)
        if df.empty or trend_df.empty: return

        htf_trend = get_market_structure_trend(df_slice=trend_df)
        new_setup = find_ifvg_arrow_signal(df, self.params)

        if new_setup:
            self.evaluate_and_place_order(new_setup, htf_trend)
        else:
            logging.info(f"현재 추세: {htf_trend}. 유효한 IFVG 셋업을 찾지 못했습니다.")

    def monitor_open_position(self, position_size):
        # [수정] '몸통 마감 손절' 로직을 사용하지 않도록 비활성화합니다.
        #        이제 모든 손절은 진입 시 설정된 조건부 주문(가격 터치)에 의해서만 처리됩니다.
        return

    def evaluate_and_place_order(self, setup, htf_trend):
        entry_price, sl_price = setup['entry_price'], setup['sl_price']
        risk_dist = abs(entry_price - sl_price)
        
        if risk_dist <= 0 or self.quanto_multiplier <= 0: return

        if self.reinvestment_mode_activated:
            risk_amount = self.reinvestment_amount
            logging.info(f"복리 모드 거래 실행. 리스크 금액: ${risk_amount:.4f}")
        else:
            risk_amount = self.params['risk_per_trade_usd']
        
        if risk_amount <= 0: 
            logging.warning("계산된 리스크 금액이 0 이하입니다. 거래를 건너뜁니다.")
            return
        
        size = int(risk_amount / (risk_dist * self.quanto_multiplier))

        if size == 0:
            logging.warning(f"계산된 주문 수량이 0입니다. 리스크 금액에 비해 진입-손절 거리가 너무 넓습니다. (Risk: ${risk_amount}, Dist: {risk_dist})")
            return
        
        order_side, order_size = None, 0
        if setup['type'] == 'bullish' and htf_trend == 'UPTREND':
            logging.info(f"📈 [롱 셋업 발견] 기준가: {entry_price}, 손절가: {sl_price}, 계산된 수량: {size}")
            order_side, order_size = 'long', size
        elif setup['type'] == 'bearish' and htf_trend == 'DOWNTREND':
            logging.info(f"📉 [숏 셋업 발견] 기준가: {entry_price}, 손절가: {sl_price}, 계산된 수량: {size}")
            order_side, order_size = 'short', -size
        
        if order_side:
            order = self.place_order(order_size, '0')
            if order:
                self.active_order = order
                self.order_creation_time = datetime.now()
                self.position_details = {
                    'sl': sl_price, 
                    'tp': entry_price + risk_dist * self.params['rr_ratio'] if order_side == 'long' else entry_price - risk_dist * self.params['rr_ratio'], 
                    'side': order_side, 
                    'size': order_size, 
                    'entry_price': entry_price,
                    'sl_trigger_price': setup['sl_trigger_price'],
                    'ifvg_zone': setup['ifvg_zone']
                }
                time.sleep(2)
                self.check_active_order_status()

    def check_active_order_status(self):
        if not self.active_order: return
        try:
            order_status = self.futures_api.get_futures_order(settle=self.settle, order_id=self.active_order.id)
            if order_status.status == 'finished':
                if order_status.finish_as == 'filled':
                    logging.info(f"🎉 주문 체결! {self.position_details['side'].upper()} 포지션에 진입합니다.")
                    position_size = abs(self.position_details['size'])
                    
                    if self.place_tp_sl_orders(size=position_size, side=self.position_details['side'], sl_price=self.position_details['sl'], tp_price=self.position_details['tp']):
                        logging.info("✅ TP(지정가)/SL(안전장치) 주문 제출 완료.")
                    
                    self.active_order = None
                    self.order_creation_time = None
                    self.last_candle_timestamp = None
                else:
                    logging.info(f"주문이 체결되지 않고 종료되었습니다: {order_status.finish_as}")
                    self.active_order = None; self.position_details = {}; self.order_creation_time = None
        except GateApiException as ex:
            if "order not found" in str(ex.body):
                logging.warning("주문을 찾을 수 없습니다. (이미 체결 및 처리되었을 가능성이 높습니다.)")
            else:
                self.handle_api_exception(ex, "주문 확인")
            self.active_order = None; self.position_details = {}; self.order_creation_time = None

    def handle_closed_position(self, position_obj):
        logging.info("="*50)
        logging.info("포지션이 종료되었습니다. 거래 기록 및 복리 로직을 처리합니다.")
        
        realised_pnl = float(position_obj.realised_pnl) if position_obj and position_obj.realised_pnl else 0
        logging.info(f"거래 기록: Side: {self.position_details.get('side', 'N/A')}, PnL: {realised_pnl}")
        self.position_details = {}

        if not self.params['use_reinvestment']:
            self._save_state()
            return

        self.current_capital += realised_pnl
        capital_threshold = self.params['initial_capital'] * 2
        is_capital_sufficient = self.current_capital >= capital_threshold

        if not is_capital_sufficient and self.reinvestment_mode_activated:
            logging.info(f"🔔 복리 모드 종료: 총자산(${self.current_capital:.2f})이 기준(${capital_threshold:.2f}) 미만으로 하락했습니다.")
            self.reinvestment_mode_activated = False
            self.reinvestment_win_streak = 0
            self.reinvestment_amount = 0

        if realised_pnl > 0:
            if self.reinvestment_mode_activated:
                self.reinvestment_win_streak += 1
                logging.info(f"🎉 복리 모드 중 익절! 연속 {self.reinvestment_win_streak}번째 익절입니다.")
                if self.reinvestment_win_streak >= 2:
                    logging.info("🔔 복리 모드 종료: 2연속 익절 목표를 달성했습니다.")
                    self.reinvestment_mode_activated = False
                    self.reinvestment_win_streak = 0
                    self.reinvestment_amount = 0
                else:
                    self.reinvestment_amount = realised_pnl * self.params['reinvestment_percent']
                    logging.info(f"다음 거래 리스크가 ${self.reinvestment_amount:.4f}로 설정되었습니다.")
            elif is_capital_sufficient:
                self.reinvestment_mode_activated = True
                self.reinvestment_win_streak = 1
                self.reinvestment_amount = realised_pnl * self.params['reinvestment_percent']
                logging.info(f"🚀 복리 모드 시작! 첫 거래 리스크는 ${self.reinvestment_amount:.4f} 입니다.")
        
        else:
            if self.reinvestment_mode_activated:
                logging.info("🔔 복리 모드 종료: 손절이 발생했습니다.")
                self.reinvestment_mode_activated = False
                self.reinvestment_win_streak = 0
                self.reinvestment_amount = 0

        self._save_state()

    def get_historical_data(self, contract, timeframe, limit):
        try:
            api_response = self.futures_api.list_futures_candlesticks(settle=self.settle, contract=contract, interval=timeframe, limit=limit)
            data = [[candle.t, candle.v, candle.c, candle.h, candle.l, candle.o] for candle in api_response]
            df = pd.DataFrame(data, columns=['t', 'v', 'c', 'h', 'l', 'o'])
            df = df.rename(columns={'t': 'timestamp', 'o': 'open', 'h': 'high', 'l': 'low', 'c': 'close', 'v': 'volume'})
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
            df = df.set_index('timestamp')
            return df.astype(float).sort_index()
        except GateApiException as ex:
            self.handle_api_exception(ex, f"{timeframe} 데이터 다운로드")
            return pd.DataFrame()

# =================================================================================
# STRATEGY LOGIC (단일 캔들 V자 반등/반락 전략)
# =================================================================================
def get_market_structure_trend(df_slice):
    if df_slice.empty or len(df_slice) < 20: return 'SIDEWAYS'
    try:
        price_std = df_slice['high'].std()
        high_peaks_idx, _ = find_peaks(df_slice['high'], prominence=price_std * 0.7, distance=5)
        low_peaks_idx, _ = find_peaks(-df_slice['low'], prominence=price_std * 0.7, distance=5)
        if len(high_peaks_idx) < 2 or len(low_peaks_idx) < 2: return 'SIDEWAYS'
        last_high, prev_high = df_slice['high'].iloc[high_peaks_idx[-1]], df_slice['high'].iloc[high_peaks_idx[-2]]
        last_low, prev_low = df_slice['low'].iloc[low_peaks_idx[-1]], df_slice['low'].iloc[low_peaks_idx[-2]]
        last_high_time, last_low_time = df_slice.index[high_peaks_idx[-1]], df_slice.index[low_peaks_idx[-1]]
        is_uptrend = (last_high > prev_high) and (last_low > prev_low) and (last_high_time > last_low_time)
        is_downtrend = (last_low < prev_low) and (last_high < prev_high) and (last_low_time > last_high_time)
        if is_uptrend: return 'UPTREND'
        elif is_downtrend: return 'DOWNTREND'
        else: return 'SIDEWAYS'
    except Exception: return 'SIDEWAYS'

def find_ifvg_arrow_signal(df, params):
    if len(df) < 5: return None
    
    df.ta.atr(length=params['atr_period'], append=True)
    atr_threshold = df[f'ATRr_{params["atr_period"]}'] * params['atr_multiplier']

    bull_fvgs, bear_fvgs = [], []
    bull_inv_fvgs, bear_inv_fvgs = [], []

    for i in range(2, len(df)):
        fvg_up = df['low'].iloc[i] > df['high'].iloc[i-2]
        fvg_down = df['high'].iloc[i] < df['low'].iloc[i-2]

        if fvg_up and (df['low'].iloc[i] - df['high'].iloc[i-2]) > atr_threshold.iloc[i]:
            bull_fvgs.append({'top': df['low'].iloc[i], 'bot': df['high'].iloc[i-2], 'inverted': False, 'index': i})
        
        if fvg_down and (df['low'].iloc[i-2] - df['high'].iloc[i]) > atr_threshold.iloc[i]:
            bear_fvgs.append({'top': df['low'].iloc[i-2], 'bot': df['high'].iloc[i], 'inverted': False, 'index': i})
        
        for fvg in bull_fvgs:
            if not fvg['inverted'] and df['close'].iloc[i] < fvg['bot']:
                fvg['inverted'] = True
                bear_inv_fvgs.append(fvg)

        for fvg in bear_fvgs:
            if not fvg['inverted'] and df['close'].iloc[i] > fvg['top']:
                fvg['inverted'] = True
                bull_inv_fvgs.append(fvg)
    
    if len(df) < 2: return None
    signal_candle = df.iloc[-2]
    
    for fvg in reversed(bull_inv_fvgs):
        midpoint = (fvg['top'] + fvg['bot']) / 2
        
        is_bullish_signal = signal_candle['low'] <= midpoint and signal_candle['close'] > midpoint
        
        if is_bullish_signal:
            entry_price = signal_candle['close']
            sl_price = signal_candle['low']
            return {
                'type': 'bullish',
                'entry_price': entry_price,
                'sl_price': sl_price,
                'sl_trigger_price': sl_price,
                'ifvg_zone': [fvg['top'], fvg['bot']]
            }

    for fvg in reversed(bear_inv_fvgs):
        midpoint = (fvg['top'] + fvg['bot']) / 2
        
        is_bearish_signal = signal_candle['high'] >= midpoint and signal_candle['close'] < midpoint
        
        if is_bearish_signal:
            entry_price = signal_candle['close']
            sl_price = signal_candle['high']
            return {
                'type': 'bearish',
                'entry_price': entry_price,
                'sl_price': sl_price,
                'sl_trigger_price': sl_price,
                'ifvg_zone': [fvg['top'], fvg['bot']]
            }

    return None

# =================================================================================
# MAIN EXECUTION
# =================================================================================
def main():
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()])
    try:
        bot = TradingBot(API_KEY, API_SECRET, DEFAULT_CONFIG)
        bot.run()
    except Exception as e:
        logging.critical(f"봇 초기화 또는 실행 중 심각한 오류 발생: {e}", exc_info=True)

if __name__ == "__main__":
    main()
