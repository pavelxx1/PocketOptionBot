import tensorflow as tf
print(f"TensorFlow version: {tf.__version__}")
import base64
import json
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
from selenium.webdriver.common.by import By
from driver import get_driver
from utils import get_quotes, get_value

BASE_URL = 'https://pocketoption.com'
PERIOD = 0
CANDLES = []
ACTIONS = {}
MAX_ACTIONS = 100
ACTIONS_SECONDS = PERIOD
CURRENCY = None
CURRENCY_CHANGE = False
INITIAL_DEPOSIT = None
CURRENCY_CHANGE_DATE = datetime.now()

# Глобальные параметры для ручной настройки
OPTIMIZATION_ENABLED = 1
MANUAL_LOOKBACK = 12
MANUAL_THRESHOLD = 0.30

IS_AMOUNT_SET = False

driver = get_driver()

def load_web_driver():
    url = f'{BASE_URL}/en/cabinet/demo-quick-high-low/'
    driver.get(url)

def do_action(signal):
    last_value = CANDLES[-1][2]
    global ACTIONS, IS_AMOUNT_SET
    
    try:
        print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} {signal.upper()}, currency: {CURRENCY} last_value: {last_value}")
        driver.find_element(by=By.CLASS_NAME, value=f'btn-{signal}').click()
        ACTIONS[datetime.now()] = last_value
        IS_AMOUNT_SET = False
    except Exception as e:
        print(f"Ошибка при выполнении действия {signal}: {e}")

def generate_range(start, end, step):
    """Создает список значений от start до end с шагом step"""
    values = []
    current = start
    while current <= end:
        values.append(current)
        current = round(current + step, 10)
    return values

def market_microstructure_analysis(quotes, lookback=10):
    signals = []
    
    for i in range(lookback, len(quotes)-1):
        window = quotes[i-lookback+1:i+1]
        
        # Get basic price data
        close = [float(get_value(q)) for q in window]
        open_p = [float(get_value(q, 'open')) for q in window]
        high = [float(get_value(q, 'high')) for q in window]
        low = [float(get_value(q, 'low')) for q in window]
        
        # Calculate price momentum
        price_momentum = 0
        if len(close) >= 3 and close[-3] != 0:
            price_momentum = (close[-1] - close[-3]) / close[-3] * 100
        
        # Calculate short-term RSI
        rsi = 50
        try:
            if len(close) >= 2:
                gains = [max(0, close[j] - close[j-1]) for j in range(1, len(close))]
                losses = [max(0, close[j-1] - close[j]) for j in range(1, len(close))]
                
                if not gains and not losses:
                    rsi = 50
                else:
                    avg_gain = sum(gains) / max(1, len(gains))
                    avg_loss = sum(losses) / max(1, len(losses))
                    
                    if avg_loss == 0:
                        rsi = 100
                    else:
                        rs = avg_gain / avg_loss
                        rsi = 100 - (100 / (1 + rs))
        except Exception as e:
            print(f"RSI calculation error: {e}")
            rsi = 50
        
        # Calculate Bollinger Bands
        sma = sum(close[-5:]) / min(5, len(close))
        std_dev = max(0.0001, (sum((price - sma) ** 2 for price in close[-5:]) / min(5, len(close))) ** 0.5)
        upper_band = sma + (2 * std_dev)
        lower_band = sma - (2 * std_dev)
        
        # Stochastic oscillator
        k_percent = 50
        try:
            if len(high) >= 2 and len(low) >= 2:
                lowest_low = min(low)
                highest_high = max(high)
                
                if highest_high > lowest_low:
                    k_percent = 100 * (close[-1] - lowest_low) / (highest_high - lowest_low)
        except Exception as e:
            print(f"Stochastic calculation error: {e}")
        
        # Calculate candle patterns
        hammer = (close[-1] > open_p[-1]) and (high[-1] - close[-1]) < (close[-1] - low[-1]) * 0.3
        shooting_star = (close[-1] < open_p[-1]) and (high[-1] - open_p[-1]) > (open_p[-1] - low[-1]) * 3
        
        # Calculate price volatility
        recent_volatility = 0
        if len(high) >= 3 and len(low) >= 3:
            recent_volatility = sum([high[j] - low[j] for j in range(-3, 0)]) / 3
        
        avg_volatility = 0
        if len(high) > 0 and len(low) > 0:
            avg_volatility = sum([high[j] - low[j] for j in range(len(high))]) / len(high)
        
        volatility_ratio = 1
        if avg_volatility > 0:
            volatility_ratio = recent_volatility / avg_volatility
        
        # Price reversal patterns
        three_up = False
        three_down = False
        if len(close) >= 4:
            three_up = all(close[j] > close[j-1] for j in range(-3, 0))
            three_down = all(close[j] < close[j-1] for j in range(-3, 0))
        
        # Calculate score based on technical indicators
        score = 0.5
        
        # Bullish signals
        if price_momentum > 0.2:
            score += 0.08
        if rsi < 30:
            score += 0.12
        if close[-1] < lower_band:
            score += 0.10
        if k_percent < 20:
            score += 0.08
        if hammer:
            score += 0.15
        if three_down and len(close) >= 2 and close[-1] > close[-2]:
            score += 0.20
        if volatility_ratio > 1.5 and close[-1] > open_p[-1]:
            score += 0.07
            
        # Bearish signals
        if price_momentum < -0.2:
            score -= 0.08
        if rsi > 70:
            score -= 0.12
        if close[-1] > upper_band:
            score -= 0.10
        if k_percent > 80:
            score -= 0.08
        if shooting_star:
            score -= 0.15
        if three_up and len(close) >= 2 and close[-1] < close[-2]:
            score -= 0.20
        if volatility_ratio > 1.5 and close[-1] < open_p[-1]:
            score -= 0.07
            
        # Additional 1-minute specific signals
        price_change_3 = 0
        if len(close) >= 3 and close[-3] != 0:
            price_change_3 = abs(close[-1] - close[-3]) / close[-3]
            if price_change_3 > 0.005:
                if close[-1] > close[-3]:
                    score += 0.1
                else:
                    score -= 0.1
                
        # Determine signal type
        signal = 'call' if score > 0.55 else ('put' if score < 0.45 else 'neutral')
        
        signals.append({
            'date': window[-1].date,
            'signal': signal,
            'rsi': rsi,
            'stoch': k_percent,
            'price_momentum': price_momentum,
            'volatility_ratio': volatility_ratio,
            'score': score
        })
    
    return signals

def backtest_and_optimize(quotes, lookback_range=None, strength_thresholds=None):
    if lookback_range is None:
        lookback_range = generate_range(5, 15, 5)
    if strength_thresholds is None:
        strength_thresholds = generate_range(0.15, 0.35, 0.05)
    
    best_params = {'lookback': 10, 'threshold': 0.25, 'winrate': 0.0}
    best_results = []
    
    for lookback in lookback_range:
        lookback_int = int(lookback)
        for threshold in strength_thresholds:
            wins, losses = 0, 0
            trades = []
            
            future_check = 1
            
            for i in range(lookback_int, len(quotes)-future_check-1):
                signals = market_microstructure_analysis(quotes[:i+1], lookback=lookback_int)
                
                if not signals:
                    continue
                    
                last_signal = signals[-1]
                signal_strength = abs(last_signal['score'] - 0.5) * 2
                
                if last_signal['signal'] != 'neutral' and signal_strength > threshold:
                    current_close = float(get_value(quotes[i]))
                    future_close = float(get_value(quotes[i+future_check]))
                    
                    if last_signal['signal'] == 'call':
                        result = 'win' if future_close > current_close else 'loss'
                    elif last_signal['signal'] == 'put':
                        result = 'win' if future_close < current_close else 'loss'
                    else:
                        continue
                        
                    trades.append({
                        'signal': last_signal['signal'],
                        'strength': signal_strength,
                        'result': result
                    })
                    
                    if result == 'win':
                        wins += 1
                    else:
                        losses += 1
            
            total_trades = wins + losses
            winrate = wins / total_trades if total_trades > 0 else 0
            
            if winrate > best_params['winrate'] and total_trades >= 3:
                best_params = {
                    'lookback': lookback_int,
                    'threshold': threshold,
                    'winrate': winrate,
                    'trades': total_trades
                }
                best_results = trades
    
    return best_params, best_results

def check_data():
    try:
        quotes = get_quotes(CANDLES)[-110:]
        
        if len(quotes) < 20:
            print("Недостаточно данных для анализа")
            return
        
        lookback = MANUAL_LOOKBACK
        threshold = MANUAL_THRESHOLD
        trading_allowed = True
        
        if OPTIMIZATION_ENABLED and len(quotes) >= 35:
            print("Оптимизация параметров...")
            
            lookbacks = [12]
            thresholds = [0.30]
            
            print(f"Тестируем lookbacks: {lookbacks}")  
            print(f"Тестируем thresholds: {thresholds}")
            
            best_params, results = backtest_and_optimize(quotes, lookbacks, thresholds)   
            
            if results and len(results) > 0:
                wins = sum(1 for trade in results if trade['result'] == 'win')
                total = len(results)
                winrate = best_params['winrate']
                
                print(f"Оптимальные параметры: lookback={best_params['lookback']}, threshold={best_params['threshold']}")
                print(f"Исторический винрейт: {winrate*100:.1f}% ({wins}/{total} сделок)")
                
                min_winrate = 0.6
                if winrate >= min_winrate and total >= 3:
                    trading_allowed = True
                    print(f"Торговля разрешена (винрейт > {min_winrate*100:.1f}%)")
                    
                    threshold = best_params['threshold']
                    lookback = best_params['lookback']
                else:
                    trading_allowed = False
                    print(f"Торговля запрещена (винрейт < {min_winrate*100:.1f}% или мало сделок)")
            else:
                print("Оптимизация не дала результатов, используем ручные параметры")
        else:
            if not OPTIMIZATION_ENABLED:
                print(f"Оптимизация отключена. Используем ручные параметры: lookback={lookback}, threshold={threshold}")
            else:
                print("Недостаточно данных для оптимизации, используем ручные параметры")
        
        signals = market_microstructure_analysis(quotes, lookback=lookback)
        
        if signals and len(signals) > 0:
            last_signal = signals[-1]
            signal_strength = abs(last_signal['score'] - 0.5) * 2
            expiry_time = 1
            
            print(f"Сигнал: {last_signal['signal'].upper()}, Сила: {signal_strength:.2f}")
            print(f"RSI: {last_signal['rsi']:.2f}, Stoch: {last_signal['stoch']:.2f}")
            print(f"Momentum: {last_signal['price_momentum']:.2f}, Volatility: {last_signal['volatility_ratio']:.2f}")
            print(f"Экспирация: {expiry_time} мин, Скор: {last_signal['score']:.2f}")
            
            print(f"[DEBUG] Проверка условий для {last_signal['signal'].upper()}:")
            print(f"[DEBUG] 1. Торговля разрешена: {trading_allowed}")
            print(f"[DEBUG] 2. Сигнал не нейтральный: {last_signal['signal'] != 'neutral'}")
            print(f"[DEBUG] 3. Сила сигнала {signal_strength:.2f} >= {threshold}: {signal_strength >= threshold}")
            
            can_trade = trading_allowed and last_signal['signal'] != 'neutral' and signal_strength >= threshold
            print(f"[DEBUG] ИТОГО - можно торговать: {can_trade}")
            
            if can_trade:
                if last_signal['signal'] == 'put':
                    print(f"[DEBUG] Пытаюсь выполнить PUT...")
                    do_action('put')
                    print(f"....make SELL (экспирация: {expiry_time} мин)")
                elif last_signal['signal'] == 'call':
                    print(f"[DEBUG] Пытаюсь выполнить CALL...")
                    do_action('call')
                    print(f"....make BUY (экспирация: {expiry_time} мин)")
            else:
                if not trading_allowed:
                    print("[DEBUG] Сигнал проигнорирован: торговля не разрешена")
                elif last_signal['signal'] == 'neutral':
                    print("[DEBUG] Сигнал проигнорирован: нейтральный сигнал")
                elif signal_strength < threshold:
                    print(f"[DEBUG] Сигнал проигнорирован: недостаточная сила ({signal_strength:.2f} < {threshold})")
            
        print(quotes[-1].date, 'working...')

    except Exception as e:
        print(f"Ошибка в check_data: {e}")

def websocket_log():
    global CURRENCY, CURRENCY_CHANGE, CURRENCY_CHANGE_DATE, PERIOD, CANDLES
    try:
        current_symbol = driver.find_element(by=By.CLASS_NAME, value='current-symbol').text
        if current_symbol != CURRENCY:
            CURRENCY = current_symbol
            CURRENCY_CHANGE = True
            CURRENCY_CHANGE_DATE = datetime.now()
    except:
        pass

    if CURRENCY_CHANGE and CURRENCY_CHANGE_DATE < datetime.now() - timedelta(seconds=5):
        driver.refresh()
        CURRENCY_CHANGE = False
        CANDLES = []
        PERIOD = 0

    global INITIAL_DEPOSIT

    try:
        deposit = driver.find_element(By.CSS_SELECTOR, value='body > div.wrapper > div.wrapper__top > header > div.right-block.js-right-block > div.right-block__item.js-drop-down-modal-open > div > div.balance-info-block__data > div.balance-info-block__balance > span')
        deposit = float(deposit.text.replace(',', ''))
    except Exception as e:
        deposit = None
        print(e)    

    INITIAL_DEPOSIT = deposit   

    for wsData in driver.get_log('performance'):
        message = json.loads(wsData['message'])['message']
        response = message.get('params', {}).get('response', {})
        if response.get('opcode', 0) == 2 and not CURRENCY_CHANGE:
            payload_str = base64.b64decode(response['payloadData']).decode('utf-8')
            data = json.loads(payload_str)
            if 'asset' in data and 'candles' in data:
                PERIOD = data['period']
                CANDLES = list(reversed(data['candles']))
                CANDLES.append([CANDLES[-1][0] + PERIOD, CANDLES[-1][1], CANDLES[-1][2], CANDLES[-1][3], CANDLES[-1][4]])
                for tstamp, value in data['history']:
                    tstamp = int(float(tstamp))
                    CANDLES[-1][2] = value
                    if value > CANDLES[-1][3]:
                        CANDLES[-1][3] = value
                    elif value < CANDLES[-1][4]:
                        CANDLES[-1][4] = value
                    if tstamp % PERIOD == 0:
                        if tstamp not in [c[0] for c in CANDLES]:
                            CANDLES.append([tstamp, value, value, value, value])
                print('Got', len(CANDLES), 'candles for', data['asset'])
            try:
                current_value = data[0][2]
                CANDLES[-1][2] = current_value
                if current_value > CANDLES[-1][3]:
                    CANDLES[-1][3] = current_value
                elif current_value < CANDLES[-1][4]:
                    CANDLES[-1][4] = current_value
                tstamp = int(float(data[0][1]))
                if tstamp % PERIOD == 0:
                    if tstamp not in [c[0] for c in CANDLES]:
                        CANDLES.append([tstamp, current_value, current_value, current_value, current_value])
                        try:
                            check_data()
                        except Exception as e:
                            print(e)
                        # CANDLES.append([tstamp, current_value, current_value, current_value, current_value])
            except:
                pass


if __name__ == '__main__':
    print("[DEBUG] Запуск скрипта бинарных опционов")
    print(f"[DEBUG] MAX_ACTIONS={MAX_ACTIONS} (увеличено для снятия ограничений)")
    print(f"[DEBUG] OPTIMIZATION_ENABLED={OPTIMIZATION_ENABLED}")
    print(f"[DEBUG] Ручные параметры: lookback={MANUAL_LOOKBACK}, threshold={MANUAL_THRESHOLD}")
    load_web_driver()
    from time import sleep
    while True:
        websocket_log()
        sleep(0.1)