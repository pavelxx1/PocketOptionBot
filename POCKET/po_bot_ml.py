import os
import base64
import json
import numpy as np
from datetime import datetime, timedelta
import pandas as pd
import time
from selenium.webdriver.common.by import By
from driver import get_driver
from utils import get_quotes, get_value
import itertools
import random
from joblib import Parallel, delayed
from multiprocessing import cpu_count

BASE_URL = 'https://pocketoption.com'
PERIOD = 0
CANDLES = []
ACTIONS = {}
ACTIONS_SECONDS = PERIOD
CURRENCY = None
CURRENCY_CHANGE = False
INITIAL_DEPOSIT = None
CURRENCY_CHANGE_DATE = datetime.now()

# Глобальные параметры для ручной настройки
OPTIMIZATION_ENABLED = 1
MANUAL_LOOKBACK = 17
MANUAL_THRESHOLD = 0.399

# Новые параметры для периодической оптимизации
OPTIMIZATION_PERIOD = 1  # Оптимизация каждые N свечей
CANDLES_SINCE_LAST_OPTIMIZATION = 0
LAST_OPTIMIZATION_TIME = None
SKIP_TRADE_AFTER_OPTIMIZATION = False
BEST_LOOKBACK = MANUAL_LOOKBACK
BEST_THRESHOLD = MANUAL_THRESHOLD
FIRST_RUN = True  # Флаг для обязательной первой оптимизации
SUCCESS_RATE = 0.69  # Минимальный винрейт для разрешения торговли
MIN_TRADES_FOR_OPTIMIZATION = 5  # Минимальное количество сделок для учета при оптимизации
ENABLE_SKIP_AFTER_OPTIMIZATION = False  # Новый параметр: пропускать ли сделку после оптимизации

IS_AMOUNT_SET = False
TRADING_ALLOWED = True  # Глобальный флаг разрешения торговли

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

def prepare_data_for_parallel(quotes):
    """Подготовка данных для параллельной обработки"""
    prepared_data = []
    for q in quotes:
        prepared_data.append({
            'close': float(get_value(q)),
            'open': float(get_value(q, 'open')),
            'high': float(get_value(q, 'high')),
            'low': float(get_value(q, 'low')),
            'date': getattr(q, 'date', datetime.now())
        })
    return prepared_data

def market_microstructure_analysis_parallel(data, lookback=10):
    """Версия market_microstructure_analysis для работы с предобработанными данными"""
    signals = []
    
    for i in range(lookback, len(data)-1):
        window = data[i-lookback+1:i+1]
        
        # Get basic price data
        close = [item['close'] for item in window]
        open_p = [item['open'] for item in window]
        high = [item['high'] for item in window]
        low = [item['low'] for item in window]
        
        # Расчет микроимпульсов
        micro_impulses = [0, 0]  # Первые два элемента без импульса
        for i in range(2, len(close)):
            delta = close[i] - open_p[i]
            prev_delta = close[i-1] - open_p[i-1]
            
            impulse = 0
            if abs(delta) > abs(prev_delta) and (delta * prev_delta > 0):
                impulse = 1 if delta > 0 else -1
            
            micro_impulses.append(impulse)
        
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
        if len(micro_impulses) >= 3:
            # Усиление восходящего импульса
            if micro_impulses[-1] == 1 and micro_impulses[-2] == 1:
                score += 0.11
            # Усиление нисходящего импульса
            elif micro_impulses[-1] == -1 and micro_impulses[-2] == -1:
                score -= 0.11
            
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
            'date': window[-1]['date'],
            'signal': signal,
            'rsi': rsi,
            'stoch': k_percent,
            'price_momentum': price_momentum,
            'volatility_ratio': volatility_ratio,
            'score': score
        })
    
    return signals

def market_microstructure_analysis(quotes, lookback=10):
    """Оригинальная функция анализа для исходных данных quotes"""
    signals = []
    
    for i in range(lookback, len(quotes)-1):
        window = quotes[i-lookback+1:i+1]
        
        # Get basic price data
        close = [float(get_value(q)) for q in window]
        open_p = [float(get_value(q, 'open')) for q in window]
        high = [float(get_value(q, 'high')) for q in window]
        low = [float(get_value(q, 'low')) for q in window]
        
        # Расчет микроимпульсов
        micro_impulses = [0, 0]  # Первые два элемента без импульса
        for i in range(2, len(close)):
            delta = close[i] - open_p[i]
            prev_delta = close[i-1] - open_p[i-1]
            
            impulse = 0
            if abs(delta) > abs(prev_delta) and (delta * prev_delta > 0):
                impulse = 1 if delta > 0 else -1
            
            micro_impulses.append(impulse)
        
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
        if len(micro_impulses) >= 3:
            # Усиление восходящего импульса
            if micro_impulses[-1] == 1 and micro_impulses[-2] == 1:
                score += 0.11
            # Усиление нисходящего импульса
            elif micro_impulses[-1] == -1 and micro_impulses[-2] == -1:
                score -= 0.11
            
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

def test_parameter_set_parallel(params):
    """Функция тестирования параметров для параллельного запуска"""
    lookback, threshold, data = params
    lookback_int = int(lookback)
    wins, losses = 0, 0
    
    future_check = 1
    
    for i in range(lookback_int, len(data)-future_check-1):
        signals = market_microstructure_analysis_parallel(data[:i+1], lookback=lookback_int)
        
        if not signals:
            continue
            
        last_signal = signals[-1]
        signal_strength = abs(last_signal['score'] - 0.5) * 2
        
        if last_signal['signal'] != 'neutral' and signal_strength > threshold:
            current_close = data[i]['close']
            future_close = data[i+future_check]['close']
            
            if last_signal['signal'] == 'call':
                result = 'win' if future_close > current_close else 'loss'
            elif last_signal['signal'] == 'put':
                result = 'win' if future_close < current_close else 'loss'
            else:
                continue
                
            if result == 'win':
                wins += 1
            else:
                losses += 1
    
    total_trades = wins + losses
    winrate = wins / total_trades if total_trades > 0 else 0
    
    return [
        winrate,                  # Метрика для сортировки
        lookback_int,             # lookback
        threshold,                # threshold
        wins,                     # win
        losses,                   # loss
        total_trades,             # total
        (lookback_int, threshold) # параметры для использования в стратегии
    ]

def backtest_and_optimize(quotes, lookback_range=None, strength_thresholds=None):
    if lookback_range is None:
        lookback_range = generate_range(5, 15, 5)
    if strength_thresholds is None:
        strength_thresholds = generate_range(0.15, 0.35, 0.05)
    
    # Подготавливаем данные для параллельной обработки
    prepared_data = prepare_data_for_parallel(quotes)
    
    # Генерируем все комбинации параметров
    combinations = list(itertools.product(lookback_range, strength_thresholds))
    random.shuffle(combinations)
    
    print(f"[DEBUG] Запуск параллельной оптимизации с {len(combinations)} наборами параметров")
    
    # Создаем параметры для параллельной обработки
    params_list = [(lookback, threshold, prepared_data) for lookback, threshold in combinations]
    
    # Запускаем параллельные процессы
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(test_parameter_set_parallel)(params) for params in params_list
    )
    
    # Фильтруем результаты без достаточного количества сделок
    valid_results = [result for result in results if result[5] >= MIN_TRADES_FOR_OPTIMIZATION]
    
    if not valid_results:
        print("[DEBUG] Не удалось найти параметры с достаточным количеством сделок")
        return {'lookback': 10, 'threshold': 0.25, 'winrate': 0.0, 'trades': 0}, []
    
    # Сортируем по винрейту (первый элемент в результате)
    valid_results.sort(key=lambda x: x[0], reverse=True)
    best_result = valid_results[0]
    
    best_lookback, best_threshold = best_result[6]
    best_winrate = best_result[0]
    wins = best_result[3]
    total_trades = best_result[5]
    
    # Создаем список сделок для лучших параметров (для совместимости)
    trades = []
    for i in range(total_trades):
        if i < wins:
            trades.append({'result': 'win'})
        else:
            trades.append({'result': 'loss'})
            
    return {
        'lookback': best_lookback,
        'threshold': best_threshold,
        'winrate': best_winrate,
        'trades': total_trades
    }, trades

def check_data():
    global CANDLES_SINCE_LAST_OPTIMIZATION, SKIP_TRADE_AFTER_OPTIMIZATION, BEST_LOOKBACK, BEST_THRESHOLD, LAST_OPTIMIZATION_TIME
    global FIRST_RUN, TRADING_ALLOWED
    
    try:
        quotes = get_quotes(CANDLES)[-100:]
        
        if len(quotes) < 20:
            print("Недостаточно данных для анализа")
            return
        
        # Используем глобальные оптимизированные параметры или ручные
        lookback = BEST_LOOKBACK
        threshold = BEST_THRESHOLD
        
        # Проверяем необходимость периодической оптимизации или первого запуска
        need_optimization = False
        if OPTIMIZATION_ENABLED and len(quotes) >= 35:
            if FIRST_RUN or CANDLES_SINCE_LAST_OPTIMIZATION >= OPTIMIZATION_PERIOD:
                need_optimization = True
                CANDLES_SINCE_LAST_OPTIMIZATION = 0
                FIRST_RUN = False  # Сбрасываем флаг первого запуска
            
        if need_optimization:
            print("Запуск периодической оптимизации...")
            start_time = time.time()
            
            lookbacks = generate_range(8, 22, 2)
            thresholds = generate_range(0.20, 0.41, 0.05)
            
            print(f"Тестируем lookbacks: {lookbacks}")  
            print(f"Тестируем thresholds: {thresholds}")
            
            best_params, results = backtest_and_optimize(quotes, lookbacks, thresholds)   
            
            # В функции check_data() меняем этот блок:
            if results and len(results) > 0:
                wins = sum(1 for trade in results if trade['result'] == 'win')
                total = len(results)
                winrate = best_params['winrate']
                
                print(f"► Оптимальные параметры: lookback={best_params['lookback']}, threshold={best_params['threshold']}")
                print(f"► Исторический винрейт: {winrate*100:.1f}% ({wins}/{total} сделок)")
                
                # Всегда обновляем параметры на лучшие найденные
                BEST_THRESHOLD = best_params['threshold']
                BEST_LOOKBACK = best_params['lookback']
                
                # Обновляем локальные переменные для использования в текущем анализе
                lookback = BEST_LOOKBACK
                threshold = BEST_THRESHOLD
                
                # Проверяем винрейт только для разрешения торговли
                if winrate >= SUCCESS_RATE and total >= MIN_TRADES_FOR_OPTIMIZATION:
                    TRADING_ALLOWED = True
                    print(f"Торговля разрешена (винрейт > {SUCCESS_RATE*100:.1f}%)")
                else:
                    TRADING_ALLOWED = False
                    print(f"Торговля запрещена (винрейт < {SUCCESS_RATE*100:.1f}% или сделок < {MIN_TRADES_FOR_OPTIMIZATION})")
            else:
                print("Оптимизация не дала результатов, используем текущие параметры")
                
            end_time = time.time()
            optimization_time = end_time - start_time
            LAST_OPTIMIZATION_TIME = datetime.now()
            print(f"[DEBUG] Время оптимизации: {optimization_time:.2f} секунд")
            
            # Пропускаем сделку сразу после оптимизации только если настройка включена
            if ENABLE_SKIP_AFTER_OPTIMIZATION:
                SKIP_TRADE_AFTER_OPTIMIZATION = True
                return
            else:
                SKIP_TRADE_AFTER_OPTIMIZATION = False
                # Продолжаем выполнение для возможной сделки после оптимизации
            
        elif OPTIMIZATION_ENABLED and not need_optimization:
            print(f"До следующей оптимизации осталось {OPTIMIZATION_PERIOD - CANDLES_SINCE_LAST_OPTIMIZATION} свечей")
        
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
            print(f"[DEBUG] 1. Торговля разрешена: {TRADING_ALLOWED}")
            print(f"[DEBUG] 2. Сигнал не нейтральный: {last_signal['signal'] != 'neutral'}")
            print(f"[DEBUG] 3. Сила сигнала {signal_strength:.2f} >= {threshold}: {signal_strength >= threshold}")
            print(f"[DEBUG] 4. Пропуск после оптимизации: {SKIP_TRADE_AFTER_OPTIMIZATION}")
            
            can_trade = TRADING_ALLOWED and last_signal['signal'] != 'neutral' and signal_strength >= threshold and not SKIP_TRADE_AFTER_OPTIMIZATION
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
                if not TRADING_ALLOWED:
                    print("[DEBUG] Сигнал проигнорирован: торговля не разрешена")
                elif last_signal['signal'] == 'neutral':
                    print("[DEBUG] Сигнал проигнорирован: нейтральный сигнал")
                elif signal_strength < threshold:
                    print(f"[DEBUG] Сигнал проигнорирован: недостаточная сила ({signal_strength:.2f} < {threshold})")
                elif SKIP_TRADE_AFTER_OPTIMIZATION:
                    print("[DEBUG] Сигнал проигнорирован: пропуск после оптимизации")
            
            # Сбрасываем флаг пропуска сделки после оптимизации
            SKIP_TRADE_AFTER_OPTIMIZATION = False
            
        print(quotes[-1].date, 'working...')

    except Exception as e:
        print(f"Ошибка в check_data: {e}")

def append_json(data, filename='output.json'):
    # Initialize with new data if file doesn't exist or is empty
    json_data = [data] if not os.path.exists(filename) or os.path.getsize(filename) == 0 else []
    
    # If file exists, read and update data
    if json_data == []:
        try:
            with open(filename, 'r') as f:
                json_data = json.load(f)
                json_data = json_data if isinstance(json_data, list) else [json_data]
                json_data.append(data)
        except:
            json_data = [data]  # Reset if any error occurs
    
    # Write updated data back to the file
    with open(filename, 'w') as f:
        json.dump(json_data, f)        

def websocket_log():
    global CURRENCY, CURRENCY_CHANGE, CURRENCY_CHANGE_DATE, PERIOD, CANDLES, CANDLES_SINCE_LAST_OPTIMIZATION
    global FIRST_RUN
    
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
        CANDLES_SINCE_LAST_OPTIMIZATION = 0
        FIRST_RUN = True  # Сбрасываем флаг при смене валютной пары

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
                # append_json(data)
                PERIOD = data['period']
                CANDLES = list(reversed(data['candles']))
                CANDLES.append([CANDLES[-1][0] + PERIOD, CANDLES[-1][1], CANDLES[-1][2], CANDLES[-1][3], CANDLES[-1][4]])
                # Сбрасываем флаг FIRST_RUN при получении новых данных
                FIRST_RUN = True
                CANDLES_SINCE_LAST_OPTIMIZATION = 0
                
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
                        # Увеличиваем счетчик свечей для периодической оптимизации
                        CANDLES_SINCE_LAST_OPTIMIZATION += 1
                        try:
                            check_data()
                        except Exception as e:
                            print(e)
                        # CANDLES.append([tstamp, current_value, current_value, current_value, current_value])    
            except:
                pass


if __name__ == '__main__':
    # Устанавливаем максимальное количество процессов для параллельной обработки
    num_cores = cpu_count()
    print(f"[DEBUG] Запуск скрипта бинарных опционов (доступно ядер: {num_cores})")
    print(f"[DEBUG] OPTIMIZATION_ENABLED={OPTIMIZATION_ENABLED}")
    print(f"[DEBUG] Ручные параметры: lookback={MANUAL_LOOKBACK}, threshold={MANUAL_THRESHOLD}")
    print(f"[DEBUG] Период оптимизации: каждые {OPTIMIZATION_PERIOD} свечей")
    print(f"[DEBUG] Минимальный винрейт для торговли: {SUCCESS_RATE*100}%")
    print(f"[DEBUG] Минимальное кол-во сделок для оптимизации: {MIN_TRADES_FOR_OPTIMIZATION}")
    print(f"[DEBUG] Обязательная первая оптимизация: {'Да' if FIRST_RUN else 'Нет'}")
    print(f"[DEBUG] Пропуск сделки после оптимизации: {'Да' if ENABLE_SKIP_AFTER_OPTIMIZATION else 'Нет'}")
    print(f"[DEBUG] Параллельная оптимизация: {num_cores} ядер")
    load_web_driver()
    from time import sleep
    while True:
        websocket_log()
        sleep(0.1)