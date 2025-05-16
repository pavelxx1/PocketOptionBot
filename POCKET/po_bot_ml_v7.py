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
import random
from joblib import Parallel, delayed
from multiprocessing import cpu_count
import copy
from scipy.stats import norm
from scipy import signal

BASE_URL = 'https://pocketoption.com'
PERIOD = 0
CANDLES = []
ACTIONS = {}
ACTIONS_SECONDS = PERIOD
CURRENCY = None
CURRENCY_CHANGE = False
INITIAL_DEPOSIT = None
CURRENCY_CHANGE_DATE = datetime.now()

# Глобальные параметры для настройки
OPTIMIZATION_ENABLED = 1
OPTIMIZATION_PERIOD = 1  # Оптимизация каждые N свечей
CANDLES_SINCE_LAST_OPTIMIZATION = 0
LAST_OPTIMIZATION_TIME = None
SKIP_TRADE_AFTER_OPTIMIZATION = False
FIRST_RUN = True
SUCCESS_RATE = 0.54  # Минимальный винрейт для разрешения торговли
MIN_TRADES_FOR_OPTIMIZATION = 4
ENABLE_SKIP_AFTER_OPTIMIZATION = False

# Параметр для раздельной проверки винрейта по направлениям
SEPARATE_DIRECTION_FILTER = True  # Включить фильтрацию по отдельным направлениям

# Параметры для проверки будущих свечей
FUTURE_CHECK_PERIODS = 2  # Количество свечей для проверки
ALL_CANDLES_SHOULD_MATCH = True  # Требовать соответствия всех свечей

IS_AMOUNT_SET = False
TRADING_ALLOWED = True  # Глобальный флаг разрешения торговли

# Отслеживание результатов по направлениям
CALL_ENABLED = True  # Разрешение торговли в направлении CALL
PUT_ENABLED = True   # Разрешение торговли в направлении PUT

# Глобальный порог силы сигнала (будет оптимизирован)
SIGNAL_THRESHOLD = 0.2

# Параметры для оптимизации
POPULATION_SIZE = 30  # Размер популяции для генетического алгоритма
NUM_GENERATIONS = 5   # Количество поколений
MUTATION_RATE = 0.2   # Вероятность мутации
TOP_PARENTS = 5       # Количество лучших решений для скрещивания

# Диапазоны параметров для каждого продвинутого индикатора
INDICATOR_PARAMS = {
    'fisher_transform': {
        'period': [5, 8, 10, 14, 20],
        'threshold': [0.5, 1.0, 1.5, 2.0],
        'weight': [0.1, 0.15, 0.2, 0.25]
    },
    'relative_momentum': {
        'momentum_period': [2, 3, 4, 5],
        'smoothing_period': [2, 3, 4, 5],
        'threshold': [0.1, 0.2, 0.3, 0.4],
        'weight': [0.08, 0.12, 0.16, 0.2]
    },
    'squeeze_momentum': {
        'bb_period': [10, 15, 20],
        'bb_mult': [1.5, 2.0, 2.5],
        'kc_period': [10, 15, 20],
        'kc_mult': [1.0, 1.5, 2.0],
        'weight': [0.1, 0.15, 0.2]
    },
    'market_facilitation': {
        'period': [5, 10, 15],
        'threshold': [0.5, 1.0, 1.5],
        'weight': [0.05, 0.1, 0.15]
    },
    'vwap_analysis': {
        'period': [10, 20, 30],
        'std_dev': [1.0, 1.5, 2.0],
        'weight': [0.12, 0.16, 0.2]
    },
    'fractal_analysis': {
        'period': [3, 5, 7],
        'weight': [0.08, 0.12, 0.16]
    },
    'swing_strength': {
        'period': [5, 10, 15],
        'threshold': [0.3, 0.5, 0.7],
        'weight': [0.1, 0.15, 0.2]
    }
}

# Диапазон для оптимизации порога сигнала
STRATEGY_PARAMS = {
    'signal_threshold': [0.1, 0.15, 0.2, 0.25, 0.3, 0.35, 0.4]
}

# Класс для хранения параметров стратегии
class StrategyParameters:
    def __init__(self):
        # Значения по умолчанию
        self.fisher_transform = {'period': 10, 'threshold': 1.0, 'weight': 0.15}
        self.relative_momentum = {'momentum_period': 3, 'smoothing_period': 3, 'threshold': 0.2, 'weight': 0.12}
        self.squeeze_momentum = {'bb_period': 15, 'bb_mult': 2.0, 'kc_period': 15, 'kc_mult': 1.5, 'weight': 0.15}
        self.market_facilitation = {'period': 10, 'threshold': 1.0, 'weight': 0.1}
        self.vwap_analysis = {'period': 20, 'std_dev': 1.5, 'weight': 0.16}
        self.fractal_analysis = {'period': 5, 'weight': 0.12}
        self.swing_strength = {'period': 10, 'threshold': 0.5, 'weight': 0.15}
        self.signal_threshold = 0.2
        
    def to_dict(self):
        return {
            'fisher_transform': self.fisher_transform,
            'relative_momentum': self.relative_momentum,
            'squeeze_momentum': self.squeeze_momentum,
            'market_facilitation': self.market_facilitation,
            'vwap_analysis': self.vwap_analysis,
            'fractal_analysis': self.fractal_analysis,
            'swing_strength': self.swing_strength,
            'signal_threshold': self.signal_threshold
        }
    
    @staticmethod
    def create_random():
        """Создание случайного набора параметров"""
        params = StrategyParameters()
        
        # Fisher Transform
        params.fisher_transform['period'] = random.choice(INDICATOR_PARAMS['fisher_transform']['period'])
        params.fisher_transform['threshold'] = random.choice(INDICATOR_PARAMS['fisher_transform']['threshold'])
        params.fisher_transform['weight'] = random.choice(INDICATOR_PARAMS['fisher_transform']['weight'])
        
        # Relative Momentum
        params.relative_momentum['momentum_period'] = random.choice(INDICATOR_PARAMS['relative_momentum']['momentum_period'])
        params.relative_momentum['smoothing_period'] = random.choice(INDICATOR_PARAMS['relative_momentum']['smoothing_period'])
        params.relative_momentum['threshold'] = random.choice(INDICATOR_PARAMS['relative_momentum']['threshold'])
        params.relative_momentum['weight'] = random.choice(INDICATOR_PARAMS['relative_momentum']['weight'])
        
        # Squeeze Momentum
        params.squeeze_momentum['bb_period'] = random.choice(INDICATOR_PARAMS['squeeze_momentum']['bb_period'])
        params.squeeze_momentum['bb_mult'] = random.choice(INDICATOR_PARAMS['squeeze_momentum']['bb_mult'])
        params.squeeze_momentum['kc_period'] = random.choice(INDICATOR_PARAMS['squeeze_momentum']['kc_period'])
        params.squeeze_momentum['kc_mult'] = random.choice(INDICATOR_PARAMS['squeeze_momentum']['kc_mult'])
        params.squeeze_momentum['weight'] = random.choice(INDICATOR_PARAMS['squeeze_momentum']['weight'])
        
        # Market Facilitation
        params.market_facilitation['period'] = random.choice(INDICATOR_PARAMS['market_facilitation']['period'])
        params.market_facilitation['threshold'] = random.choice(INDICATOR_PARAMS['market_facilitation']['threshold'])
        params.market_facilitation['weight'] = random.choice(INDICATOR_PARAMS['market_facilitation']['weight'])
        
        # VWAP Analysis
        params.vwap_analysis['period'] = random.choice(INDICATOR_PARAMS['vwap_analysis']['period'])
        params.vwap_analysis['std_dev'] = random.choice(INDICATOR_PARAMS['vwap_analysis']['std_dev'])
        params.vwap_analysis['weight'] = random.choice(INDICATOR_PARAMS['vwap_analysis']['weight'])
        
        # Fractal Analysis
        params.fractal_analysis['period'] = random.choice(INDICATOR_PARAMS['fractal_analysis']['period'])
        params.fractal_analysis['weight'] = random.choice(INDICATOR_PARAMS['fractal_analysis']['weight'])
        
        # Swing Strength
        params.swing_strength['period'] = random.choice(INDICATOR_PARAMS['swing_strength']['period'])
        params.swing_strength['threshold'] = random.choice(INDICATOR_PARAMS['swing_strength']['threshold'])
        params.swing_strength['weight'] = random.choice(INDICATOR_PARAMS['swing_strength']['weight'])
        
        # Signal Threshold
        params.signal_threshold = random.choice(STRATEGY_PARAMS['signal_threshold'])
        
        return params
    
    @staticmethod
    def crossover(parent1, parent2):
        """Создание нового набора параметров путем скрещивания двух родителей"""
        child = StrategyParameters()
        
        # Для каждого индикатора и его параметров берем значение у одного из родителей
        indicators = [
            'fisher_transform', 'relative_momentum', 'squeeze_momentum', 
            'market_facilitation', 'vwap_analysis', 'fractal_analysis', 
            'swing_strength'
        ]
        
        for indicator in indicators:
            # С вероятностью 50% берем параметры от первого родителя, иначе от второго
            if random.random() < 0.5:
                setattr(child, indicator, copy.deepcopy(getattr(parent1, indicator)))
            else:
                setattr(child, indicator, copy.deepcopy(getattr(parent2, indicator)))
        
        # Порог сигнала
        child.signal_threshold = parent1.signal_threshold if random.random() < 0.5 else parent2.signal_threshold
        
        return child
    
    def mutate(self):
        """Мутация некоторых параметров с заданной вероятностью"""
        # Fisher Transform
        if random.random() < MUTATION_RATE:
            self.fisher_transform['period'] = random.choice(INDICATOR_PARAMS['fisher_transform']['period'])
        if random.random() < MUTATION_RATE:
            self.fisher_transform['threshold'] = random.choice(INDICATOR_PARAMS['fisher_transform']['threshold'])
        if random.random() < MUTATION_RATE:
            self.fisher_transform['weight'] = random.choice(INDICATOR_PARAMS['fisher_transform']['weight'])
        
        # Relative Momentum
        if random.random() < MUTATION_RATE:
            self.relative_momentum['momentum_period'] = random.choice(INDICATOR_PARAMS['relative_momentum']['momentum_period'])
        if random.random() < MUTATION_RATE:
            self.relative_momentum['smoothing_period'] = random.choice(INDICATOR_PARAMS['relative_momentum']['smoothing_period'])
        if random.random() < MUTATION_RATE:
            self.relative_momentum['threshold'] = random.choice(INDICATOR_PARAMS['relative_momentum']['threshold'])
        if random.random() < MUTATION_RATE:
            self.relative_momentum['weight'] = random.choice(INDICATOR_PARAMS['relative_momentum']['weight'])
        
        # Squeeze Momentum
        if random.random() < MUTATION_RATE:
            self.squeeze_momentum['bb_period'] = random.choice(INDICATOR_PARAMS['squeeze_momentum']['bb_period'])
        if random.random() < MUTATION_RATE:
            self.squeeze_momentum['bb_mult'] = random.choice(INDICATOR_PARAMS['squeeze_momentum']['bb_mult'])
        if random.random() < MUTATION_RATE:
            self.squeeze_momentum['kc_period'] = random.choice(INDICATOR_PARAMS['squeeze_momentum']['kc_period'])
        if random.random() < MUTATION_RATE:
            self.squeeze_momentum['kc_mult'] = random.choice(INDICATOR_PARAMS['squeeze_momentum']['kc_mult'])
        if random.random() < MUTATION_RATE:
            self.squeeze_momentum['weight'] = random.choice(INDICATOR_PARAMS['squeeze_momentum']['weight'])
        
        # Market Facilitation
        if random.random() < MUTATION_RATE:
            self.market_facilitation['period'] = random.choice(INDICATOR_PARAMS['market_facilitation']['period'])
        if random.random() < MUTATION_RATE:
            self.market_facilitation['threshold'] = random.choice(INDICATOR_PARAMS['market_facilitation']['threshold'])
        if random.random() < MUTATION_RATE:
            self.market_facilitation['weight'] = random.choice(INDICATOR_PARAMS['market_facilitation']['weight'])
        
        # VWAP Analysis
        if random.random() < MUTATION_RATE:
            self.vwap_analysis['period'] = random.choice(INDICATOR_PARAMS['vwap_analysis']['period'])
        if random.random() < MUTATION_RATE:
            self.vwap_analysis['std_dev'] = random.choice(INDICATOR_PARAMS['vwap_analysis']['std_dev'])
        if random.random() < MUTATION_RATE:
            self.vwap_analysis['weight'] = random.choice(INDICATOR_PARAMS['vwap_analysis']['weight'])
        
        # Fractal Analysis
        if random.random() < MUTATION_RATE:
            self.fractal_analysis['period'] = random.choice(INDICATOR_PARAMS['fractal_analysis']['period'])
        if random.random() < MUTATION_RATE:
            self.fractal_analysis['weight'] = random.choice(INDICATOR_PARAMS['fractal_analysis']['weight'])
        
        # Swing Strength
        if random.random() < MUTATION_RATE:
            self.swing_strength['period'] = random.choice(INDICATOR_PARAMS['swing_strength']['period'])
        if random.random() < MUTATION_RATE:
            self.swing_strength['threshold'] = random.choice(INDICATOR_PARAMS['swing_strength']['threshold'])
        if random.random() < MUTATION_RATE:
            self.swing_strength['weight'] = random.choice(INDICATOR_PARAMS['swing_strength']['weight'])
        
        # Signal Threshold
        if random.random() < MUTATION_RATE:
            self.signal_threshold = random.choice(STRATEGY_PARAMS['signal_threshold'])

# Глобальный экземпляр оптимизированных параметров
OPTIMIZED_PARAMS = StrategyParameters()

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

def calculate_fisher_transform(prices, period=10):
    """Расчет Fisher Transform - преобразует цены в нормальное распределение"""
    if len(prices) < period:
        return 0
    
    # Находим максимальную и минимальную цену за период
    price_period = prices[-period:]
    min_price = min(price_period)
    max_price = max(price_period)
    
    # Нормализация цены в диапазоне [-1, 1]
    range_size = max_price - min_price
    if range_size == 0:
        return 0
    
    normalized = (2 * ((prices[-1] - min_price) / range_size)) - 1
    
    # Ограничиваем значение между -0.999 и 0.999 для избежания бесконечностей
    normalized = max(-0.999, min(0.999, normalized))
    
    # Применяем обратное гиперболическое преобразование
    fisher = 0.5 * np.log((1 + normalized) / (1 - normalized))
    
    return fisher

def calculate_relative_momentum(prices, momentum_period=3, smoothing_period=3):
    """Расчет Relative Momentum Index (RMI) - вариант RSI с учетом скорости изменения"""
    if len(prices) < momentum_period + smoothing_period + 1:
        return 50
    
    # Вычисляем моментум - разницу между текущей ценой и ценой momentum_period назад
    momentum_values = [prices[i] - prices[i-momentum_period] for i in range(momentum_period, len(prices))]
    
    # Вычисляем скользящее среднее положительного и отрицательного моментума
    if len(momentum_values) < smoothing_period:
        return 50
    
    up_momentum = [max(0, m) for m in momentum_values]
    down_momentum = [max(0, -m) for m in momentum_values]
    
    avg_up_momentum = sum(up_momentum[-smoothing_period:]) / smoothing_period
    avg_down_momentum = sum(down_momentum[-smoothing_period:]) / smoothing_period
    
    if avg_down_momentum == 0:
        return 100
    
    rmi = 100 - (100 / (1 + (avg_up_momentum / avg_down_momentum)))
    
    return rmi

def calculate_squeeze_momentum(data, bb_period=15, bb_mult=2.0, kc_period=15, kc_mult=1.5):
    """Расчет Squeeze Momentum Indicator (SMI) - определяет сжатие и расширение волатильности"""
    if len(data) < bb_period:
        return 0, False
    
    # Извлекаем цены закрытия, максимумы и минимумы
    close = [item['close'] for item in data]
    high = [item['high'] for item in data]
    low = [item['low'] for item in data]
    
    # Вычисляем Bollinger Bands
    sma = sum(close[-bb_period:]) / bb_period
    std_dev = np.std(close[-bb_period:])
    bb_upper = sma + (bb_mult * std_dev)
    bb_lower = sma - (bb_mult * std_dev)
    
    # Вычисляем Keltner Channel
    tr_values = []
    for i in range(1, len(data)):
        tr = max(high[i] - low[i], abs(high[i] - close[i-1]), abs(low[i] - close[i-1]))
        tr_values.append(tr)
    
    atr = sum(tr_values[-kc_period:]) / kc_period
    kc_upper = sma + (kc_mult * atr)
    kc_lower = sma - (kc_mult * atr)
    
    # Определяем сжатие (когда BB внутри KC)
    squeeze = (bb_upper <= kc_upper) and (bb_lower >= kc_lower)
    
    # Вычисляем моментум (разница между текущей ценой и средней из высокой и низкой)
    highest_high = max(high[-bb_period:])
    lowest_low = min(low[-bb_period:])
    
    # Защита от деления на ноль
    denominator = highest_high - lowest_low
    if denominator == 0:
        return 0, squeeze
    
    momentum = (close[-1] - ((highest_high + lowest_low) / 2)) / denominator * 100
    
    return momentum, squeeze

def calculate_market_facilitation(data, period=10):
    """Расчет Market Facilitation Index - определяет эффективность движения цены"""
    if len(data) < 2:
        return 0
    
    # Извлекаем данные о цене и объеме
    high = [item['high'] for item in data]
    low = [item['low'] for item in data]
    
    # Поскольку объемов нет, используем приближение на основе диапазона цены и времени
    # Реальный MFI требует объемов, мы делаем адаптацию для бинарных опционов
    
    # Вычисляем диапазон цены
    ranges = [high[i] - low[i] for i in range(len(high))]
    
    # Вычисляем изменение диапазона как "псевдообъем"
    range_changes = [abs(ranges[i] - ranges[i-1]) for i in range(1, len(ranges))]
    
    if not range_changes:
        return 0
    
    # Берем среднее за период
    if len(range_changes) < period:
        avg_range_change = sum(range_changes) / len(range_changes)
    else:
        avg_range_change = sum(range_changes[-period:]) / period
    
    # Вычисляем MFI как отношение текущего диапазона к среднему изменению
    if avg_range_change == 0:
        return 0
    
    mfi = ranges[-1] / avg_range_change
    
    return mfi

def calculate_vwap_analysis(data, period=20):
    """Анализ на основе VWAP (Volume Weighted Average Price) с адаптацией для бинарных опционов"""
    if len(data) < period:
        return 0
    
    # Извлекаем данные
    close = [item['close'] for item in data]
    high = [item['high'] for item in data]
    low = [item['low'] for item in data]
    
    # Вычисляем типичную цену для каждой свечи (без объемов)
    typical_prices = [(high[i] + low[i] + close[i]) / 3 for i in range(len(close))]
    
    # Берем период для VWAP
    tp_period = typical_prices[-period:]
    
    # Вычисляем VWAP (в данном случае просто среднее типичных цен)
    vwap = sum(tp_period) / period
    
    # Вычисляем отклонение текущей цены от VWAP
    # Защита от деления на ноль
    if vwap == 0:
        return 0
    
    deviation = (close[-1] - vwap) / vwap * 100
    
    return deviation

def identify_fractals(high, low, period=5):
    """Определение фракталов Вильямса для определения точек разворота"""
    if len(high) < 2*period + 1:
        return None, None
    
    up_fractals = []
    down_fractals = []
    
    # Ищем фракталы вверх (пик)
    for i in range(period, len(high)-period):
        is_up_fractal = True
        for j in range(1, period+1):
            if high[i] <= high[i-j] or high[i] <= high[i+j]:
                is_up_fractal = False
                break
        
        if is_up_fractal:
            up_fractals.append(i)
    
    # Ищем фракталы вниз (впадина)
    for i in range(period, len(low)-period):
        is_down_fractal = True
        for j in range(1, period+1):
            if low[i] >= low[i-j] or low[i] >= low[i+j]:
                is_down_fractal = False
                break
        
        if is_down_fractal:
            down_fractals.append(i)
    
    # Возвращаем последний фрактал вверх и вниз, если есть
    last_up = up_fractals[-1] if up_fractals else None
    last_down = down_fractals[-1] if down_fractals else None
    
    return last_up, last_down

def calculate_swing_strength(data, period=10):
    """Расчет силы колебания цены - отношение последнего движения к среднему"""
    if len(data) < period + 1:
        return 0
    
    # Извлекаем цены закрытия
    close = [item['close'] for item in data]
    
    # Вычисляем последнее изменение
    last_change = abs(close[-1] - close[-2])
    
    # Вычисляем среднее изменение за период
    changes = [abs(close[i] - close[i-1]) for i in range(1, len(close))]
    avg_change = sum(changes[-period:]) / period
    
    # Вычисляем силу колебания
    if avg_change == 0:
        return 0
    
    swing_strength = last_change / avg_change
    
    return swing_strength

def market_microstructure_analysis_parallel(data, params):
    """Анализ рынка с продвинутыми индикаторами и заданными параметрами"""
    signals = []
    
    # Определяем максимальный период для всех индикаторов
    max_lookback = max(
        params.fisher_transform['period'],
        params.relative_momentum['momentum_period'] + params.relative_momentum['smoothing_period'],
        params.squeeze_momentum['bb_period'],
        params.market_facilitation['period'],
        params.vwap_analysis['period'],
        2 * params.fractal_analysis['period'] + 1,
        params.swing_strength['period'] + 1
    )
    
    # Если данных меньше максимального периода, возвращаем пустой список
    if len(data) < max_lookback + 1:
        return signals
    
    for i in range(max_lookback, len(data)-1):
        window = data[i-max_lookback+1:i+1]
        
        # Получаем базовые данные цен
        close = [item['close'] for item in window]
        open_p = [item['open'] for item in window]
        high = [item['high'] for item in window]
        low = [item['low'] for item in window]
        
        # 1. Fisher Transform
        fisher = calculate_fisher_transform(close, period=params.fisher_transform['period'])
        
        # 2. Relative Momentum Index
        rmi = calculate_relative_momentum(
            close, 
            momentum_period=params.relative_momentum['momentum_period'], 
            smoothing_period=params.relative_momentum['smoothing_period']
        )
        
        # 3. Squeeze Momentum Indicator
        squeeze_momentum, is_squeeze = calculate_squeeze_momentum(
            window, 
            bb_period=params.squeeze_momentum['bb_period'], 
            bb_mult=params.squeeze_momentum['bb_mult'],
            kc_period=params.squeeze_momentum['kc_period'],
            kc_mult=params.squeeze_momentum['kc_mult']
        )
        
        # 4. Market Facilitation Index
        mfi = calculate_market_facilitation(window, period=params.market_facilitation['period'])
        
        # 5. VWAP Анализ
        vwap_dev = calculate_vwap_analysis(window, period=params.vwap_analysis['period'])
        
        # 6. Фрактальный Анализ
        last_up_fractal, last_down_fractal = identify_fractals(
            high, low, period=params.fractal_analysis['period']
        )
        
        fractal_signal = 0
        if last_up_fractal is not None and last_down_fractal is not None:
            # Последний фрактал - вверх (потенциально нисходящий тренд)
            if last_up_fractal > last_down_fractal:
                fractal_signal = -1
            # Последний фрактал - вниз (потенциально восходящий тренд)
            else:
                fractal_signal = 1
        
        # 7. Сила колебания цены
        swing = calculate_swing_strength(window, period=params.swing_strength['period'])
        
        # Вычисление скора на основе продвинутых индикаторов с весами
        score = 0.5
        
        # Сигналы на повышение с весами из параметров
        if fisher > params.fisher_transform['threshold']:
            score += params.fisher_transform['weight']
        if rmi < 30:
            score += params.relative_momentum['weight']
        if squeeze_momentum > params.relative_momentum['threshold'] and is_squeeze:
            score += params.squeeze_momentum['weight']
        if mfi > params.market_facilitation['threshold'] and close[-1] > open_p[-1]:
            score += params.market_facilitation['weight']
        if vwap_dev < -params.vwap_analysis['std_dev']:
            score += params.vwap_analysis['weight']  # Цена ниже VWAP - потенциал для роста
        if fractal_signal == 1:
            score += params.fractal_analysis['weight']
        if swing > params.swing_strength['threshold'] and close[-1] > close[-2]:
            score += params.swing_strength['weight']
        
        # Сигналы на понижение с весами
        if fisher < -params.fisher_transform['threshold']:
            score -= params.fisher_transform['weight']
        if rmi > 70:
            score -= params.relative_momentum['weight']
        if squeeze_momentum < -params.relative_momentum['threshold'] and is_squeeze:
            score -= params.squeeze_momentum['weight']
        if mfi > params.market_facilitation['threshold'] and close[-1] < open_p[-1]:
            score -= params.market_facilitation['weight']
        if vwap_dev > params.vwap_analysis['std_dev']:
            score -= params.vwap_analysis['weight']  # Цена выше VWAP - потенциал для падения
        if fractal_signal == -1:
            score -= params.fractal_analysis['weight']
        if swing > params.swing_strength['threshold'] and close[-1] < close[-2]:
            score -= params.swing_strength['weight']
        
        # Определение типа сигнала
        signal = 'call' if score > 0.55 else ('put' if score < 0.45 else 'neutral')
        
        signals.append({
            'date': window[-1]['date'],
            'signal': signal,
            'fisher': fisher,
            'rmi': rmi,
            'squeeze_momentum': squeeze_momentum,
            'is_squeeze': is_squeeze,
            'mfi': mfi,
            'vwap_dev': vwap_dev,
            'fractal_signal': fractal_signal,
            'swing': swing,
            'score': score
        })
    
    return signals

def test_strategy_parameters(params, data, min_trades=10):
    """Тестирование стратегии с заданными параметрами"""
    wins, losses = 0, 0
    future_check = FUTURE_CHECK_PERIODS
    
    # Определяем максимальный период для анализа
    max_lookback = max(
        params.fisher_transform['period'],
        params.relative_momentum['momentum_period'] + params.relative_momentum['smoothing_period'],
        params.squeeze_momentum['bb_period'],
        params.market_facilitation['period'],
        params.vwap_analysis['period'],
        2 * params.fractal_analysis['period'] + 1,
        params.swing_strength['period'] + 1
    )
    
    # Тестируем на исторических данных
    for i in range(max_lookback, len(data)-future_check-1):
        signals = market_microstructure_analysis_parallel(data[:i+1], params)
        
        if not signals:
            continue
            
        last_signal = signals[-1]
        signal_strength = abs(last_signal['score'] - 0.5) * 2
        
        if last_signal['signal'] != 'neutral' and signal_strength >= params.signal_threshold:
            current_close = data[i]['close']
            
            # Проверка всех свечей в диапазоне
            if ALL_CANDLES_SHOULD_MATCH:
                # Для CALL проверяем, что все свечи выше
                if last_signal['signal'] == 'call':
                    win = True
                    for j in range(1, future_check+1):
                        if data[i+j]['close'] <= current_close:
                            win = False
                            break
                    result = 'win' if win else 'loss'
                # Для PUT проверяем, что все свечи ниже
                elif last_signal['signal'] == 'put':
                    win = True
                    for j in range(1, future_check+1):
                        if data[i+j]['close'] >= current_close:
                            win = False
                            break
                    result = 'win' if win else 'loss'
            else:
                # Проверка только последней свечи в диапазоне
                future_close = data[i+future_check]['close']
                if last_signal['signal'] == 'call':
                    result = 'win' if future_close > current_close else 'loss'
                elif last_signal['signal'] == 'put':
                    result = 'win' if future_close < current_close else 'loss'
                    
            if result == 'win':
                wins += 1
            else:
                losses += 1
    
    total_trades = wins + losses
    winrate = wins / total_trades if total_trades > 0 else 0
    
    return {
        'params': params,
        'winrate': winrate,
        'wins': wins,
        'losses': losses,
        'total': total_trades
    }

def evaluate_population(population, data, min_trades=10):
    """Оценка всей популяции параметров"""
    results = Parallel(n_jobs=-1, backend="loky")(
        delayed(test_strategy_parameters)(params, data, min_trades) for params in population
    )
    
    # Фильтруем результаты с достаточным количеством сделок
    valid_results = [result for result in results if result['total'] >= min_trades]
    
    # Если нет валидных результатов, возвращаем исходные (даже с малым количеством сделок)
    if not valid_results:
        print(f"[WARN] Не найдено параметров с достаточным количеством сделок (>= {min_trades})")
        valid_results = results
    
    # Сортируем по винрейту
    valid_results.sort(key=lambda x: x['winrate'], reverse=True)
    
    return valid_results

def optimize_strategy_parameters(quotes, min_trades=10):
    """Оптимизация всех параметров стратегии с помощью генетического алгоритма"""
    global OPTIMIZED_PARAMS, SIGNAL_THRESHOLD
    
    print("Запуск комплексной оптимизации параметров стратегии...")
    prepared_data = prepare_data_for_parallel(quotes)
    
    # Создаем начальную популяцию из случайных наборов параметров
    population = [StrategyParameters.create_random() for _ in range(POPULATION_SIZE)]
    
    best_result = None
    
    # Эволюция на протяжении нескольких поколений
    for generation in range(NUM_GENERATIONS):
        print(f"Оптимизация: поколение {generation+1}/{NUM_GENERATIONS}")
        
        # Оценка текущей популяции
        results = evaluate_population(population, prepared_data, min_trades)
        
        # Сохраняем лучший результат всех поколений
        if results and (best_result is None or results[0]['winrate'] > best_result['winrate']):
            best_result = results[0]
        
        # В последнем поколении не создаем новую популяцию
        if generation == NUM_GENERATIONS - 1:
            break
        
        # Отбор лучших особей для размножения
        parents = [result['params'] for result in results[:TOP_PARENTS]] if results else population[:TOP_PARENTS]
        
        # Создание новой популяции
        new_population = []
        
        # Элитизм - сохраняем лучших родителей
        new_population.extend(parents[:2])
        
        # Заполняем остальную популяцию потомками
        while len(new_population) < POPULATION_SIZE:
            # Выбираем двух случайных родителей из лучших
            parent1 = random.choice(parents)
            parent2 = random.choice(parents)
            
            # Создаем нового потомка скрещиванием и мутацией
            child = StrategyParameters.crossover(parent1, parent2)
            child.mutate()
            
            new_population.append(child)
        
        population = new_population
    
    # Если не удалось найти хороший набор параметров
    if best_result is None:
        print("[WARN] Не удалось найти оптимальные параметры, используем настройки по умолчанию")
        return OPTIMIZED_PARAMS, 0
    
    # Обновляем глобальные параметры лучшим набором
    OPTIMIZED_PARAMS = best_result['params']
    SIGNAL_THRESHOLD = best_result['params'].signal_threshold
    
    print("\n=== ЛУЧШИЕ НАЙДЕННЫЕ ПАРАМЕТРЫ ===")
    for indicator, params in OPTIMIZED_PARAMS.to_dict().items():
        if indicator != 'signal_threshold':
            print(f"► {indicator}: {params}")
    print(f"► Порог силы сигнала: {SIGNAL_THRESHOLD}")
    print(f"► Винрейт: {best_result['winrate']*100:.1f}% ({best_result['wins']}/{best_result['total']} сделок)")
    
    return OPTIMIZED_PARAMS, best_result['winrate']

def get_detailed_strategy_results(quotes, params):
    """Получение детальных результатов стратегии с заданными параметрами"""
    prepared_data = prepare_data_for_parallel(quotes)
    
    wins, losses = 0, 0
    future_check = FUTURE_CHECK_PERIODS
    trades_data = []  # Для сохранения данных о сделках
    
    # Определяем максимальный период для анализа
    max_lookback = max(
        params.fisher_transform['period'],
        params.relative_momentum['momentum_period'] + params.relative_momentum['smoothing_period'],
        params.squeeze_momentum['bb_period'],
        params.market_facilitation['period'],
        params.vwap_analysis['period'],
        2 * params.fractal_analysis['period'] + 1,
        params.swing_strength['period'] + 1
    )
    
    # Тестируем на исторических данных
    for i in range(max_lookback, len(prepared_data)-future_check-1):
        signals = market_microstructure_analysis_parallel(prepared_data[:i+1], params)
        
        if not signals:
            continue
            
        last_signal = signals[-1]
        signal_strength = abs(last_signal['score'] - 0.5) * 2
        
        if last_signal['signal'] != 'neutral' and signal_strength >= params.signal_threshold:
            current_close = prepared_data[i]['close']
            trade_data = {
                'date': prepared_data[i]['date'],
                'signal': last_signal['signal'],
                'score': last_signal['score'],
                'price': current_close,
                'fisher': last_signal['fisher'],
                'rmi': last_signal['rmi'],
                'squeeze': last_signal['squeeze_momentum'],
                'vwap_dev': last_signal['vwap_dev']
            }
            
            # Проверка всех свечей в диапазоне
            if ALL_CANDLES_SHOULD_MATCH:
                # Для CALL проверяем, что все свечи выше
                if last_signal['signal'] == 'call':
                    win = True
                    for j in range(1, future_check+1):
                        if prepared_data[i+j]['close'] <= current_close:
                            win = False
                            break
                    result = 'win' if win else 'loss'
                # Для PUT проверяем, что все свечи ниже
                elif last_signal['signal'] == 'put':
                    win = True
                    for j in range(1, future_check+1):
                        if prepared_data[i+j]['close'] >= current_close:
                            win = False
                            break
                    result = 'win' if win else 'loss'
            else:
                # Проверка только последней свечи в диапазоне
                future_close = prepared_data[i+future_check]['close']
                if last_signal['signal'] == 'call':
                    result = 'win' if future_close > current_close else 'loss'
                elif last_signal['signal'] == 'put':
                    result = 'win' if future_close < current_close else 'loss'
            
            trade_data['result'] = result
            trades_data.append(trade_data)
                
            if result == 'win':
                wins += 1
            else:
                losses += 1
    
    total_trades = wins + losses
    winrate = wins / total_trades if total_trades > 0 else 0
    
    # Детальная статистика по типам сигналов
    call_wins = sum(1 for t in trades_data if t['signal'] == 'call' and t['result'] == 'win')
    call_total = sum(1 for t in trades_data if t['signal'] == 'call')
    put_wins = sum(1 for t in trades_data if t['signal'] == 'put' and t['result'] == 'win')
    put_total = sum(1 for t in trades_data if t['signal'] == 'put')
    
    call_winrate = call_wins / call_total if call_total > 0 else 0
    put_winrate = put_wins / put_total if put_total > 0 else 0
    
    return {
        'winrate': winrate,
        'wins': wins,
        'losses': losses,
        'total': total_trades,
        'call_winrate': call_winrate,
        'put_winrate': put_winrate,
        'call_wins': call_wins,
        'call_total': call_total,
        'put_wins': put_wins,
        'put_total': put_total,
        'trades_data': trades_data
    }

def market_microstructure_analysis(quotes, params=None):
    """Анализ рынка с продвинутыми индикаторами для исходных данных quotes"""
    if params is None:
        params = OPTIMIZED_PARAMS
    
    signals = []
    
    # Определяем максимальный период для всех индикаторов
    max_lookback = max(
        params.fisher_transform['period'],
        params.relative_momentum['momentum_period'] + params.relative_momentum['smoothing_period'],
        params.squeeze_momentum['bb_period'],
        params.market_facilitation['period'],
        params.vwap_analysis['period'],
        2 * params.fractal_analysis['period'] + 1,
        params.swing_strength['period'] + 1
    )
    
    # Если данных меньше максимального периода, возвращаем пустой список
    if len(quotes) < max_lookback + 1:
        return signals
    
    for i in range(max_lookback, len(quotes)-1):
        window = quotes[i-max_lookback+1:i+1]
        
        # Подготовка данных
        data = []
        for q in window:
            data.append({
                'close': float(get_value(q)),
                'open': float(get_value(q, 'open')),
                'high': float(get_value(q, 'high')),
                'low': float(get_value(q, 'low')),
                'date': q.date
            })
        
        # Получаем базовые данные цен
        close = [float(get_value(q)) for q in window]
        open_p = [float(get_value(q, 'open')) for q in window]
        high = [float(get_value(q, 'high')) for q in window]
        low = [float(get_value(q, 'low')) for q in window]
        
        # 1. Fisher Transform
        fisher = calculate_fisher_transform(close, period=params.fisher_transform['period'])
        
        # 2. Relative Momentum Index
        rmi = calculate_relative_momentum(
            close, 
            momentum_period=params.relative_momentum['momentum_period'], 
            smoothing_period=params.relative_momentum['smoothing_period']
        )
        
        # 3. Squeeze Momentum Indicator
        squeeze_momentum, is_squeeze = calculate_squeeze_momentum(
            data, 
            bb_period=params.squeeze_momentum['bb_period'], 
            bb_mult=params.squeeze_momentum['bb_mult'],
            kc_period=params.squeeze_momentum['kc_period'],
            kc_mult=params.squeeze_momentum['kc_mult']
        )
        
        # 4. Market Facilitation Index
        mfi = calculate_market_facilitation(data, period=params.market_facilitation['period'])
        
        # 5. VWAP Анализ
        vwap_dev = calculate_vwap_analysis(data, period=params.vwap_analysis['period'])
        
        # 6. Фрактальный Анализ
        last_up_fractal, last_down_fractal = identify_fractals(
            high, low, period=params.fractal_analysis['period']
        )
        
        fractal_signal = 0
        if last_up_fractal is not None and last_down_fractal is not None:
            # Последний фрактал - вверх (потенциально нисходящий тренд)
            if last_up_fractal > last_down_fractal:
                fractal_signal = -1
            # Последний фрактал - вниз (потенциально восходящий тренд)
            else:
                fractal_signal = 1
        
        # 7. Сила колебания цены
        swing = calculate_swing_strength(data, period=params.swing_strength['period'])
        
        # Вычисление скора на основе продвинутых индикаторов с весами
        score = 0.5
        
        # Сигналы на повышение с весами из параметров
        if fisher > params.fisher_transform['threshold']:
            score += params.fisher_transform['weight']
        if rmi < 30:
            score += params.relative_momentum['weight']
        if squeeze_momentum > params.relative_momentum['threshold'] and is_squeeze:
            score += params.squeeze_momentum['weight']
        if mfi > params.market_facilitation['threshold'] and close[-1] > open_p[-1]:
            score += params.market_facilitation['weight']
        if vwap_dev < -params.vwap_analysis['std_dev']:
            score += params.vwap_analysis['weight']  # Цена ниже VWAP - потенциал для роста
        if fractal_signal == 1:
            score += params.fractal_analysis['weight']
        if swing > params.swing_strength['threshold'] and close[-1] > close[-2]:
            score += params.swing_strength['weight']
        
        # Сигналы на понижение с весами
        if fisher < -params.fisher_transform['threshold']:
            score -= params.fisher_transform['weight']
        if rmi > 70:
            score -= params.relative_momentum['weight']
        if squeeze_momentum < -params.relative_momentum['threshold'] and is_squeeze:
            score -= params.squeeze_momentum['weight']
        if mfi > params.market_facilitation['threshold'] and close[-1] < open_p[-1]:
            score -= params.market_facilitation['weight']
        if vwap_dev > params.vwap_analysis['std_dev']:
            score -= params.vwap_analysis['weight']  # Цена выше VWAP - потенциал для падения
        if fractal_signal == -1:
            score -= params.fractal_analysis['weight']
        if swing > params.swing_strength['threshold'] and close[-1] < close[-2]:
            score -= params.swing_strength['weight']
        
        # Определение типа сигнала
        signal = 'call' if score > 0.55 else ('put' if score < 0.45 else 'neutral')
        
        signals.append({
            'date': window[-1].date,
            'signal': signal,
            'fisher': fisher,
            'rmi': rmi,
            'squeeze_momentum': squeeze_momentum,
            'is_squeeze': is_squeeze,
            'mfi': mfi,
            'vwap_dev': vwap_dev,
            'fractal_signal': fractal_signal,
            'swing': swing,
            'score': score
        })
    
    return signals

def save_optimized_params(params, filename='optimized_params.json'):
    """Сохранение оптимизированных параметров в файл"""
    try:
        params_dict = params.to_dict()
        with open(filename, 'w') as f:
            json.dump(params_dict, f)
        print(f"Оптимизированные параметры сохранены в {filename}")
    except Exception as e:
        print(f"Ошибка при сохранении параметров: {e}")

def load_optimized_params(filename='optimized_params.json'):
    """Загрузка оптимизированных параметров из файла"""
    global OPTIMIZED_PARAMS, SIGNAL_THRESHOLD
    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                params_dict = json.load(f)
            
            # Восстановление параметров
            for indicator, params in params_dict.items():
                if indicator != 'signal_threshold' and hasattr(OPTIMIZED_PARAMS, indicator):
                    setattr(OPTIMIZED_PARAMS, indicator, params)
                elif indicator == 'signal_threshold':
                    OPTIMIZED_PARAMS.signal_threshold = params
                    SIGNAL_THRESHOLD = params
            
            print(f"Оптимизированные параметры загружены из {filename}")
            return True
    except Exception as e:
        print(f"Ошибка при загрузке параметров: {e}")
    
    return False

def check_data():
    global CANDLES_SINCE_LAST_OPTIMIZATION, SKIP_TRADE_AFTER_OPTIMIZATION, LAST_OPTIMIZATION_TIME
    global FIRST_RUN, TRADING_ALLOWED, SIGNAL_THRESHOLD, CALL_ENABLED, PUT_ENABLED
    
    try:
        quotes = get_quotes(CANDLES)[-100:]
        
        if len(quotes) < 20:
            print("Недостаточно данных для анализа")
            return
        
        # Проверяем необходимость периодической оптимизации или первого запуска
        need_optimization = False
        if OPTIMIZATION_ENABLED and len(quotes) >= 35:
            if FIRST_RUN or CANDLES_SINCE_LAST_OPTIMIZATION >= OPTIMIZATION_PERIOD:
                need_optimization = True
                CANDLES_SINCE_LAST_OPTIMIZATION = 0
                FIRST_RUN = False  # Сбрасываем флаг первого запуска
            
        if need_optimization:
            start_time = time.time()
            
            # Оптимизируем все параметры стратегии одновременно с генетическим алгоритмом
            _, winrate = optimize_strategy_parameters(quotes, min_trades=MIN_TRADES_FOR_OPTIMIZATION)
            
            # Получаем детальные результаты для лучшего набора параметров
            results = get_detailed_strategy_results(quotes, OPTIMIZED_PARAMS)
            
            print(f"\n=== РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ СТРАТЕГИИ ===")
            print(f"► Общий винрейт: {results['winrate']*100:.1f}% ({results['wins']}/{results['total']} сделок)")
            print(f"► CALL винрейт: {results['call_winrate']*100:.1f}% ({results['call_wins']}/{results['call_total']})")
            print(f"► PUT винрейт: {results['put_winrate']*100:.1f}% ({results['put_wins']}/{results['put_total']})")
            
            # Разрешение торговли по направлениям на основе их винрейтов
            if SEPARATE_DIRECTION_FILTER:
                CALL_ENABLED = results['call_winrate'] >= SUCCESS_RATE and results['call_total'] >= MIN_TRADES_FOR_OPTIMIZATION
                PUT_ENABLED = results['put_winrate'] >= SUCCESS_RATE and results['put_total'] >= MIN_TRADES_FOR_OPTIMIZATION
                
                print(f"► CALL-сделки: {'РАЗРЕШЕНЫ' if CALL_ENABLED else 'ЗАПРЕЩЕНЫ'} (винрейт: {results['call_winrate']*100:.1f}%)")
                print(f"► PUT-сделки: {'РАЗРЕШЕНЫ' if PUT_ENABLED else 'ЗАПРЕЩЕНЫ'} (винрейт: {results['put_winrate']*100:.1f}%)")
                
                # Если хотя бы одно направление разрешено, разрешаем торговлю в целом
                TRADING_ALLOWED = CALL_ENABLED or PUT_ENABLED
            else:
                # Разрешаем торговлю только если общий винрейт достаточно хороший
                TRADING_ALLOWED = winrate >= SUCCESS_RATE and results['total'] >= MIN_TRADES_FOR_OPTIMIZATION
                print(f"Торговля {'разрешена' if TRADING_ALLOWED else 'запрещена'} (винрейт: {winrate*100:.1f}%)")
            
            # Сохраняем оптимизированные параметры
            # save_optimized_params(OPTIMIZED_PARAMS)
                
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
        
        # Анализируем рынок с оптимизированными параметрами
        signals = market_microstructure_analysis(quotes, OPTIMIZED_PARAMS)
        
        if signals and len(signals) > 0:
            last_signal = signals[-1]
            signal_strength = abs(last_signal['score'] - 0.5) * 2
            expiry_time = 1
            
            print(f"Сигнал: {last_signal['signal'].upper()}, Сила: {signal_strength:.2f}")
            print(f"Fisher: {last_signal['fisher']:.2f}, RMI: {last_signal['rmi']:.2f}")
            print(f"Squeeze: {last_signal['squeeze_momentum']:.2f}, VWAP: {last_signal['vwap_dev']:.2f}")
            print(f"Экспирация: {expiry_time} мин, Скор: {last_signal['score']:.2f}")
            
            print(f"[DEBUG] Проверка условий для {last_signal['signal'].upper()}:")
            print(f"[DEBUG] 1. Торговля разрешена: {TRADING_ALLOWED}")
            print(f"[DEBUG] 2. Сигнал не нейтральный: {last_signal['signal'] != 'neutral'}")
            print(f"[DEBUG] 3. Сила сигнала {signal_strength:.2f} >= {SIGNAL_THRESHOLD}: {signal_strength >= SIGNAL_THRESHOLD}")
            print(f"[DEBUG] 4. Пропуск после оптимизации: {SKIP_TRADE_AFTER_OPTIMIZATION}")
            
            if SEPARATE_DIRECTION_FILTER:
                print(f"[DEBUG] 5. Торговля в направлении CALL: {CALL_ENABLED}")
                print(f"[DEBUG] 6. Торговля в направлении PUT: {PUT_ENABLED}")
            
            # Используем оптимизированный порог силы сигнала и проверяем разрешение по направлению
            can_trade = TRADING_ALLOWED and last_signal['signal'] != 'neutral' and round(signal_strength, 2) >= SIGNAL_THRESHOLD and not SKIP_TRADE_AFTER_OPTIMIZATION
            
            # Добавляем проверку разрешения по конкретному направлению
            if SEPARATE_DIRECTION_FILTER:
                if last_signal['signal'] == 'call':
                    can_trade = can_trade and CALL_ENABLED
                elif last_signal['signal'] == 'put':
                    can_trade = can_trade and PUT_ENABLED
            
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
                elif signal_strength < SIGNAL_THRESHOLD:
                    print(f"[DEBUG] Сигнал проигнорирован: недостаточная сила ({signal_strength:.2f} < {SIGNAL_THRESHOLD})")
                elif SKIP_TRADE_AFTER_OPTIMIZATION:
                    print("[DEBUG] Сигнал проигнорирован: пропуск после оптимизации")
                elif SEPARATE_DIRECTION_FILTER:
                    if last_signal['signal'] == 'call' and not CALL_ENABLED:
                        print("[DEBUG] Сигнал проигнорирован: торговля CALL не разрешена")
                    elif last_signal['signal'] == 'put' and not PUT_ENABLED:
                        print("[DEBUG] Сигнал проигнорирован: торговля PUT не разрешена")
            
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
    print(f"[DEBUG] Запуск скрипта бинарных опционов с продвинутыми индикаторами (доступно ядер: {num_cores})")
    print(f"[DEBUG] OPTIMIZATION_ENABLED={OPTIMIZATION_ENABLED}")
    print(f"[DEBUG] Период оптимизации: каждые {OPTIMIZATION_PERIOD} свечей")
    print(f"[DEBUG] Минимальный винрейт для торговли: {SUCCESS_RATE*100}%")
    print(f"[DEBUG] Раздельная фильтрация по направлениям: {'Да' if SEPARATE_DIRECTION_FILTER else 'Нет'}")
    print(f"[DEBUG] Минимальное кол-во сделок для оптимизации: {MIN_TRADES_FOR_OPTIMIZATION}")
    print(f"[DEBUG] Начальный порог силы сигнала: {SIGNAL_THRESHOLD}")
    print(f"[DEBUG] Обязательная первая оптимизация: {'Да' if FIRST_RUN else 'Нет'}")
    print(f"[DEBUG] FUTURE_CHECK_PERIODS={FUTURE_CHECK_PERIODS}")
    print(f"[DEBUG] Пропуск сделки после оптимизации: {'Да' if ENABLE_SKIP_AFTER_OPTIMIZATION else 'Нет'}")
    print(f"[DEBUG] Параллельная оптимизация: {num_cores} ядер")
    print(f"[DEBUG] Размер популяции: {POPULATION_SIZE}, Поколений: {NUM_GENERATIONS}")
    
    # Попытка загрузить ранее оптимизированные параметры
    # if not load_optimized_params():
    #     print("[DEBUG] Ранее сохраненные параметры не найдены, будут использованы значения по умолчанию до оптимизации")
    
    # Вывод диапазонов параметров для оптимизации
    print("\n=== ДИАПАЗОНЫ ПАРАМЕТРОВ ПРОДВИНУТЫХ ИНДИКАТОРОВ ===")
    for indicator, ranges in INDICATOR_PARAMS.items():
        print(f"► {indicator}: {ranges}")
    
    print(f"\n=== ДИАПАЗОНЫ ПОРОГА СИЛЫ СИГНАЛА ===")
    print(f"► {STRATEGY_PARAMS['signal_threshold']}")
    
    load_web_driver()
    from time import sleep
    while True:
        websocket_log()
        sleep(0.1)