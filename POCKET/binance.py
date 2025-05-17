import requests
import pandas as pd
import datetime
import json
import numpy as np
import time
import random

def get_binance_klines(symbol, interval='15m', limit=100):
    """Получает данные свечей с Binance API"""
    interval_map = {
        '1min': '1m', '5min': '5m', '15min': '15m',
        '1час': '1h', '4часа': '4h', '1день': '1d'
    }
    
    if interval in interval_map:
        interval = interval_map[interval]
        
    url = "https://api.binance.com/api/v3/klines"
    params = {'symbol': symbol, 'interval': interval, 'limit': limit}
    
    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"Ошибка при получении данных для {symbol}: {response.text}")
        return None
    
    klines = response.json()
    
    if klines:
        df = pd.DataFrame(klines, columns=[
            'open_time', 'open', 'high', 'low', 'close', 'volume',
            'close_time', 'quote_asset_volume', 'number_of_trades',
            'taker_buy_base_asset_volume', 'taker_buy_quote_asset_volume', 'ignore'
        ])
        
        df['open_time'] = pd.to_datetime(df['open_time'], unit='ms')
        df['close_time'] = pd.to_datetime(df['close_time'], unit='ms')
        
        for col in ['open', 'high', 'low', 'close', 'volume']:
            df[col] = df[col].astype(float)
            
        df = df.rename(columns={'open_time': 'date'})
        
        return df
    
    return None

def convert_to_list_format(df):
    """Конвертирует DataFrame в формат списка [timestamp, open, high, low, close]"""
    if df is None or df.empty:
        return []
    
    df['timestamp'] = df['date'].apply(
        lambda s: int(datetime.datetime.timestamp(s))
    )
    
    arr = df[['timestamp', 'open', 'high', 'low', 'close']].values.astype(object)
    arr[:, 0] = arr[:, 0].astype(int)
    
    return arr.tolist()

def fetch_all_usdt_pairs(timeframe='15m', candle_limit=100, num_pairs=None, specific_pair=None):
    """
    Получает данные свечей для торговых пар с USDT
    
    Параметры:
    timeframe (str): Таймфрейм ('1min', '5min', '15min', '1час', '4часа', '1день')
    candle_limit (int): Количество свечей для каждой пары (по умолчанию 100)
    num_pairs (int, optional): Количество случайных пар для обработки
    specific_pair (str, optional): Конкретная пара для получения данных
    
    Возвращает:
    dict: Словарь с данными свечей в формате {symbol: [[timestamp, open, high, low, close], ...]}
    """
    candles = {}
    
    # Если указана конкретная пара, получаем данные только для неё
    if specific_pair:
        print(f"Получение данных для указанной пары {specific_pair}...")
        df = get_binance_klines(specific_pair, interval=timeframe, limit=candle_limit)
        if df is not None and not df.empty:
            candles[specific_pair] = convert_to_list_format(df)
            print(f"Данные для {specific_pair} успешно получены")
        else:
            print(f"Не удалось получить данные для {specific_pair}")
        return candles
    
    # Если конкретная пара не указана, получаем список всех доступных пар
    exchange_info_url = "https://api.binance.com/api/v3/exchangeInfo"
    response = requests.get(exchange_info_url)
    
    if response.status_code != 200:
        print(f"Ошибка при получении информации о торговых парах: {response.text}")
        return {}
    
    symbols = response.json()['symbols']
    usdt_pairs = [symbol['symbol'] for symbol in symbols 
                 if symbol['quoteAsset'] == 'USDT' and symbol['status'] == 'TRADING']
    
    total_pairs = len(usdt_pairs)
    print(f"Найдено {total_pairs} торговых пар с USDT")
    
    # Выбираем случайные пары, если указан параметр num_pairs
    if num_pairs is not None and num_pairs > 0:
        if num_pairs > total_pairs:
            print(f"Запрошено {num_pairs} пар, но доступно только {total_pairs}")
            selected_pairs = usdt_pairs
        else:
            selected_pairs = random.sample(usdt_pairs, num_pairs)
            print(f"Случайно выбрано {num_pairs} пар из {total_pairs}")
    else:
        selected_pairs = usdt_pairs
    
    for pair in selected_pairs:
        try:
            print(f"Получение данных для {pair}...")
            df = get_binance_klines(pair, interval=timeframe, limit=candle_limit)
            
            if df is not None and not df.empty:
                candles[pair] = convert_to_list_format(df)
                print(f"Данные для {pair} успешно получены")
            else:
                print(f"Не удалось получить данные для {pair}")
                
            time.sleep(0.5)
        except Exception as e:
            print(f"Ошибка при обработке {pair}: {str(e)}")
    
    print(f"Данные получены для {len(candles)} торговых пар")
    return candles

# Примеры использования:
# Получить данные только для BTC/USDT
# btc_data = fetch_all_usdt_pairs(timeframe='15m', candle_limit=100, specific_pair='BTCUSDT')

# Получить данные для 5 случайных пар
# random_pairs_data = fetch_all_usdt_pairs(timeframe='15m', candle_limit=100, num_pairs=5)

# Получить данные для всех пар
# all_pairs_data = fetch_all_usdt_pairs(timeframe='15m', candle_limit=100)