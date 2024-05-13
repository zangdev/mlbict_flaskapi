import csv
import json
import os
import time

import numpy as np
import pandas as pd
import requests
from keras.models import load_model
from keras.models import save_model
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

global_signal = None
signal_history = []
name_mlbict = "mlbict_1m_150K_candle"
backtrack = 0
candle = 150000


class Model:

    def __init__(self):
        self.model = None

    def build_model(self, input_shape):
        if os.path.exists(f"{name_mlbict}.h5"):
            # Nếu tệp mô hình tồn tại, tải nó
            self.model = load_model(f"{name_mlbict}.h5")
            self.model.compile(optimizer='adam', loss='mean_squared_error')
            print("Model loaded successfully!")
        else:
            with open(f"{name_mlbict}.h5", "w") as file:
                print("Created File: ../mlbict_v1.h5", file=file)
            model = Sequential()
            model.add(LSTM(units=100, return_sequences=True, input_shape=input_shape))
            model.add(Dropout(0.2))
            model.add(LSTM(units=100, return_sequences=True))
            model.add(Dropout(0.2))
            model.add(LSTM(units=100))
            model.add(Dropout(0.2))
            model.add(Dense(units=1))
            model.compile(optimizer='adam', loss='mean_squared_error')
            self.model = model
            save_model(self.model, f"{name_mlbict}.h5")
            print("New model created!")

    def preprocess_data(self, df):
        scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = scaler.fit_transform(df)
        return scaled_data

    def train_model(self, X_train, y_train, epochs=100, batch_size=32):
        print("Train Model ...")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        save_model(self.model, f"{name_mlbict}.h5")
        print("Model Trained successfully! And saved model successfully!")

    def predict(self, X):
        predict_data = self.model.predict(X)
        return predict_data

    def read_dataframe_into_model(self, df_dict):
        columns = ['Price', 'High', 'Low', 'Open', 'EMA_5', 'EMA_15', 'EMA_30', 'EMA_60', 'EMA_100', 'EMA_200', 'WMA',
                   'MACD', 'MACD_Signal', 'MACD_Hist', 'ATR', 'HMA', 'KAMA', 'CMO', 'Z_Score', 'QStick']
        historical_data = pd.DataFrame(columns=columns)

        for i in range(len(df_dict['Price'])):
            row_data = []
            for column in columns:
                row_data.append(df_dict[column][i])
            historical_data.loc[len(historical_data)] = row_data

        return historical_data

    def call_api_each(self, indicator_type, backtrack, symbols, interval):
        tail_url = ""
        result = 100
        if indicator_type == "ema5":
            tail_url = f"results={result}&period=5"
        elif indicator_type == "ema15":
            tail_url = f"results={result}&period=15"
        elif indicator_type == "ema30":
            tail_url = f"results={result}&period=30"
        elif indicator_type == "ema60":
            tail_url = f"results={result}&period=60"
        elif indicator_type == "ema100":
            tail_url = f"results={result}&period=100"
        elif indicator_type == "ema200":
            tail_url = f"results={result}&period=200"
        elif indicator_type == "wma":
            tail_url = f"results={result}"
        elif indicator_type == "macd":
            tail_url = f"results={result}"
        elif indicator_type == "atr":
            tail_url = f"results={result}"
        elif indicator_type == "hma":
            tail_url = f"results={result}&period=50"
        elif indicator_type == "kama":
            tail_url = f"results={result}"
        elif indicator_type == "cmo":
            tail_url = f"results={result}"
        elif indicator_type == "candles":
            tail_url = f"period={result}"
        else:
            raise ValueError("Invalid indicator type")

        if "ema" in indicator_type:
            url = "https://api.taapi.io/ema?"
        elif indicator_type == "wma":
            url = "https://api.taapi.io/wma?"
        elif indicator_type == "macd":
            url = "https://api.taapi.io/macd?"
        elif indicator_type == "atr":
            url = "https://api.taapi.io/atr?"
        elif indicator_type == "hma":
            url = "https://api.taapi.io/hma?"
        elif indicator_type == "kama":
            url = "https://api.taapi.io/kama?"
        elif indicator_type == "cmo":
            url = "https://api.taapi.io/cmo?"
        elif indicator_type == "rsi":
            url = "https://api.taapi.io/rsi?"
        elif indicator_type == "candles":
            url = "https://api.taapi.io/candles?"
        else:
            raise ValueError("Invalid indicator type")

        # Thêm các thông số cần thiết vào URL
        API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjYwYzAxMGRmNWFmOTRlZWNlYTBjOGEyIiwiaWF0IjoxNzE1MjMxMjkwLCJleHAiOjMzMjE5Njk1MjkwfQ.5eizojFZBSJzJVJqEbnAjmwHPXyyBUiWd7RXuBJd2YY"
        url += f"secret={API_KEY}&exchange=binance&symbol={symbols}&interval={interval}"
        if tail_url != "":
            if backtrack != 0:
                url += f"&{tail_url}&backtrack={backtrack}"
            else:
                url += f"&{tail_url}"
        response = requests.get(url)
        return response

    def call_api(self):
        url = "https://api.taapi.io/bulk"
        payload = {
            "secret": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjYxMzdhN2RmNWFmOTRlZWNlYzFiZDg4IiwiaWF0IjoxNzEyNTgyODU2LCJleHAiOjMzMjE3MDQ2ODU2fQ.FFUeWfuKkO1_AWLzVIiZRbejUDBAVHIgy6owHgTQ1QM",
            "construct": {
                "exchange": "binance",
                "symbol": "BTC/USDT",
                "interval": "15m",
                "indicators": [
                    {"indicator": "ema", "results": 1, "period": 5},
                    {"indicator": "ema", "results": 1, "period": 15},
                    {"indicator": "ema", "results": 1, "period": 30},
                    {"indicator": "ema", "results": 1, "period": 60},
                    {"indicator": "ema", "results": 1, "period": 100},
                    {"indicator": "ema", "results": 1, "period": 200},
                    {"indicator": "wma", "results": 1},
                    {"indicator": "macd", "results": 1},
                    {"indicator": "atr", "results": 1},
                    {"indicator": "hma", "results": 1, "period": 50},
                    {"indicator": "kama", "results": 1},
                    {"indicator": "cmo", "results": 1},
                    {"indicator": "candles", "period": 1}
                ]
            }
        }
        headers = {"Content-Type": "application/json"}

        response = requests.post(url, json=payload, headers=headers)

        return response

    def process_json_to_dataframe(self, json_data, indicator_type, df_dict):
        if indicator_type == "macd":
            df_dict['MACD'].extend(json_data['valueMACD'])
            df_dict['MACD_Signal'].extend(json_data['valueMACDSignal'])
            df_dict['MACD_Hist'].extend(json_data['valueMACDHist'])
            return df_dict
        else:
            if indicator_type == "ema5":
                df_dict['EMA_5'].extend(json_data['value'])
            elif indicator_type == "ema15":
                df_dict['EMA_15'].extend(json_data['value'])
            elif indicator_type == "ema30":
                df_dict['EMA_30'].extend(json_data['value'])
            elif indicator_type == "ema60":
                df_dict['EMA_60'].extend(json_data['value'])
            elif indicator_type == "ema100":
                df_dict['EMA_100'].extend(json_data['value'])
            elif indicator_type == "ema200":
                df_dict['EMA_200'].extend(json_data['value'])
            elif indicator_type == "wma":
                df_dict['WMA'].extend(json_data['value'])
            elif indicator_type == "atr":
                df_dict['ATR'].extend(json_data['value'])
            elif indicator_type == "hma":
                df_dict['HMA'].extend(json_data['value'])
            elif indicator_type == "kama":
                df_dict['KAMA'].extend(json_data['value'])
            elif indicator_type == "cmo":
                df_dict['CMO'].extend(json_data['value'])
            elif indicator_type == "candles":
                data = json.loads(json.dumps(json_data))
                for candle in data:
                    df_dict['Price'].append(candle['close'])
                    df_dict['High'].append(candle['high'])
                    df_dict['Low'].append(candle['low'])
                    df_dict['Open'].append(candle['open'])

            return df_dict


# Sử dụng
import numpy as np
import pandas as pd
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import requests


class ModelPredict:
    def __init__(self):
        self.model = Model()
        self.indicator_types = ["ema5", "ema15", "ema30", "ema60", "ema100", "ema200", "wma", "macd", "atr", "hma",
                                "kama", "cmo", "candles"]
        self.df_dict = {'Price': [], 'High': [], 'Low': [], 'Open': [], 'EMA_5': [], 'EMA_15': [], 'EMA_30': [],
                        'EMA_60': [], 'EMA_100': [], 'EMA_200': [], 'WMA': [], 'MACD': [], 'MACD_Signal': [],
                        'MACD_Hist': [], 'ATR': [], 'HMA': [], 'KAMA': [], 'CMO': [], 'Z_Score': [], 'QStick': []}

    def read_csv_to_dict(self, filename):
        candle = 150000
        try:
            with open(filename, 'r') as csvfile:
                reader = csv.DictReader(csvfile)
                next(reader, None)

                for row in reader:
                    if candle < 0:
                        break
                    print(f"df size: {len(self.df_dict['Price'])}")
                    data_valid = True
                    for col in ['close', 'high', 'low', 'open', 'ema_5', 'ema_15', 'ema_30', 'ema_60', 'ema_100',
                                'ema_200', 'WMA', 'MACD',
                                'MACD_Signal', 'MACD_Hist', 'ATR', 'HMA', 'KAMA', 'CMO', 'Z-Score', 'QStick']:
                        if row.get(col) is None:
                            data_valid = False
                            break
                    if data_valid:
                        df_dict_temp = {'Price': [], 'High': [], 'Low': [], 'Open': [], 'EMA_5': [], 'EMA_15': [],
                                        'EMA_30': [],
                                        'EMA_60': [], 'EMA_100': [], 'EMA_200': [], 'WMA': [], 'MACD': [],
                                        'MACD_Signal': [],
                                        'MACD_Hist': [], 'ATR': [], 'HMA': [], 'KAMA': [], 'CMO': [], 'Z_Score': [],
                                        'QStick': []}
                        df_dict_temp['Price'].append(float(row['close']))
                        df_dict_temp['High'].append(float(row['high']))
                        df_dict_temp['Low'].append(float(row['low']))
                        df_dict_temp['Open'].append(float(row['open']))
                        df_dict_temp['EMA_5'].append(float(row['ema_5']) if row['ema_5'] else None)
                        df_dict_temp['EMA_15'].append(float(row['ema_15']) if row['ema_15'] else None)
                        df_dict_temp['EMA_30'].append(float(row['ema_30']) if row['ema_30'] else None)
                        df_dict_temp['EMA_60'].append(float(row['ema_60']) if row['ema_60'] else None)
                        df_dict_temp['EMA_100'].append(float(row['ema_100']) if row['ema_100'] else None)
                        df_dict_temp['EMA_200'].append(float(row['ema_200']) if row['ema_200'] else None)
                        df_dict_temp['WMA'].append(float(row['WMA']) if row['WMA'] else None)
                        df_dict_temp['MACD'].append(float(row['MACD']) if row['MACD'] else None)
                        df_dict_temp['MACD_Signal'].append(float(row['MACD_Signal']) if row['MACD_Signal'] else None)
                        df_dict_temp['MACD_Hist'].append(float(row['MACD_Hist']) if row['MACD_Hist'] else None)
                        df_dict_temp['ATR'].append(float(row['ATR']) if row['ATR'] else None)
                        df_dict_temp['HMA'].append(float(row['HMA']) if row['HMA'] else None)
                        df_dict_temp['KAMA'].append(float(row['KAMA']) if row['KAMA'] else None)
                        df_dict_temp['CMO'].append(float(row['CMO']) if row['CMO'] else None)
                        df_dict_temp['Z_Score'].append(float(row['Z-Score']) if row['Z-Score'] else None)
                        df_dict_temp['QStick'].append(float(row['QStick']) if row['QStick'] else None)
                        reference_length = len(list(df_dict_temp.values())[0])  # Get length from the first value
                        if all(len(item) == reference_length for item in df_dict_temp.values()):
                            for key, value in df_dict_temp.items():
                                val = value[0]
                                self.df_dict[key].append(val)
                            candle = candle - 1

                return True

        except FileNotFoundError:
            print(f"Error: File '{filename}' not found.")
            return False

    def run(self, mode='Training', epochs=100, batch_size=32):
        global backtrack
        blackstrack_demo = backtrack
        self.df_dict.clear()
        self.df_dict = {'Price': [], 'High': [], 'Low': [], 'Open': [], 'EMA_5': [], 'EMA_15': [], 'EMA_30': [],
                        'EMA_60': [], 'EMA_100': [], 'EMA_200': [], 'WMA': [], 'MACD': [], 'MACD_Signal': [],
                        'MACD_Hist': [], 'ATR': [], 'HMA': [], 'KAMA': [], 'CMO': [], 'Z_Score': [], 'QStick': []}
        symbols, interval = "BTC/USDT", "15m"
        print(f"Predicting {symbols}, interval: {interval}")
        print(f"Size {blackstrack_demo * blackstrack_demo} candles")
        # self.read_csv_to_dict("power_data.csv")
        while blackstrack_demo >= 0:
            try:
                print(f"Backtrack: {blackstrack_demo}")
                for indicator in self.indicator_types:
                    response = self.model.call_api_each(indicator, blackstrack_demo, symbols, interval)
                    if response.status_code == 200:
                        self.df_dict = self.model.process_json_to_dataframe(response.json(), indicator, self.df_dict)
                    else:
                        print("API call failed with status code:", response.status_code)
                print(len(self.df_dict['EMA_5']))
                blackstrack_demo -= 1
                time.sleep(2)
            except Exception as e:  # Catch any exceptions
                print(f"Error encountered: {e}")
                print("Pausing for 10 seconds...")
                time.sleep(10)

        # Calculate Z-Score
        z_score_data = [self.df_dict['EMA_5'], self.df_dict['EMA_15'], self.df_dict['EMA_30'], self.df_dict['EMA_60'],
                        self.df_dict['EMA_100'], self.df_dict['EMA_200'], self.df_dict['WMA'], self.df_dict['MACD'],
                        self.df_dict['MACD_Signal'], self.df_dict['MACD_Hist'], self.df_dict['ATR'],
                        self.df_dict['HMA']]
        z_score_mean = np.mean(z_score_data)
        z_score_std = np.std(z_score_data)
        self.df_dict['Z_Score'] = (self.df_dict['EMA_5'] - z_score_mean) / z_score_std

        # Calculate Q-Stick
        self.df_dict['QStick'] = np.array(self.df_dict['Price']) - np.array(self.df_dict['Open'])
        historical_data = pd.DataFrame(self.df_dict)
        scaled_data = self.model.preprocess_data(historical_data)
        X_train = []
        y_train = []

        # Xây dựng dữ liệu đầu vào và đầu ra cho mô hình
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i - 60:i, :])
            y_train.append(scaled_data[i, 0])  # Giá là cột đầu tiên
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))

        # # Xây dựng và huấn luyện mô hình
        # if os.path.exists(f"../{name_mlbict}.h5"):
        #     print("Predicteding ....")
        #     self.model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        #     predicted_price = self.model.predict(X_train[-1].reshape(1, X_train.shape[1], X_train.shape[2]))
        #     last_scaled_data_row = scaled_data[-1]
        #     last_scaled_data_first_column = last_scaled_data_row[0]
        #     print("Predicteding Completed!")
        #     if last_scaled_data_first_column < predicted_price:
        #         signal = "Buy"
        #     else:
        #         signal = "Sell"
        # else:
        #     self.model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        #     self.model.train_model(X_train, y_train, epochs, batch_size)
        #     signal = "Hold"
        print("Predicteding ....")
        self.model.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        predicted_price = self.model.predict(X_train[-1].reshape(1, X_train.shape[1], X_train.shape[2]))
        last_scaled_data_row = scaled_data[-1]
        last_scaled_data_first_column = last_scaled_data_row[0]
        print("Predicteding Completed!")
        if last_scaled_data_first_column < predicted_price:
            signal = "Buy"
        else:
            signal = "Sell"
        return signal

    def train_model(self, X_train, y_train, epochs=100, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, verbose=0)

    def predict(self, X):
        predict_data = self.model.predict(X)
        return predict_data


def check_consecutive_signals(signal):
    global signal_history
    signal_history.append(signal)
    if len(signal_history) >= 2 and all(x == signal_history[-1] for x in signal_history[-4:]):
        signal_history = []
        return True
    return False


if __name__ == "__main__":
    model = ModelPredict()
    while True:
        signal = model.run()
        if global_signal is None:
            global_signal = signal
            print(time.strftime('%Y-%m-%d %H:%M:%S'))
            print("Signal: ", signal)
            print()
            # predictor.write_data_to_file("history_signal.txt", time.strftime('%Y-%m-%d %H:%M:%S'))
            # predictor.write_data_to_file("history_signal.txt", "Signal:" + signal)
            # predictor.write_data_to_file("history_signal.txt", " ")
        elif signal != "Hold" and signal != global_signal:
            if check_consecutive_signals(signal):
                global_signal = signal
                signal_history = []
                print(time.strftime('%Y-%m-%d %H:%M:%S'))
                print("Signal: ", signal)
                print()
                # predictor.write_data_to_file("history_signal.txt", time.strftime('%Y-%m-%d %H:%M:%S'))
                # predictor.write_data_to_file("history_signal.txt", "Signal:" + signal)
                # predictor.write_data_to_file("history_signal.txt", " ")
            else:
                print("Warning of minor fluctuations in trend: ", signal)
        else:
            signal_history = []
        break
