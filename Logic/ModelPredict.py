import ast
import json
import os
import time

import numpy as np
import pandas as pd
from keras.models import load_model
from keras.models import save_model
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
from sklearn.preprocessing import MinMaxScaler

global_signal = None
signal_history = []
name_mlbict = "mlbict_15m_2_0_0"
backtrack = 0
candle = 150000


class Model:

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
            tail_url = f"results={result}"

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
        elif indicator_type == "abs":
            url = "https://api.taapi.io/abs?"
        elif indicator_type == "accbands":
            url = "https://api.taapi.io/accbands?"
        elif indicator_type == "ad":
            url = "https://api.taapi.io/ad?"
        elif indicator_type == "add":
            url = "https://api.taapi.io/add?"
        elif indicator_type == "adosc":
            url = "https://api.taapi.io/adosc?"
        elif indicator_type == "advanceblock":
            url = "https://api.taapi.io/advanceblock?"
        elif indicator_type == "adx":
            url = "https://api.taapi.io/adx?"
        elif indicator_type == "adxr":
            url = "https://api.taapi.io/adxr?"
        elif indicator_type == "ao":
            url = "https://api.taapi.io/ao?"
        elif indicator_type == "apo":
            url = "https://api.taapi.io/apo?"
        elif indicator_type == "aroon":
            url = "https://api.taapi.io/aroon?"
        elif indicator_type == "aroonosc":
            url = "https://api.taapi.io/aroonosc?"
        elif indicator_type == "atan":
            url = "https://api.taapi.io/atan?"
        elif indicator_type == "atr":
            url = "https://api.taapi.io/atr?"
        elif indicator_type == "avgprice":
            url = "https://api.taapi.io/avgprice?"
        elif indicator_type == "bbands":
            url = "https://api.taapi.io/bbands?"
        elif indicator_type == "belthold":
            url = "https://api.taapi.io/belthold?"
        elif indicator_type == "beta":
            url = "https://api.taapi.io/beta?"
        elif indicator_type == "bop":
            url = "https://api.taapi.io/bop?"
        elif indicator_type == "breakaway":
            url = "https://api.taapi.io/breakaway?"
        elif indicator_type == "candle":
            url = "https://api.taapi.io/candle?"
        elif indicator_type == "candles":
            url = "https://api.taapi.io/candles?"
        elif indicator_type == "cci":
            url = "https://api.taapi.io/cci?"
        elif indicator_type == "ceil":
            url = "https://api.taapi.io/ceil?"
        elif indicator_type == "chop":
            url = "https://api.taapi.io/chop?"
        elif indicator_type == "closingmarubozu":
            url = "https://api.taapi.io/closingmarubozu?"
        elif indicator_type == "cmf":
            url = "https://api.taapi.io/cmf?"
        elif indicator_type == "cmo":
            url = "https://api.taapi.io/cmo?"
        elif indicator_type == "concealbabyswall":
            url = "https://api.taapi.io/concealbabyswall?"
        elif indicator_type == "coppockcurve":
            url = "https://api.taapi.io/coppockcurve?"
        elif indicator_type == "correl":
            url = "https://api.taapi.io/correl?"
        elif indicator_type == "cos":
            url = "https://api.taapi.io/cos?"
        elif indicator_type == "counterattack":
            url = "https://api.taapi.io/counterattack?"
        elif indicator_type == "darkcloudcover":
            url = "https://api.taapi.io/darkcloudcover?"
        elif indicator_type == "dema":
            url = "https://api.taapi.io/dema?"
        elif indicator_type == "div":
            url = "https://api.taapi.io/div?"
        elif indicator_type == "dm":
            url = "https://api.taapi.io/dm?"
        elif indicator_type == "dmi":
            url = "https://api.taapi.io/dmi?"
        elif indicator_type == "doji":
            url = "https://api.taapi.io/doji?"
        elif indicator_type == "dojistar":
            url = "https://api.taapi.io/dojistar?"
        elif indicator_type == "donchianchannels":
            url = "https://api.taapi.io/donchianchannels?"
        elif indicator_type == "dpo":
            url = "https://api.taapi.io/dpo?"
        elif indicator_type == "dragonflydoji":
            url = "https://api.taapi.io/dragonflydoji?"
        elif indicator_type == "dx":
            url = "https://api.taapi.io/dx?"
        elif indicator_type == "ema":
            url = "https://api.taapi.io/ema?"
        elif indicator_type == "engulfing":
            url = "https://api.taapi.io/engulfing?"
        elif indicator_type == "eom":
            url = "https://api.taapi.io/eom?"
        elif indicator_type == "eveningdojistar":
            url = "https://api.taapi.io/eveningdojistar?"
        elif indicator_type == "eveningstar":
            url = "https://api.taapi.io/eveningstar?"
        elif indicator_type == "fibonacciretracement":
            url = "https://api.taapi.io/fibonacciretracement?"
        elif indicator_type == "fisher":
            url = "https://api.taapi.io/fisher?"
        elif indicator_type == "floor":
            url = "https://api.taapi.io/floor?"
        elif indicator_type == "fosc":
            url = "https://api.taapi.io/fosc?"
        elif indicator_type == "gapsidesidewhite":
            url = "https://api.taapi.io/gapsidesidewhite?"
        elif indicator_type == "gravestonedoji":
            url = "https://api.taapi.io/gravestonedoji?"
        elif indicator_type == "hammer":
            url = "https://api.taapi.io/hammer?"
        elif indicator_type == "hangingman":
            url = "https://api.taapi.io/hangingman?"
        elif indicator_type == "harami":
            url = "https://api.taapi.io/harami?"
        elif indicator_type == "haramicross":
            url = "https://api.taapi.io/haramicross?"
        elif indicator_type == "highwave":
            url = "https://api.taapi.io/highwave?"
        elif indicator_type == "hikkake":
            url = "https://api.taapi.io/hikkake?"
        elif indicator_type == "hikkakemod":
            url = "https://api.taapi.io/hikkakemod?"
        elif indicator_type == "hma":
            url = "https://api.taapi.io/hma?"
        elif indicator_type == "homingpigeon":
            url = "https://api.taapi.io/homingpigeon?"
        elif indicator_type == "ht_dcperiod":
            url = "https://api.taapi.io/ht_dcperiod?"
        elif indicator_type == "ht_dcphase":
            url = "https://api.taapi.io/ht_dcphase?"
        elif indicator_type == "ht_phasor":
            url = "https://api.taapi.io/ht_phasor?"
        elif indicator_type == "ht_sine":
            url = "https://api.taapi.io/ht_sine?"
        elif indicator_type == "ht_trendline":
            url = "https://api.taapi.io/ht_trendline?"
        elif indicator_type == "ht_trendmode":
            url = "https://api.taapi.io/ht_trendmode?"
        elif indicator_type == "ichimoku":
            url = "https://api.taapi.io/ichimoku?"
        elif indicator_type == "identical3crows":
            url = "https://api.taapi.io/identical3crows?"
        elif indicator_type == "inneck":
            url = "https://api.taapi.io/inneck?"
        elif indicator_type == "invertedhammer":
            url = "https://api.taapi.io/invertedhammer?"
        elif indicator_type == "kama":
            url = "https://api.taapi.io/kama?"
        elif indicator_type == "kdj":
            url = "https://api.taapi.io/kdj?"
        elif indicator_type == "keltnerchannels":
            url = "https://api.taapi.io/keltnerchannels?"
        elif indicator_type == "kicking":
            url = "https://api.taapi.io/kicking?"
        elif indicator_type == "kickingbylength":
            url = "https://api.taapi.io/kickingbylength?"
        elif indicator_type == "kvo":
            url = "https://api.taapi.io/kvo?"
        elif indicator_type == "ladderbottom":
            url = "https://api.taapi.io/ladderbottom?"
        elif indicator_type == "lantern":
            url = "https://api.taapi.io/lantern?"
        elif indicator_type == "lanterns":
            url = "https://api.taapi.io/lanterns?"
        elif indicator_type == "linearreg":
            url = "https://api.taapi.io/linearreg?"
        elif indicator_type == "linearreg_angle":
            url = "https://api.taapi.io/linearreg_angle?"
        elif indicator_type == "linearreg_intercept":
            url = "https://api.taapi.io/linearreg_intercept?"
        elif indicator_type == "linearreg_slope":
            url = "https://api.taapi.io/linearreg_slope?"
        elif indicator_type == "ln":
            url = "https://api.taapi.io/ln?"
        elif indicator_type == "log10":
            url = "https://api.taapi.io/log10?"
        elif indicator_type == "longleggeddoji":
            url = "https://api.taapi.io/longleggeddoji?"
        elif indicator_type == "longline":
            url = "https://api.taapi.io/longline?"
        elif indicator_type == "ma":
            url = "https://api.taapi.io/ma?"
        elif indicator_type == "macd":
            url = "https://api.taapi.io/macd?"
        elif indicator_type == "macdext":
            url = "https://api.taapi.io/macdext?"
        elif indicator_type == "mama":
            url = "https://api.taapi.io/mama?"
        elif indicator_type == "marketfi":
            url = "https://api.taapi.io/marketfi?"
        elif indicator_type == "marubozu":
            url = "https://api.taapi.io/marubozu?"
        elif indicator_type == "mass":
            url = "https://api.taapi.io/mass?"
        elif indicator_type == "matchinglow":
            url = "https://api.taapi.io/matchinglow?"
        elif indicator_type == "mathold":
            url = "https://api.taapi.io/mathold?"
        elif indicator_type == "max":
            url = "https://api.taapi.io/max?"
        elif indicator_type == "maxindex":
            url = "https://api.taapi.io/maxindex?"
        elif indicator_type == "medprice":
            url = "https://api.taapi.io/medprice?"
        elif indicator_type == "mfi":
            url = "https://api.taapi.io/mfi?"
        elif indicator_type == "midpoint":
            url = "https://api.taapi.io/midpoint?"
        elif indicator_type == "midprice":
            url = "https://api.taapi.io/midprice?"
        elif indicator_type == "min":
            url = "https://api.taapi.io/min?"
        elif indicator_type == "minindex":
            url = "https://api.taapi.io/minindex?"
        elif indicator_type == "minmax":
            url = "https://api.taapi.io/minmax?"
        elif indicator_type == "minmaxindex":
            url = "https://api.taapi.io/minmaxindex?"
        elif indicator_type == "minus_di":
            url = "https://api.taapi.io/minus_di?"
        elif indicator_type == "minus_dm":
            url = "https://api.taapi.io/minus_dm?"
        elif indicator_type == "mom":
            url = "https://api.taapi.io/mom?"
        elif indicator_type == "morningdojistar":
            url = "https://api.taapi.io/morningdojistar?"
        elif indicator_type == "morningstar":
            url = "https://api.taapi.io/morningstar?"
        elif indicator_type == "msw":
            url = "https://api.taapi.io/msw?"
        elif indicator_type == "mul":
            url = "https://api.taapi.io/mul?"
        elif indicator_type == "mult":
            url = "https://api.taapi.io/mult?"
        elif indicator_type == "natr":
            url = "https://api.taapi.io/natr?"
        elif indicator_type == "nvi":
            url = "https://api.taapi.io/nvi?"
        elif indicator_type == "obv":
            url = "https://api.taapi.io/obv?"
        elif indicator_type == "onneck":
            url = "https://api.taapi.io/onneck?"
        elif indicator_type == "pd":
            url = "https://api.taapi.io/pd?"
        elif indicator_type == "piercing":
            url = "https://api.taapi.io/piercing?"
        elif indicator_type == "pivotpoints":
            url = "https://api.taapi.io/pivotpoints?"
        elif indicator_type == "plus_di":
            url = "https://api.taapi.io/plus_di?"
        elif indicator_type == "plus_dm":
            url = "https://api.taapi.io/plus_dm?"
        elif indicator_type == "ppo":
            url = "https://api.taapi.io/ppo?"
        elif indicator_type == "price":
            url = "https://api.taapi.io/price?"
        elif indicator_type == "priorswinghigh":
            url = "https://api.taapi.io/priorswinghigh?"
        elif indicator_type == "priorswinglow":
            url = "https://api.taapi.io/priorswinglow?"
        elif indicator_type == "psar":
            url = "https://api.taapi.io/psar?"
        elif indicator_type == "pvi":
            url = "https://api.taapi.io/pvi?"
        elif indicator_type == "qstick":
            url = "https://api.taapi.io/qstick?"
        elif indicator_type == "rickshawman":
            url = "https://api.taapi.io/rickshawman?"
        elif indicator_type == "risefall3methods":
            url = "https://api.taapi.io/risefall3methods?"
        elif indicator_type == "roc":
            url = "https://api.taapi.io/roc?"
        elif indicator_type == "rocp":
            url = "https://api.taapi.io/rocp?"
        elif indicator_type == "rocr":
            url = "https://api.taapi.io/rocr?"
        elif indicator_type == "rocr100":
            url = "https://api.taapi.io/rocr100?"
        elif indicator_type == "round":
            url = "https://api.taapi.io/round?"
        elif indicator_type == "rsi":
            url = "https://api.taapi.io/rsi?"
        elif indicator_type == "separatinglines":
            url = "https://api.taapi.io/separatinglines?"
        elif indicator_type == "shootingstar":
            url = "https://api.taapi.io/shootingstar?"
        elif indicator_type == "shortline":
            url = "https://api.taapi.io/shortline?"
        elif indicator_type == "sin":
            url = "https://api.taapi.io/sin?"
        elif indicator_type == "sma":
            url = "https://api.taapi.io/sma?"
        elif indicator_type == "smma":
            url = "https://api.taapi.io/smma?"
        elif indicator_type == "spinningtop":
            url = "https://api.taapi.io/spinningtop?"
        elif indicator_type == "sqrt":
            url = "https://api.taapi.io/sqrt?"
        elif indicator_type == "stalledpattern":
            url = "https://api.taapi.io/stalledpattern?"
        elif indicator_type == "stddev":
            url = "https://api.taapi.io/stddev?"
        elif indicator_type == "sticksandwich":
            url = "https://api.taapi.io/sticksandwich?"
        elif indicator_type == "stoch":
            url = "https://api.taapi.io/stoch?"
        elif indicator_type == "stochf":
            url = "https://api.taapi.io/stochf?"
        elif indicator_type == "stochrsi":
            url = "https://api.taapi.io/stochrsi?"
        elif indicator_type == "sub":
            url = "https://api.taapi.io/sub?"
        elif indicator_type == "sum":
            url = "https://api.taapi.io/sum?"
        elif indicator_type == "supertrend":
            url = "https://api.taapi.io/supertrend?"
        elif indicator_type == "t3":
            url = "https://api.taapi.io/t3?"
        elif indicator_type == "takuri":
            url = "https://api.taapi.io/takuri?"
        elif indicator_type == "tan":
            url = "https://api.taapi.io/tan?"
        elif indicator_type == "tanh":
            url = "https://api.taapi.io/tanh?"
        elif indicator_type == "tasukigap":
            url = "https://api.taapi.io/tasukigap?"
        elif indicator_type == "tdsequential":
            url = "https://api.taapi.io/tdsequential?"
        elif indicator_type == "tema":
            url = "https://api.taapi.io/tema?"
        elif indicator_type == "thrusting":
            url = "https://api.taapi.io/thrusting?"
        elif indicator_type == "todeg":
            url = "https://api.taapi.io/todeg?"
        elif indicator_type == "torad":
            url = "https://api.taapi.io/torad?"
        elif indicator_type == "tr":
            url = "https://api.taapi.io/tr?"
        elif indicator_type == "trima":
            url = "https://api.taapi.io/trima?"
        elif indicator_type == "tristar":
            url = "https://api.taapi.io/tristar?"
        elif indicator_type == "trix":
            url = "https://api.taapi.io/trix?"
        elif indicator_type == "trunc":
            url = "https://api.taapi.io/trunc?"
        elif indicator_type == "tsf":
            url = "https://api.taapi.io/tsf?"
        elif indicator_type == "typprice":
            url = "https://api.taapi.io/typprice?"
        elif indicator_type == "ultosc":
            url = "https://api.taapi.io/ultosc?"
        elif indicator_type == "unique3river":
            url = "https://api.taapi.io/unique3river?"
        elif indicator_type == "upsidegap2crows":
            url = "https://api.taapi.io/upsidegap2crows?"
        elif indicator_type == "var":
            url = "https://api.taapi.io/var?"
        elif indicator_type == "vhf":
            url = "https://api.taapi.io/vhf?"
        elif indicator_type == "vidya":
            url = "https://api.taapi.io/vidya?"
        elif indicator_type == "volatility":
            url = "https://api.taapi.io/volatility?"
        elif indicator_type == "vortex":
            url = "https://api.taapi.io/vortex?"
        elif indicator_type == "vosc":
            url = "https://api.taapi.io/vosc?"
        elif indicator_type == "vwap":
            url = "https://api.taapi.io/vwap?"
        elif indicator_type == "vwma":
            url = "https://api.taapi.io/vwma?"
        elif indicator_type == "wad":
            url = "https://api.taapi.io/wad?"
        elif indicator_type == "wclprice":
            url = "https://api.taapi.io/wclprice?"
        elif indicator_type == "wilders":
            url = "https://api.taapi.io/wilders?"
        elif indicator_type == "williamsalligator":
            url = "https://api.taapi.io/williamsalligator?"
        elif indicator_type == "willr":
            url = "https://api.taapi.io/willr?"
        elif indicator_type == "wma":
            url = "https://api.taapi.io/wma?"
        elif indicator_type == "xsidegap3methods":
            url = "https://api.taapi.io/xsidegap3methods?"
        elif indicator_type == "zlema":
            url = "https://api.taapi.io/zlema?"
        else:
            raise ValueError(f"Invalid indicator type: {indicator_type}")

        # Thêm các thông số cần thiết vào URL
        API_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJjbHVlIjoiNjY0NWMzYWNmNWFmOTRlZWNlOWI4ZTE0IiwiaWF0IjoxNzE1ODQ4MTczLCJleHAiOjMzMjIwMzEyMTczfQ.m8hlXdtBk8do4fiEw9LkTm7acKMhHpu5SyTdMXvktcE"
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
            elif indicator_type == "abs":
                df_dict['abs'].extend(json_data['value'])
            elif indicator_type == "accbands":
                df_dict['valueUpperBand'].extend(json_data['valueUpperBand'])
                df_dict['valueMiddleBand'].extend(json_data['valueMiddleBand'])
                df_dict['valueLowerBand'].extend(json_data['valueLowerBand'])
            elif indicator_type == "ad":
                df_dict['ad'].extend(json_data['value'])
            elif indicator_type == "add":
                df_dict['add'].extend(json_data['value'])
            elif indicator_type == "adosc":
                df_dict['adosc'].extend(json_data['value'])
            elif indicator_type == "advanceblock":
                df_dict['advanceblock'].extend(json_data['value'])
            elif indicator_type == "adx":
                df_dict['adx'].extend(json_data['value'])
            elif indicator_type == "adxr":
                df_dict['adxr'].extend(json_data['value'])
            elif indicator_type == "ao":
                df_dict['ao'].extend(json_data['value'])
            elif indicator_type == "apo":
                df_dict['apo'].extend(json_data['value'])
            elif indicator_type == "aroon":
                df_dict['valueAroonDown'].extend(json_data['valueAroonDown'])
                df_dict['valueAroonUp'].extend(json_data['valueAroonUp'])
            elif indicator_type == "aroonosc":
                df_dict['aroonosc'].extend(json_data['value'])
            elif indicator_type == "atan":
                df_dict['atan'].extend(json_data['value'])
            elif indicator_type == "avgprice":
                df_dict['avgprice'].extend(json_data['value'])
            elif indicator_type == "bbands":
                df_dict['valueUpperBand1'].extend(json_data['valueUpperBand'])
                df_dict['valueMiddleBand1'].extend(json_data['valueMiddleBand'])
                df_dict['valueLowerBand1'].extend(json_data['valueLowerBand'])
            elif indicator_type == "belthold":
                df_dict['belthold'].extend(json_data['value'])
            elif indicator_type == "beta":
                df_dict['beta'].extend(json_data['value'])
            elif indicator_type == "bop":
                df_dict['bop'].extend(json_data['value'])
            elif indicator_type == "breakaway":
                df_dict['breakaway'].extend(json_data['value'])
            elif indicator_type == "cci":
                df_dict['cci'].extend(json_data['value'])
            elif indicator_type == "ceil":
                df_dict['ceil'].extend(json_data['value'])
            elif indicator_type == "chop":
                df_dict['chop'].extend(json_data['value'])
            elif indicator_type == "closingmarubozu":
                df_dict['closingmarubozu'].extend(json_data['value'])
            elif indicator_type == "cmf":
                df_dict['cmf'].extend(json_data['value'])
            elif indicator_type == "concealbabyswall":
                df_dict['concealbabyswall'].extend(json_data['value'])
            elif indicator_type == "coppockcurve":
                df_dict['coppockcurve'].extend(json_data['value'])
            elif indicator_type == "correl":
                df_dict['correl'].extend(json_data['value'])
            elif indicator_type == "cos":
                df_dict['cos'].extend(json_data['value'])
            elif indicator_type == "counterattack":
                df_dict['counterattack'].extend(json_data['value'])
            elif indicator_type == "darkcloudcover":
                df_dict['darkcloudcover'].extend(json_data['value'])
            elif indicator_type == "dema":
                df_dict['dema'].extend(json_data['value'])
            elif indicator_type == "div":
                df_dict['div'].extend(json_data['value'])
            elif indicator_type == "dm":
                df_dict['dm'].extend(json_data['value'])
            elif indicator_type == "dmi":
                df_dict['dmi'].extend(json_data['value'])
            elif indicator_type == "doji":
                df_dict['doji'].extend(json_data['value'])
            elif indicator_type == "dojistar":
                df_dict['dojistar'].extend(json_data['value'])
            elif indicator_type == "donchianchannels":
                df_dict['donchianchannels'].extend(json_data['value'])
            elif indicator_type == "dpo":
                df_dict['dpo'].extend(json_data['value'])
            elif indicator_type == "dragonflydoji":
                df_dict['dragonflydoji'].extend(json_data['value'])
            elif indicator_type == "dx":
                df_dict['dx'].extend(json_data['value'])
            elif indicator_type == "engulfing":
                df_dict['engulfing'].extend(json_data['value'])
            elif indicator_type == "eom":
                df_dict['eom'].extend(json_data['value'])
            elif indicator_type == "eveningdojistar":
                df_dict['eveningdojistar'].extend(json_data['value'])
            elif indicator_type == "eveningstar":
                df_dict['eveningstar'].extend(json_data['value'])
            elif indicator_type == "fibonacciretracement":
                df_dict['fibonacciretracement'].extend(json_data['value'])
            elif indicator_type == "fisher":
                df_dict['fisher'].extend(json_data['value'])
            elif indicator_type == "floor":
                df_dict['floor'].extend(json_data['value'])
            elif indicator_type == "fosc":
                df_dict['fosc'].extend(json_data['value'])
            elif indicator_type == "gapsidesidewhite":
                df_dict['gapsidesidewhite'].extend(json_data['value'])
            elif indicator_type == "gravestonedoji":
                df_dict['gravestonedoji'].extend(json_data['value'])
            elif indicator_type == "hammer":
                df_dict['hammer'].extend(json_data['value'])
            elif indicator_type == "hangingman":
                df_dict['hangingman'].extend(json_data['value'])
            elif indicator_type == "harami":
                df_dict['harami'].extend(json_data['value'])
            elif indicator_type == "haramicross":
                df_dict['haramicross'].extend(json_data['value'])
            elif indicator_type == "highwave":
                df_dict['highwave'].extend(json_data['value'])
            elif indicator_type == "hikkake":
                df_dict['hikkake'].extend(json_data['value'])
            elif indicator_type == "hikkakemod":
                df_dict['hikkakemod'].extend(json_data['value'])
            elif indicator_type == "hma":
                df_dict['hma'].extend(json_data['value'])
            elif indicator_type == "homingpigeon":
                df_dict['homingpigeon'].extend(json_data['value'])
            elif indicator_type == "ht_dcperiod":
                df_dict['ht_dcperiod'].extend(json_data['value'])
            elif indicator_type == "ht_dcphase":
                df_dict['ht_dcphase'].extend(json_data['value'])
            elif indicator_type == "ht_phasor":
                df_dict['ht_phasor'].extend(json_data['value'])
            elif indicator_type == "ht_sine":
                df_dict['ht_sine'].extend(json_data['value'])
            elif indicator_type == "ht_trendline":
                df_dict['ht_trendline'].extend(json_data['value'])
            elif indicator_type == "ht_trendmode":
                df_dict['ht_trendmode'].extend(json_data['value'])
            elif indicator_type == "ichimoku":
                df_dict['ichimoku'].extend(json_data['value'])
            elif indicator_type == "identical3crows":
                df_dict['identical3crows'].extend(json_data['value'])
            elif indicator_type == "inneck":
                df_dict['inneck'].extend(json_data['value'])
            elif indicator_type == "invertedhammer":
                df_dict['invertedhammer'].extend(json_data['value'])
            elif indicator_type == "kama":
                df_dict['kama'].extend(json_data['value'])
            elif indicator_type == "kdj":
                df_dict['kdj'].extend(json_data['value'])
            elif indicator_type == "keltnerchannels":
                df_dict['keltnerchannels'].extend(json_data['value'])
            elif indicator_type == "kicking":
                df_dict['kicking'].extend(json_data['value'])
            elif indicator_type == "kickingbylength":
                df_dict['kickingbylength'].extend(json_data['value'])
            elif indicator_type == "kvo":
                df_dict['kvo'].extend(json_data['value'])
            elif indicator_type == "ladderbottom":
                df_dict['ladderbottom'].extend(json_data['value'])
            elif indicator_type == "lantern":
                df_dict['lantern'].extend(json_data['value'])
            elif indicator_type == "lanterns":
                df_dict['lanterns'].extend(json_data['value'])
            elif indicator_type == "linearreg":
                df_dict['linearreg'].extend(json_data['value'])
            elif indicator_type == "linearreg_angle":
                df_dict['linearreg_angle'].extend(json_data['value'])
            elif indicator_type == "linearreg_intercept":
                df_dict['linearreg_intercept'].extend(json_data['value'])
            elif indicator_type == "linearreg_slope":
                df_dict['linearreg_slope'].extend(json_data['value'])
            elif indicator_type == "ln":
                df_dict['ln'].extend(json_data['value'])
            elif indicator_type == "log10":
                df_dict['log10'].extend(json_data['value'])
            elif indicator_type == "longleggeddoji":
                df_dict['longleggeddoji'].extend(json_data['value'])
            elif indicator_type == "longline":
                df_dict['longline'].extend(json_data['value'])
            elif indicator_type == "ma":
                df_dict['ma'].extend(json_data['value'])
            elif indicator_type == "macdext":
                df_dict['macdext'].extend(json_data['value'])
            elif indicator_type == "mama":
                df_dict['mama'].extend(json_data['value'])
            elif indicator_type == "marketfi":
                df_dict['marketfi'].extend(json_data['value'])
            elif indicator_type == "marubozu":
                df_dict['marubozu'].extend(json_data['value'])
            elif indicator_type == "mass":
                df_dict['mass'].extend(json_data['value'])
            elif indicator_type == "matchinglow":
                df_dict['matchinglow'].extend(json_data['value'])
            elif indicator_type == "mathold":
                df_dict['mathold'].extend(json_data['value'])
            elif indicator_type == "max":
                df_dict['max'].extend(json_data['value'])
            elif indicator_type == "maxindex":
                df_dict['maxindex'].extend(json_data['value'])
            elif indicator_type == "medprice":
                df_dict['medprice'].extend(json_data['value'])
            elif indicator_type == "mfi":
                df_dict['mfi'].extend(json_data['value'])
            elif indicator_type == "midpoint":
                df_dict['midpoint'].extend(json_data['value'])
            elif indicator_type == "midprice":
                df_dict['midprice'].extend(json_data['value'])
            elif indicator_type == "min":
                df_dict['min'].extend(json_data['value'])
            elif indicator_type == "minindex":
                df_dict['minindex'].extend(json_data['value'])
            elif indicator_type == "minmax":
                df_dict['minmax'].extend(json_data['value'])
            elif indicator_type == "minmaxindex":
                df_dict['minmaxindex'].extend(json_data['value'])
            elif indicator_type == "minus_di":
                df_dict['minus_di'].extend(json_data['value'])
            elif indicator_type == "minus_dm":
                df_dict['minus_dm'].extend(json_data['value'])
            elif indicator_type == "mom":
                df_dict['mom'].extend(json_data['value'])
            elif indicator_type == "morningdojistar":
                df_dict['morningdojistar'].extend(json_data['value'])
            elif indicator_type == "morningstar":
                df_dict['morningstar'].extend(json_data['value'])
            elif indicator_type == "msw":
                df_dict['msw'].extend(json_data['value'])
            elif indicator_type == "mul":
                df_dict['mul'].extend(json_data['value'])
            elif indicator_type == "mult":
                df_dict['mult'].extend(json_data['value'])
            elif indicator_type == "natr":
                df_dict['natr'].extend(json_data['value'])
            elif indicator_type == "nvi":
                df_dict['nvi'].extend(json_data['value'])
            elif indicator_type == "obv":
                df_dict['obv'].extend(json_data['value'])
            elif indicator_type == "onneck":
                df_dict['onneck'].extend(json_data['value'])
            elif indicator_type == "pd":
                df_dict['pd'].extend(json_data['value'])
            elif indicator_type == "piercing":
                df_dict['piercing'].extend(json_data['value'])
            elif indicator_type == "pivotpoints":
                df_dict['pivotpoints'].extend(json_data['value'])
            elif indicator_type == "plus_di":
                df_dict['plus_di'].extend(json_data['value'])
            elif indicator_type == "plus_dm":
                df_dict['plus_dm'].extend(json_data['value'])
            elif indicator_type == "ppo":
                df_dict['ppo'].extend(json_data['value'])
            elif indicator_type == "price":
                df_dict['price'].extend(json_data['value'])
            elif indicator_type == "priorswinghigh":
                df_dict['priorswinghigh'].extend(json_data['value'])
            elif indicator_type == "priorswinglow":
                df_dict['priorswinglow'].extend(json_data['value'])
            elif indicator_type == "psar":
                df_dict['psar'].extend(json_data['value'])
            elif indicator_type == "pvi":
                df_dict['pvi'].extend(json_data['value'])
            elif indicator_type == "qstick":
                df_dict['qstick'].extend(json_data['value'])
            elif indicator_type == "rickshawman":
                df_dict['rickshawman'].extend(json_data['value'])
            elif indicator_type == "risefall3methods":
                df_dict['risefall3methods'].extend(json_data['value'])
            elif indicator_type == "roc":
                df_dict['roc'].extend(json_data['value'])
            elif indicator_type == "rocp":
                df_dict['rocp'].extend(json_data['value'])
            elif indicator_type == "rocr":
                df_dict['rocr'].extend(json_data['value'])
            elif indicator_type == "rocr100":
                df_dict['rocr100'].extend(json_data['value'])
            elif indicator_type == "round":
                df_dict['round'].extend(json_data['value'])
            elif indicator_type == "rsi":
                df_dict['rsi'].extend(json_data['value'])
            elif indicator_type == "separatinglines":
                df_dict['separatinglines'].extend(json_data['value'])
            elif indicator_type == "shootingstar":
                df_dict['shootingstar'].extend(json_data['value'])
            elif indicator_type == "shortline":
                df_dict['shortline'].extend(json_data['value'])
            elif indicator_type == "sin":
                df_dict['sin'].extend(json_data['value'])
            elif indicator_type == "sma":
                df_dict['sma'].extend(json_data['value'])
            elif indicator_type == "smma":
                df_dict['smma'].extend(json_data['value'])
            elif indicator_type == "spinningtop":
                df_dict['spinningtop'].extend(json_data['value'])
            elif indicator_type == "sqrt":
                df_dict['sqrt'].extend(json_data['value'])
            elif indicator_type == "stalledpattern":
                df_dict['stalledpattern'].extend(json_data['value'])
            elif indicator_type == "stddev":
                df_dict['stddev'].extend(json_data['value'])
            elif indicator_type == "sticksandwich":
                df_dict['sticksandwich'].extend(json_data['value'])
            elif indicator_type == "stoch":
                df_dict['stoch'].extend(json_data['value'])
            elif indicator_type == "stochf":
                df_dict['stochf'].extend(json_data['value'])
            elif indicator_type == "stochrsi":
                df_dict['stochrsi'].extend(json_data['value'])
            elif indicator_type == "sub":
                df_dict['sub'].extend(json_data['value'])
            elif indicator_type == "sum":
                df_dict['sum'].extend(json_data['value'])
            elif indicator_type == "supertrend":
                df_dict['supertrend'].extend(json_data['value'])
            elif indicator_type == "t3":
                df_dict['t3'].extend(json_data['value'])
            elif indicator_type == "takuri":
                df_dict['takuri'].extend(json_data['value'])
            elif indicator_type == "tan":
                df_dict['tan'].extend(json_data['value'])
            elif indicator_type == "tanh":
                df_dict['tanh'].extend(json_data['value'])
            elif indicator_type == "tasukigap":
                df_dict['tasukigap'].extend(json_data['value'])
            elif indicator_type == "tdsequential":
                df_dict['tdsequential'].extend(json_data['value'])
            elif indicator_type == "tema":
                df_dict['tema'].extend(json_data['value'])
            elif indicator_type == "thrusting":
                df_dict['thrusting'].extend(json_data['value'])
            elif indicator_type == "todeg":
                df_dict['todeg'].extend(json_data['value'])
            elif indicator_type == "torad":
                df_dict['torad'].extend(json_data['value'])
            elif indicator_type == "tr":
                df_dict['tr'].extend(json_data['value'])
            elif indicator_type == "trima":
                df_dict['trima'].extend(json_data['value'])
            elif indicator_type == "tristar":
                df_dict['tristar'].extend(json_data['value'])
            elif indicator_type == "trix":
                df_dict['trix'].extend(json_data['value'])
            elif indicator_type == "trunc":
                df_dict['trunc'].extend(json_data['value'])
            elif indicator_type == "tsf":
                df_dict['tsf'].extend(json_data['value'])
            elif indicator_type == "typprice":
                df_dict['typprice'].extend(json_data['value'])
            elif indicator_type == "ultosc":
                df_dict['ultosc'].extend(json_data['value'])
            elif indicator_type == "unique3river":
                df_dict['unique3river'].extend(json_data['value'])
            elif indicator_type == "upsidegap2crows":
                df_dict['upsidegap2crows'].extend(json_data['value'])
            elif indicator_type == "var":
                df_dict['var'].extend(json_data['value'])
            elif indicator_type == "vhf":
                df_dict['vhf'].extend(json_data['value'])
            elif indicator_type == "vidya":
                df_dict['vidya'].extend(json_data['value'])
            elif indicator_type == "volatility":
                df_dict['volatility'].extend(json_data['value'])
            elif indicator_type == "vortex":
                df_dict['vortex'].extend(json_data['value'])
            elif indicator_type == "vosc":
                df_dict['vosc'].extend(json_data['value'])
            elif indicator_type == "vwap":
                df_dict['vwap'].extend(json_data['value'])
            elif indicator_type == "vwma":
                df_dict['vwma'].extend(json_data['value'])
            elif indicator_type == "wad":
                df_dict['wad'].extend(json_data['value'])
            elif indicator_type == "wclprice":
                df_dict['wclprice'].extend(json_data['value'])
            elif indicator_type == "wilders":
                df_dict['wilders'].extend(json_data['value'])
            elif indicator_type == "williamsalligator":
                df_dict['williamsalligator'].extend(json_data['value'])
            elif indicator_type == "willr":
                df_dict['willr'].extend(json_data['value'])
            elif indicator_type == "xsidegap3methods":
                df_dict['xsidegap3methods'].extend(json_data['value'])
            elif indicator_type == "zlema":
                df_dict['zlema'].extend(json_data['value'])

            return df_dict


import requests


class ModelPredict:
    def __init__(self):
        self.model = Model()
        self.indicator_types = [
            "ema5", "ema15", "ema30", "ema60", "ema100", "ema200", "wma", "macd", "atr", "hma",
            "kama", "cmo", "candles",
            "abs", "accbands", "ad", "add", "adosc", "advanceblock", "adx", "adxr", "ao", "apo",
            "aroon", "aroonosc", "atan", "avgprice", "bbands", "belthold", "beta", "bop",
            "breakaway", "cci", "ceil", "chop", "closingmarubozu", "cmf", "concealbabyswall",
            "coppockcurve", "correl", "cos", "counterattack", "darkcloudcover", "dema", "div",
            "doji", "dojistar", "dragonflydoji", "dx",
            "engulfing", "eom", "eveningdojistar", "eveningstar",
            "floor", "gapsidesidewhite", "gravestonedoji", "hammer",
            "hangingman", "harami", "haramicross", "highwave", "hikkake", "hikkakemod",
            "homingpigeon", "ht_dcperiod", "ht_dcphase",
            "ht_trendline", "ht_trendmode", "identical3crows", "inneck",
            "invertedhammer", "kicking", "kickingbylength",
            "ladderbottom", "linearreg", "linearreg_angle",
            "linearreg_intercept", "linearreg_slope", "ln", "log10", "longleggeddoji",
            "longline", "ma", "marketfi", "marubozu", "matchinglow",
            "mathold", "max", "maxindex", "medprice", "mfi", "midpoint", "midprice", "min",
            "minindex", "minus_di", "minus_dm", "mom", "morningdojistar",
            "morningstar", "mul", "mult", "natr", "nvi", "obv", "onneck", "pd", "piercing",
            "plus_di", "plus_dm", "ppo", "price",
            "psar", "pvi", "rickshawman", "risefall3methods", "roc", "rocp", "rocr",
            "rocr100", "round", "rsi", "separatinglines", "shootingstar", "shortline", "sin",
            "sma", "smma", "spinningtop", "sqrt", "stalledpattern", "stddev", "sticksandwich",
            "sub", "sum", "supertrend", "t3", "takuri", "tan",
            "tanh", "tasukigap", "tema", "thrusting", "todeg", "torad", "tr",
            "trima", "tristar", "trix", "trunc", "tsf", "typprice", "ultosc", "unique3river",
            "upsidegap2crows", "var", "vosc", "vwap",
            "wad", "wclprice", "willr", "xsidegap3methods",
            "zlema"
        ]
        self.df_dict = {
            'Price': [], 'High': [], 'Low': [], 'Open': [], 'EMA_5': [], 'EMA_15': [], 'EMA_30': [],
            'EMA_60': [], 'EMA_100': [], 'EMA_200': [], 'WMA': [], 'MACD': [], 'MACD_Signal': [],
            'MACD_Hist': [], 'ATR': [], 'HMA': [], 'KAMA': [], 'CMO': [],
            'abs': [], 'valueUpperBand': [], 'valueMiddleBand': [], 'valueLowerBand': [], 'ad': [], 'add': [],
            'adosc': [], 'advanceblock': [], 'adx': [],
            'adxr': [], 'ao': [], 'apo': [], 'valueAroonDown': [], 'valueAroonUp': [], 'aroonosc': [], 'atan': [],
            'avgprice': [],
            'valueUpperBand1': [], 'valueMiddleBand1': [], 'valueLowerBand1': [], 'belthold': [], 'beta': [], 'bop': [],
            'breakaway': [], 'cci': [], 'ceil': [],
            'chop': [], 'closingmarubozu': [], 'cmf': [], 'concealbabyswall': [], 'coppockcurve': [],
            'correl': [], 'cos': [], 'counterattack': [], 'darkcloudcover': [], 'dema': [], 'div': [],
            'doji': [], 'dojistar': [],
            'dragonflydoji': [], 'dx': [], 'engulfing': [], 'eom': [], 'eveningdojistar': [],
            'eveningstar': [], 'floor': [],
            'gapsidesidewhite': [], 'gravestonedoji': [], 'hammer': [], 'hangingman': [], 'harami': [],
            'haramicross': [], 'highwave': [], 'hikkake': [], 'hikkakemod': [],
            'homingpigeon': [], 'ht_dcperiod': [], 'ht_dcphase': [],
            'ht_trendline': [], 'ht_trendmode': [], 'identical3crows': [], 'inneck': [],
            'invertedhammer': [], 'kicking': [],
            'kickingbylength': [], 'ladderbottom': [],
            'linearreg': [], 'linearreg_angle': [], 'linearreg_intercept': [], 'linearreg_slope': [],
            'ln': [], 'log10': [], 'longleggeddoji': [], 'longline': [], 'ma': [],
            'marketfi': [], 'marubozu': [], 'matchinglow': [], 'mathold': [],
            'max': [], 'maxindex': [], 'medprice': [], 'mfi': [], 'midpoint': [], 'midprice': [],
            'min': [], 'minindex': [], 'minus_di': [], 'minus_dm': [],
            'mom': [], 'morningdojistar': [], 'morningstar': [], 'mul': [], 'mult': [],
            'natr': [], 'nvi': [], 'obv': [], 'onneck': [], 'pd': [], 'piercing': [],
            'plus_di': [], 'plus_dm': [], 'ppo': [], 'price': [],
            'psar': [], 'pvi': [], 'rickshawman': [], 'risefall3methods': [], 'roc': [],
            'rocp': [], 'rocr': [], 'rocr100': [], 'round': [], 'rsi': [], 'separatinglines': [],
            'shootingstar': [], 'shortline': [], 'sin': [], 'sma': [], 'smma': [], 'spinningtop': [],
            'sqrt': [], 'stalledpattern': [], 'stddev': [], 'sticksandwich': [], 'sub': [], 'sum': [], 'supertrend': [],
            't3': [], 'takuri': [], 'tan': [],
            'tanh': [], 'tasukigap': [], 'tema': [], 'thrusting': [], 'todeg': [],
            'torad': [], 'tr': [], 'trima': [], 'tristar': [], 'trix': [], 'trunc': [], 'tsf': [],
            'typprice': [], 'ultosc': [], 'unique3river': [], 'upsidegap2crows': [], 'var': [],
            'vosc': [], 'vwap': [], 'wad': [],
            'wclprice': [], 'willr': [], 'xsidegap3methods': [],
            'zlema': []
        }

    def doc_du_lieu_tu_file(self, ten_file):
        with open(ten_file, 'r') as file:
            self.df_dict = {
                'Price': [], 'High': [], 'Low': [], 'Open': [], 'EMA_5': [], 'EMA_15': [], 'EMA_30': [],
                'EMA_60': [], 'EMA_100': [], 'EMA_200': [], 'WMA': [], 'MACD': [], 'MACD_Signal': [],
                'MACD_Hist': [], 'ATR': [], 'HMA': [], 'KAMA': [], 'CMO': [],
                'abs': [], 'valueUpperBand': [], 'valueMiddleBand': [], 'valueLowerBand': [], 'ad': [], 'add': [],
                'adosc': [], 'advanceblock': [], 'adx': [],
                'adxr': [], 'ao': [], 'apo': [], 'valueAroonDown': [], 'valueAroonUp': [], 'aroonosc': [], 'atan': [],
                'avgprice': [],
                'valueUpperBand1': [], 'valueMiddleBand1': [], 'valueLowerBand1': [], 'belthold': [], 'beta': [],
                'bop': [],
                'breakaway': [], 'cci': [], 'ceil': [],
                'chop': [], 'closingmarubozu': [], 'cmf': [], 'concealbabyswall': [], 'coppockcurve': [],
                'correl': [], 'cos': [], 'counterattack': [], 'darkcloudcover': [], 'dema': [], 'div': [],
                'doji': [], 'dojistar': [],
                'dragonflydoji': [], 'dx': [], 'engulfing': [], 'eom': [], 'eveningdojistar': [],
                'eveningstar': [], 'floor': [],
                'gapsidesidewhite': [], 'gravestonedoji': [], 'hammer': [], 'hangingman': [], 'harami': [],
                'haramicross': [], 'highwave': [], 'hikkake': [], 'hikkakemod': [],
                'homingpigeon': [], 'ht_dcperiod': [], 'ht_dcphase': [],
                'ht_trendline': [], 'ht_trendmode': [], 'identical3crows': [], 'inneck': [],
                'invertedhammer': [], 'kicking': [],
                'kickingbylength': [], 'ladderbottom': [],
                'linearreg': [], 'linearreg_angle': [], 'linearreg_intercept': [], 'linearreg_slope': [],
                'ln': [], 'log10': [], 'longleggeddoji': [], 'longline': [], 'ma': [],
                'marketfi': [], 'marubozu': [], 'matchinglow': [], 'mathold': [],
                'max': [], 'maxindex': [], 'medprice': [], 'mfi': [], 'midpoint': [], 'midprice': [],
                'min': [], 'minindex': [], 'minus_di': [], 'minus_dm': [],
                'mom': [], 'morningdojistar': [], 'morningstar': [], 'mul': [], 'mult': [],
                'natr': [], 'nvi': [], 'obv': [], 'onneck': [], 'pd': [], 'piercing': [],
                'plus_di': [], 'plus_dm': [], 'ppo': [], 'price': [],
                'psar': [], 'pvi': [], 'rickshawman': [], 'risefall3methods': [], 'roc': [],
                'rocp': [], 'rocr': [], 'rocr100': [], 'round': [], 'rsi': [], 'separatinglines': [],
                'shootingstar': [], 'shortline': [], 'sin': [], 'sma': [], 'smma': [], 'spinningtop': [],
                'sqrt': [], 'stalledpattern': [], 'stddev': [], 'sticksandwich': [], 'sub': [], 'sum': [],
                'supertrend': [],
                't3': [], 'takuri': [], 'tan': [],
                'tanh': [], 'tasukigap': [], 'tema': [], 'thrusting': [], 'todeg': [],
                'torad': [], 'tr': [], 'trima': [], 'tristar': [], 'trix': [], 'trunc': [], 'tsf': [],
                'typprice': [], 'ultosc': [], 'unique3river': [], 'upsidegap2crows': [], 'var': [],
                'vosc': [], 'vwap': [], 'wad': [],
                'wclprice': [], 'willr': [], 'xsidegap3methods': [],
                'zlema': []
            }
            for line in file:
                print(line)
                data = line.strip().split(': ')
                data_list = ast.literal_eval(data[1])
                self.df_dict[data[0]].extend(data_list)
                # if value == '[]':
                #     value = []
                # self.df_dict[key] = value

    def ghi_du_lieu_vao_file(self, ten_file):
        with open(ten_file, 'w') as file:
            for key, value in self.df_dict.items():
                file.write(f"{key}: {value}\n")

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
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        scaled_data = self.scaler.fit_transform(df)
        return scaled_data

    def train_model(self, X_train, y_train, epochs=100, batch_size=32):
        print("Train Model ...")
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)
        save_model(self.model, f"{name_mlbict}.h5")
        print("Model Trained successfully! And saved model successfully!")

    def predict(self, X):
        print("Predict Model ...")
        predict_data = self.model.predict(X)
        return predict_data

    def run(self, mode='Training', epochs=100, batch_size=32):
        global backtrack
        blackstrack_demo = backtrack
        self.df_dict.clear()
        self.df_dict = {
            'Price': [], 'High': [], 'Low': [], 'Open': [], 'EMA_5': [], 'EMA_15': [], 'EMA_30': [],
            'EMA_60': [], 'EMA_100': [], 'EMA_200': [], 'WMA': [], 'MACD': [], 'MACD_Signal': [],
            'MACD_Hist': [], 'ATR': [], 'HMA': [], 'KAMA': [], 'CMO': [],
            'abs': [], 'valueUpperBand': [], 'valueMiddleBand': [], 'valueLowerBand': [], 'ad': [], 'add': [],
            'adosc': [], 'advanceblock': [], 'adx': [],
            'adxr': [], 'ao': [], 'apo': [], 'valueAroonDown': [], 'valueAroonUp': [], 'aroonosc': [], 'atan': [],
            'avgprice': [],
            'valueUpperBand1': [], 'valueMiddleBand1': [], 'valueLowerBand1': [], 'belthold': [], 'beta': [], 'bop': [],
            'breakaway': [], 'cci': [], 'ceil': [],
            'chop': [], 'closingmarubozu': [], 'cmf': [], 'concealbabyswall': [], 'coppockcurve': [],
            'correl': [], 'cos': [], 'counterattack': [], 'darkcloudcover': [], 'dema': [], 'div': [],
            'doji': [], 'dojistar': [],
            'dragonflydoji': [], 'dx': [], 'engulfing': [], 'eom': [], 'eveningdojistar': [],
            'eveningstar': [], 'floor': [],
            'gapsidesidewhite': [], 'gravestonedoji': [], 'hammer': [], 'hangingman': [], 'harami': [],
            'haramicross': [], 'highwave': [], 'hikkake': [], 'hikkakemod': [],
            'homingpigeon': [], 'ht_dcperiod': [], 'ht_dcphase': [],
            'ht_trendline': [], 'ht_trendmode': [], 'identical3crows': [], 'inneck': [],
            'invertedhammer': [], 'kicking': [],
            'kickingbylength': [], 'ladderbottom': [],
            'linearreg': [], 'linearreg_angle': [], 'linearreg_intercept': [], 'linearreg_slope': [],
            'ln': [], 'log10': [], 'longleggeddoji': [], 'longline': [], 'ma': [],
            'marketfi': [], 'marubozu': [], 'matchinglow': [], 'mathold': [],
            'max': [], 'maxindex': [], 'medprice': [], 'mfi': [], 'midpoint': [], 'midprice': [],
            'min': [], 'minindex': [], 'minus_di': [], 'minus_dm': [],
            'mom': [], 'morningdojistar': [], 'morningstar': [], 'mul': [], 'mult': [],
            'natr': [], 'nvi': [], 'obv': [], 'onneck': [], 'pd': [], 'piercing': [],
            'plus_di': [], 'plus_dm': [], 'ppo': [], 'price': [],
            'psar': [], 'pvi': [], 'rickshawman': [], 'risefall3methods': [], 'roc': [],
            'rocp': [], 'rocr': [], 'rocr100': [], 'round': [], 'rsi': [], 'separatinglines': [],
            'shootingstar': [], 'shortline': [], 'sin': [], 'sma': [], 'smma': [], 'spinningtop': [],
            'sqrt': [], 'stalledpattern': [], 'stddev': [], 'sticksandwich': [], 'sub': [], 'sum': [], 'supertrend': [],
            't3': [], 'takuri': [], 'tan': [],
            'tanh': [], 'tasukigap': [], 'tema': [], 'thrusting': [], 'todeg': [],
            'torad': [], 'tr': [], 'trima': [], 'tristar': [], 'trix': [], 'trunc': [], 'tsf': [],
            'typprice': [], 'ultosc': [], 'unique3river': [], 'upsidegap2crows': [], 'var': [],
            'vosc': [], 'vwap': [], 'wad': [],
            'wclprice': [], 'willr': [], 'xsidegap3methods': [],
            'zlema': []
        }
        symbols, interval = "BTC/USDT", "15m"
        print(f"Predicting {symbols}, interval: {interval}")
        print(f"Size {blackstrack_demo * blackstrack_demo} candles")
        # self.read_csv_to_dict("power_data.csv")
        while blackstrack_demo >= 0:
            try:
                print(f"Backtrack: {blackstrack_demo}")
                print("Calling API ...")
                for indicator in self.indicator_types:
                    print(f"Indicator: {indicator}")
                    response = self.model.call_api_each(indicator, blackstrack_demo, symbols, interval)
                    if response.status_code == 200:
                        self.df_dict = self.model.process_json_to_dataframe(response.json(), indicator, self.df_dict)
                    else:
                        print("API call failed with status code:", response.status_code)
                blackstrack_demo -= 1
                print(f"Finished {blackstrack_demo}")
            except Exception as e:  # Catch any exceptions
                print(f"Error encountered: {e}")
                print("Pausing for 10 seconds...")
                time.sleep(10)
        min_length = len(self.df_dict['Price'])
        for key, value in self.df_dict.items():
            if len(value) < min_length:
                min_length = len(value)
        for key, value in self.df_dict.items():
            self.df_dict[key] = value[:min_length]
        historical_data = pd.DataFrame(self.df_dict)
        scaled_data = self.preprocess_data(historical_data)
        X_train = []
        y_train = []
        print(self.df_dict)
        print(historical_data)
        # Xây dựng dữ liệu đầu vào và đầu ra cho mô hình
        for i in range(60, len(scaled_data)):
            X_train.append(scaled_data[i - 60:i, :])
            y_train.append(scaled_data[i, 0])  # Giá là cột đầu tiên
        X_train, y_train = np.array(X_train), np.array(y_train)
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], X_train.shape[2]))
        self.build_model(input_shape=(X_train.shape[1], X_train.shape[2]))
        predicted_price = self.predict(X_train[-1].reshape(1, X_train.shape[1], X_train.shape[2]))
        last_scaled_data_row = scaled_data[-1]
        last_scaled_data_first_column = last_scaled_data_row[0]
        print("Predicteding Completed!")
        print("Price Scale:", predicted_price)
        if last_scaled_data_first_column < predicted_price:
            signal = "Buy"
        else:
            signal = "Sell"
        return signal


if __name__ == "__main__":
    model = ModelPredict()
    signal = model.run()
    print(signal)
