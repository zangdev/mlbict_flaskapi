import time

from fastapi import FastAPI
from Logic.ModelPredict import ModelPredict

app = FastAPI()


@app.get("/get_signal")
def get_data():
    model = ModelPredict()
    signal = model.run()
    return {
        "time": time.strftime('%Y-%m-%d %H:%M:%S'),
        "signal": signal
    }
