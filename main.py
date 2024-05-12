from typing import Optional

from fastapi import FastAPI

from Logic.ModelPredict import ModelPredict

app = FastAPI()
@app.route('/api/data', methods=['GET'])
def get_data():
    model = ModelPredict()
    signal = model.run()
    return signal

@app.route('/get_signal', methods=['GET'])
def get_data_signal():
    # model = ModelPredict()
    # signal = model.run()
    return "signal"
