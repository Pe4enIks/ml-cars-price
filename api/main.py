from fastapi import FastAPI
import numpy as np
import pickle
import uvicorn

from models import Item, Items
from utils import prepare_other, prepare_torque, fill_missing

app = FastAPI()

with open('../ckpt/ckpt.pkl', 'rb') as f:
    dump_dct = pickle.load(f)

model = dump_dct['regression_model']
scaler = dump_dct['standart_scaler']
encoder = dump_dct['one_hot_encoder']
median = dump_dct['data_median']


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    torque, max_torque_rpm = prepare_torque(item.torque)

    mileage = prepare_other(item.mileage)
    engine = prepare_other(item.engine)
    max_power = prepare_other(item.max_power)

    power_per_liter = max_power / engine * 1000
    mileage_per_liter = mileage / engine * 1000

    year = item.year
    km_driven = item.km_driven

    to_fill = [
        ('year', year),
        ('km_driven', km_driven),
        ('mileage', mileage),
        ('engine', engine),
        ('max_power', max_power),
        ('torque', torque),
        ('max_torque_rpm', max_torque_rpm),
        ('power_per_liter', power_per_liter),
        ('mileage_per_liter', mileage_per_liter)
    ]
    filled_missing = fill_missing(to_fill, median)

    if np.isnan(item.seats):
        seats = str(int(median['seats']))
    else:
        seats = str(int(item.seats))

    categorical_data = np.array([
        item.fuel,
        item.seller_type,
        item.transmission,
        item.owner,
        seats
    ]).reshape(1, -1)
    one_hot_data = encoder.transform(categorical_data)

    sample_to_scale = np.array(filled_missing).reshape(1, -1)

    sample = scaler.transform(sample_to_scale)
    sample = np.hstack((
        sample,
        one_hot_data
    ))

    pred = model.predict(sample)
    return pred


@app.post("/predict_items")
def predict_items(items: Items) -> list[float]:
    pred = []
    for item in items.objects:
        pred += [predict_item(item)]
    return pred


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
