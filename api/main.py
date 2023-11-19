from fastapi import FastAPI
import numpy as np
import pickle
import uvicorn

from models import Item, Items
from utils import prepare_other, prepare_torque, prepare_seats, prepare_fuel, \
    prepare_seller, prepare_transmission, prepare_owner

app = FastAPI()

with open('../ckpt/ckpt.pkl', 'rb') as f:
    dump_dct = pickle.load(f)

model = dump_dct['model']
scaler = dump_dct['scaler']
median = dump_dct['median']


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

    if np.isnan(km_driven):
        km_driven = median['km_driven']

    if np.isnan(mileage):
        mileage = median['mileage']

    if np.isnan(engine):
        engine = median['engine']

    if np.isnan(max_power):
        max_power = median['max_power']

    if np.isnan(torque):
        torque = median['torque']

    if np.isnan(max_torque_rpm):
        max_torque_rpm = median['max_torque_rpm']

    if np.isnan(item.seats):
        seats = median['seats']
    else:
        seats = item.seats

    one_hot_fuel = prepare_fuel(item.fuel)
    one_hot_seller = prepare_seller(item.seller_type)
    one_hot_transmission = prepare_transmission(item.transmission)
    one_hot_owner = prepare_owner(item.owner)
    one_hot_seats = prepare_seats(seats)

    sample_to_scale = np.array([
        float(year),
        float(km_driven),
        float(mileage),
        float(engine),
        float(max_power),
        float(torque),
        float(max_torque_rpm),
        float(power_per_liter),
        float(mileage_per_liter),
    ]).reshape(1, -1)

    sample = scaler.transform(sample_to_scale)
    sample = np.hstack((
        sample,
        one_hot_fuel.reshape(1, -1),
        one_hot_seller.reshape(1, -1),
        one_hot_transmission.reshape(1, -1),
        one_hot_owner.reshape(1, -1),
        one_hot_seats.reshape(1, -1)
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
