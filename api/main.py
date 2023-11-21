from fastapi import FastAPI, HTTPException
import numpy as np
import pickle
import uvicorn
import logging

from models import Item, Items
from utils import prepare_other, prepare_torque, fill_missing

app = FastAPI()

with open('../ckpt/ckpt.pkl', 'rb') as f:
    dump_dct = pickle.load(f)

model = dump_dct['regression_model']
scaler = dump_dct['standart_scaler']
encoder = dump_dct['one_hot_encoder']
median = dump_dct['data_median']

logger = logging.getLogger('uvicorn')
formatter = logging.Formatter('%(asctime)s [%(levelname)-5.5s] %(message)s')

file_handler = logging.FileHandler('../logs/api.log')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

logger.setLevel('DEBUG')

status_codes = {
    'format': 480,
    'less_than_zero': 481,
    'zero_division': 482,
    'one_hot_encoder': 483
}


@app.post("/predict_item")
def predict_item(item: Item) -> float:
    '''
        Предсказание на одном объекте.

        Parameters
        ----------
        item : Item
            Признаковое описание объекта.

        Returns
        -------
        float
            Цена объекта.
    '''
    logger.debug(f'predicting on item = {item}')
    try:
        torque, max_torque_rpm = prepare_torque(item.torque)
        logger.debug(
            f'initial torque = {item.torque} -> torque = {torque}, '
            f'max_torque_rpm = {max_torque_rpm}'
        )
    except ValueError:
        raise HTTPException(
            status_code=status_codes['format'],
            detail='Неверный формат ввода параметра torque.'
        )

    try:
        mileage = prepare_other(item.mileage)
        logger.debug(
            f'initial mileage = {item.mileage} -> mileage = {mileage}'
        )
    except ValueError:
        raise HTTPException(
            status_code=status_codes['format'],
            detail='Неверный формат ввода параметра mileage.'
        )

    try:
        engine = prepare_other(item.engine)
        logger.debug(
            f'initial engine = {item.engine} -> engine = {engine}'
        )
    except ValueError:
        raise HTTPException(
            status_code=status_codes['format'],
            detail='Неверный формат ввода параметра engine.'
        )

    try:
        max_power = prepare_other(item.max_power)
        logger.debug(
            f'initial max_power = {item.max_power} -> max_power = {max_power}'
        )
    except ValueError:
        raise HTTPException(
            status_code=status_codes['format'],
            detail='Неверный формат ввода параметра max_power.'
        )

    if torque < 0:
        raise HTTPException(
            status_code=status_codes['less_than_zero'],
            detail='Torque параметра torque не может быть отрицательным.'
        )

    if max_torque_rpm < 0:
        raise HTTPException(
            status_code=status_codes['less_than_zero'],
            detail='RPM параметра torque не может быть отрицательным.'
        )

    if mileage < 0:
        raise HTTPException(
            status_code=status_codes['less_than_zero'],
            detail='Параметр mileage не может быть отрицательным.'
        )

    if engine < 0:
        raise HTTPException(
            status_code=status_codes['less_than_zero'],
            detail='Параметр engine не может быть отрицательным.'
        )

    if max_power < 0:
        raise HTTPException(
            status_code=status_codes['less_than_zero'],
            detail='Параметр max_power не может быть отрицательным.'
        )

    try:
        power_per_liter = max_power / engine * 1000
        mileage_per_liter = mileage / engine * 1000
        logger.debug(
            f'max_power = {max_power}, engine = {engine} -> '
            f'power_per_liter = {power_per_liter}, '
            f'mileage_per_liter = {mileage_per_liter}'
        )
    except ZeroDivisionError:
        raise HTTPException(
            status_code=status_codes['zero_division'],
            detail='Параметр engine не может быть нулевым.'
        )

    year = item.year
    km_driven = item.km_driven

    if km_driven < 0:
        raise HTTPException(
            status_code=status_codes['less_than_zero'],
            detail='Параметр km_driven не может быть отрицательным.'
        )

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
    logger.debug('missing filled')

    if np.isnan(item.seats):
        seats = str(int(median['seats']))
    else:
        seats = str(int(item.seats))
    logger.debug(
        f'initial seats = {item.seats} -> seats = {seats}'
    )

    name = item.name.split()[0]
    categorical_data = np.array([
        name,
        item.fuel,
        item.seller_type,
        item.transmission,
        item.owner,
        seats
    ]).reshape(1, -1)
    try:
        one_hot_data = encoder.transform(categorical_data)
        logger.debug('encoder into one-hot')
    except ValueError:
        raise HTTPException(
            status_code=status_codes['one_hot_encoder'],
            detail='Категориальные признаки имеют ограниченный набор значений.'
        )

    sample_to_scale = np.array(filled_missing).reshape(1, -1)

    sample = scaler.transform(sample_to_scale)
    logger.debug('features scaled')

    sample = np.hstack((
        sample,
        one_hot_data
    ))
    logger.debug(f'model input = {sample}')

    pred = model.predict(sample)
    return pred


@app.post("/predict_items")
def predict_items(items: Items) -> list[float]:
    '''
        Предсказание на нескольких объектах.

        Parameters
        ----------
        items : Items
            Признаковое описание объектов.

        Returns
        -------
        list[float]
            Цена каждого объекта.
    '''
    pred = []
    for item in items.objects:
        pred += [predict_item(item)]
    return pred


if __name__ == '__main__':
    uvicorn.run('main:app', host='0.0.0.0', port=8000)
