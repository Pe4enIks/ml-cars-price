import numpy as np
import pandas as pd


def kgm_to_nm(val: float) -> float:
    '''
        Конвертация из кгc-м в Н-м.

        Parameters
        ----------
        val : float
            Значение в кгс-м.

        Returns
        -------
        float
            Значение в Н-м.
    '''
    return val * 9.80665


def convert_to_float(text: str) -> float:
    '''
        Конвертация из str в float.

        Parameters
        ----------
        text : str
            Строка для конвертации.

        Returns
        -------
        float
            Значение в float.
    '''
    try:
        tmp_rmp_val = text.split(',')
        if len(tmp_rmp_val) > 1:
            rpm_val = 1000 * float(tmp_rmp_val[0]) + float(tmp_rmp_val[1])
        else:
            rpm_val = float(tmp_rmp_val[0])
    except:
        return np.nan
    return rpm_val


def convert_rpm(text: str) -> float:
    '''
        Конвертация RPM из строки с лишними символами в float.

        Parameters
        ----------
        text : str
            Строка для конвертации.

        Returns
        -------
        float
            Значение в float.
    '''
    tmp_text = text[:-3] if 'rpm' in text.lower() else text
    if '~' in text:
        rpms = tmp_text.split('~')
    else:
        rpms = tmp_text.split('-')
    if len(rpms) > 1:
        rpm_val = convert_to_float(rpms[1].strip())
    else:
        rpm_val = convert_to_float(rpms[0].strip())
    return rpm_val


def prepare_torque(val: str) -> tuple[float, float]:
    '''
        Конвертация torque из строки с лишними символами в два значения float.
        На выходе - torque и max_torque_rpm.

        Parameters
        ----------
        val : str
            Строка для конвертации.

        Returns
        -------
        tuple[float, float]
            Значения torque и max_torque_rpm.
    '''
    if str(val) == 'nan':
        return np.nan, np.nan
    try:
        if '@' in val:
            lst = val.split('@')
        elif 'at' in val:
            lst = val.split('at')
        elif '/' in val:
            lst = val.split('/')
        else:
            lst = []
        nm_val, rpm_val = np.nan, np.nan
        if len(lst) == 2:
            rpm_val = convert_rpm(lst[1])
            if 'kgm' in lst[0].lower():
                nm_val = kgm_to_nm(convert_to_float(lst[0][:-3].split()[0]))
            else:
                nm_val = convert_to_float(lst[0][:-2].split()[0])
        elif len(lst) == 3:
            rpm_val, units = lst[1].split('(')
            rpm_val = convert_rpm(rpm_val)
            if 'kgm' in units.lower():
                nm_val = kgm_to_nm(float(lst[0]))
            else:
                nm_val = float(lst[0])
        return nm_val, rpm_val
    except:
        if 'Nm' in val:
            lst = val.split()
            if len(lst) == 1:
                return float(lst[0][:-2]), np.nan
            else:
                return float(lst[0]), np.nan
        return np.nan, np.nan


def prepare_other(val: str) -> float:
    '''
        Конвертация любой строки формата "число что-то" или "число".

        Parameters
        ----------
        val : str
            Строка для конвертации.

        Returns
        -------
        float
            Значение в float.
    '''
    if str(val) == 'nan':
        return np.nan
    try:
        num, _ = val.split()
    except:
        try:
            return float(val)
        except:
            return np.nan
    return float(num)


def fill_missing(
    values: list[tuple[str, float]],
    median: pd.Series
) -> list[float]:
    '''
        Заполнение пропущенных значений медианой из train датасета.

        Parameters
        ----------
        values : list[tuple[str, float]]
            Список объектов-пар: название признака - значение.

        Returns
        -------
        list[float]
            Список значений без пропусков.
    '''
    filled_values = []
    for name, value in values:
        if np.isnan(value):
            filled_values += [float(median[name])]
        else:
            filled_values += [float(value)]
    return filled_values


if __name__ == '__main__':
    torque = '450 Nm @ 2500 rpm'
    mileage = '21.78 kmpl'

    nm_val, rpm_val = prepare_torque(torque)
    mil = prepare_other(mileage)

    print(nm_val, rpm_val, mil)
