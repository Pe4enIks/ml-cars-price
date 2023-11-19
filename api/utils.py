import numpy as np


def kgm_to_nm(val):
    return val * 9.80665


def convert_to_float(text):
    try:
        tmp_rmp_val = text.split(',')
        if len(tmp_rmp_val) > 1:
            rpm_val = 1000 * float(tmp_rmp_val[0]) + float(tmp_rmp_val[1])
        else:
            rpm_val = float(tmp_rmp_val[0])
    except:
        return np.nan
    return rpm_val


def convert_rpm(text):
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


def prepare_torque(val):
    if str(val) == 'nan':
        return val
    try:
        if '@' in val:
            lst = val.split('@')
        elif 'at' in val:
            lst = val.split('at')
        elif '/' in val:
            lst = val.split('/')
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


def prepare_other(val):
    if str(val) == 'nan':
        return val
    try:
        num, _ = val.split()
    except:
        try:
            return float(val)
        except:
            return np.nan
    return float(num)


def fill_missing(values, median):
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
