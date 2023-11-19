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
        v1, _ = val.split()
    except:
        try:
            return float(val)
        except:
            return np.nan
    return float(v1)


def prepare_seats(val):
    map_seats_to_ind = {
        4: 0,
        5: 1,
        6: 2,
        7: 3,
        8: 4,
        9: 5,
        10: 6,
        14: 7
    }
    one_hot = [0] * 8
    int_val = int(val)
    if int_val not in map_seats_to_ind.keys():
        return np.array(one_hot)
    one_hot[map_seats_to_ind[int_val]] = 1
    return np.array(one_hot)


def prepare_fuel(val):
    one_hot = [0] * 3

    if val == 'Diesel':
        one_hot[0] = 1
    elif val == 'LPG':
        one_hot[1] = 1
    elif val == 'Petrol':
        one_hot[2] = 1

    return np.array(one_hot)


def prepare_seller(val):
    one_hot = [0] * 2

    if val == 'Individual':
        one_hot[0] = 1
    elif val == 'Trustmark Dealer':
        one_hot[1] = 1

    return np.array(one_hot)


def prepare_transmission(val):
    one_hot = [0]

    if val == 'Manual':
        one_hot[0] = 1

    return np.array(one_hot)


def prepare_owner(val):
    one_hot = [0] * 4

    if val == 'Fourth & Above Owner':
        one_hot[0] = 1
    elif val == 'Second Owner':
        one_hot[1] = 1
    elif val == 'Test Drive Car':
        one_hot[2] = 1
    elif val == 'Third Owner':
        one_hot[3] = 1

    return np.array(one_hot)


if __name__ == '__main__':
    torque = '450 Nm @ 2500 rpm'
    mileage = '21.78 kmpl'

    nm_val, rpm_val = prepare_torque(torque)
    mil = prepare_other(mileage)

    print(nm_val, rpm_val, mil)
