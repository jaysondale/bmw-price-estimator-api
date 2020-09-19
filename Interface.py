from sklearn.linear_model import LinearRegression
import numpy as np

def getResult(year, model, odometer, condition, engine, transmission, cylinders, drive):
    linreg = LinearRegression()
    # Pretrained coefficients
    linreg.coef_ = np.array([ 4.15688785e+02, -4.43167349e-02,  2.12651230e-10, -4.29025704e-11,
        2.51503707e-10,  8.23214342e+02, -6.41425274e+02, -1.42694245e+03,
        9.45874490e-11, -1.32009177e+02, -3.30702492e+03, -5.98655733e+02,
       -3.25718092e+02, -4.17621623e+02,  9.98182003e+02,  5.17219110e+02,
        6.56854016e+02,  3.71803894e+02,  2.42936761e+03, -1.84882795e+02,
       -5.00999670e+02,  4.94002073e+02, -2.02945042e+03, -5.59599756e+02,
       -2.05897742e+03, -2.34529423e+03,  4.04687899e+02,  1.52142986e+03,
        9.98071313e+02,  1.98711159e+02, -8.07920998e+02, -1.82481530e+03,
        1.21496766e+03,  8.08414295e+02, -2.64840938e+03,  5.51285004e+03,
       -1.12757053e+03, -8.81546752e+02, -1.11122893e+03,  3.93289308e+02,
        9.87711205e+02,  2.20741028e+03,  1.33915108e+03, -3.31410140e+02,
       -8.48852168e+02, -1.15545497e+02])
    
    linreg.intercept_ = -818769.414838612
    
    input_arr = processInputs(year, model, odometer, condition, engine, transmission, cylinders, drive)
    result = linreg.predict(input_arr)
    return result[0]

'''
ORDER OF INPUTS
['year', 'odometer', '10 cylinders', '12 cylinders', '3 cylinders',
       '4 cylinders', '6 cylinders', '8 cylinders', 'other', 'excellent',
       'fair', 'good', 'like new', 'automatic', 'manual', 'diesel', 'electric',
       'gas', 'hybrid', '3 series', '3-series', '325i', '328i', '328i xdrive',
       '328xi', '330i', '335i', '4 series', '5 series', '5-series', '528i',
       '530i', '535i', '7 series', 'i3', 'm3', 'x1', 'x1 xdrive28i', 'x3',
       'x3 xdrive28i', 'x5', 'x5 xdrive35i', 'x5 xdrive35i awd suv',
       'drive_4wd', 'drive_fwd', 'drive_rwd']
'''
# Onehot keys
model_keys = ['3 series', '3-series', '325i',
       '328i', '328i xdrive', '328xi', '330i', '335i', '4 series', '5 series',
       '5-series', '528i', '530i', '535i', '7 series', 'i3', 'm3', 'x1',
       'x1 xdrive28i', 'x3', 'x3 xdrive28i', 'x5', 'x5 xdrive35i',
       'x5 xdrive35i awd suv']
condition_keys = ['excellent', 'fair', 'good', 'like new']
transmission_keys = ['automatic', 'manual']
engine_keys = ['diesel', 'electric', 'gas', 'hybrid']
cylinder_keys = ['10 cylinders', '12 cylinders',
       '3 cylinders', '4 cylinders', '6 cylinders', '8 cylinders', 'other']
drive_keys = ['drive_4wd', 'drive_fwd', 'drive_rwd']

def oneHotEncode(value, key_set):
    return_set = [0 for _ in range(len(key_set))]
    for key in range(len(key_set)):
        if key_set[key] == value:
            return_set[key] = 1
            break
    return return_set

       
def processInputs(year, model, odometer, condition, engine, transmission, cylinders, drive):
    # Encode parameters
    model_hot = oneHotEncode(model, model_keys)
    condition_hot = oneHotEncode(condition, condition_keys)
    engine_hot = oneHotEncode(engine, engine_keys)
    transmission_hot = oneHotEncode(transmission, transmission_keys)
    cylinder_hot = oneHotEncode(cylinders, cylinder_keys)
    drive_hot = oneHotEncode(drive, drive_keys)

    # Generate input array
    input_raw = [year, odometer] + cylinder_hot + condition_hot + transmission_hot + engine_hot + model_hot + drive_hot
    input_arr = np.array([input_raw])

    # Compute result
    return input_arr