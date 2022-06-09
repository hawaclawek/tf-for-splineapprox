from multiprocessing import Pool
from functools import partial
from tensorflow import keras
import numpy as np
from tfspline import model
import inspect

# https://docs.python.org/2/library/multiprocessing.html#windows

def job(param, kwargs):
    if 'data_x' in kwargs:
        data_x = kwargs['data_x']
    else:
        raise Exception("Missing x-data")
    if 'data_y' in kwargs:
        data_y = kwargs['data_y']
    else:
        raise Exception("Missing y-data")
    if 'degree' in kwargs:
        degree = kwargs['degree']
    else:
        degree = 5
    if 'polynum' in kwargs:
        polynum = kwargs['polynum']
    else:
        polynum = 1
    if 'learning_rate' in kwargs:
        learning_rate = kwargs['learning_rate']
    else:
        learning_rate = 0.1
    if 'continuity' in kwargs:
        continuity = kwargs['continuity']
    else:
        continuity = False
    if 'optimizer' in kwargs: # all arguments to Process.__init__() need to be picklable. This is not the case for keras optimizers
        optimizer = kwargs['optimizer']
    else:
        optimizer = 'AMSGrad'
    if 'n_epochs' in kwargs:
        n_epochs = kwargs['n_epochs']
    else:
        raise Exception('Missing epochs parameter')
    if 'factor_ck_pressure' in kwargs:
        factor_ck_pressure = kwargs['factor_ck_pressure']
    else:
        factor_ck_pressure = 0
    if 'factor_approximation_quality' in kwargs:
        factor_approximation_quality = kwargs['factor_approximation_quality']
    else:
        factor_approximation_quality = 1
    if 'factor_curvature' in kwargs:
        factor_curvature = kwargs['factor_curvature']
    else:
        factor_curvature = 0
    if 'seg_overlap' in kwargs:
        seg_overlap = kwargs['seg_overlap']
    else:
        seg_overlap = 0.4
    if 'gradient_regularization' in kwargs:
        gradient_regularization = kwargs['gradient_regularization']
    else:
        gradient_regularization = True
    if 'ck' in kwargs:
        ck = kwargs['ck']
    else:
        ck = 2
    if 'mode' in kwargs:
        mode = kwargs['mode']
    else:
        raise Exception('Missing mode!')
    if 'initialize_l2_fit' in kwargs:
        initialize_l2_fit = kwargs['initialize_l2_fit']
    else:
        initialize_l2_fit = False
    if 'shift_polynomial_centers' in kwargs:
        shift_polynomial_centers = kwargs['shift_polynomial_centers']
    else:
        shift_polynomial_centers = 'mean'
    if 'split_uniform' in kwargs:
        split_uniform = kwargs['split_uniform']
    else:
        split_uniform = True

    if optimizer.upper() == 'ADAM':
        opt = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=False)
    elif optimizer.upper() == 'AMSGRAD':
        opt = keras.optimizers.Adam(learning_rate=learning_rate, amsgrad=True)
    else:
        opt = keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.95, nesterov=True)

    spline = model.Spline(polydegree=degree, polynum=polynum, ck=ck)
    spline.continuity = continuity
    
    if mode == 'approx_ck':
        if param < 0 or param > 1:
            raise Exception("Factor_approximation_quality must be between 0 and 1")
        factor_approximation_quality = param
        factor_ck_pressure = 1 - param
        factor_curvature = 0

    print(".", end="")
    spline.fit(data_x, data_y, optimizer=opt, n_epochs=n_epochs, factor_approximation_quality=factor_approximation_quality,
        factor_ck_pressure=factor_ck_pressure, factor_curvature=factor_curvature, gradient_regularization=gradient_regularization, overlap_segments=seg_overlap,
        initialize_l2fit=initialize_l2_fit, shift_polynomial_centers=shift_polynomial_centers, uniform_split=split_uniform)
    
    print("#", end="")
    
    return [{'optimizer': optimizer, 'param_value': param}, [spline.total_loss_values, spline.e_loss_values, spline.D_loss_values], spline.coeffs]


if __name__ == '__main__':
    p = Process(target=sweep_overlap)
    p.start()
