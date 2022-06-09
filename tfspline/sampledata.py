 #!/usr/bin/env python

"""sampledata.py: Helper functions to generate sample input data.  
"""


import numpy as np
import math
from numpy.core.function_base import linspace
import sklearn.preprocessing as preprocessing


__author__ = "Hannes Waclawek"
__version__ = "1.0"
__email__ = "hannes.waclawek@fh-salzburg.ac.at"


def generate_sample_data(n = 100, d = 5, noise_factor=0):
    """Generate sample data consisting of n points.

    Uses polynomial of order d as a basis function and adds noise."""
    data_x = []
    data_y = []
    
    data_x = np.linspace(-1, 1, n, endpoint=True) 
    # Set up raw output data based on a degree d polynomial
    trY_coeffs = [random.uniform(-1, 1) for i in range(d + 1)]
    data_y = 0 
    for i in range(d + 1): 
        data_y += trY_coeffs[i] * np.power(data_x, i)
        # Add some noise 
        data_y += np.random.randn(*data_x.shape) * noise_factor
    
    return [data_x, data_y]


def generate_sample_data_sin(n = 100, f = 1, noise_factor=0):
    """Generate sample data consisting of n points.

    Uses sin as a basis function and adds noise."""
    data_x = []
    data_y = []

    data_x = np.linspace(0, 2*np.pi, n, endpoint=True)
    data_y = np.sin(f*data_x)
    data_y += 2*np.random.randn(*data_x.shape) * noise_factor

    return [data_x, data_y]


def generate_sample_data_cam_table(n = 100, noise_factor=0):
    """Generate DRDRD sample data consisting of n points.

    Uses sine waveform for rise and return phase."""
    x = linspace(0,3.3,n)
    y = np.zeros(n)

    for i in range(n):
        if x[i] <= 0 and x[i] <= 0.6:
            y[i] = 0
        elif x[i] > 0.6 and x[i] <= 1.4:
            y[i] = math.sin(4*(x[i]-1))+1
        elif x[i] > 1.4 and x[i] <= 2:
            y[i] = 2
        elif x[i] > 2 and x[i] <= 2.79:
            y[i] = math.sin(4*(x[i]-1.61))+1
        else:
            y[i] = 0
    
    y += 2*np.random.randn(*x.shape) * noise_factor

    return [x, y]


def rescale_input_data(data_x, n_segments):
    """Scale 2D input data axes to [0, 1]."""
    min_max_scaler = preprocessing.MinMaxScaler((0, n_segments))

    if not isinstance(data_x,(np.ndarray)):
        data_x = np.array(data_x)

    x = data_x.reshape(-1, 1)
    data_x = min_max_scaler.fit_transform(x)
    data_x = data_x.flatten()

    return data_x