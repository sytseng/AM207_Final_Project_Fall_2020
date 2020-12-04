from autograd import numpy as np

def generate_data(number_of_points=10, noise_variance=9, dim_in=1):
    """Generate toy regression data with function
            y = (x_1)^3 + (x_2)^3 + ... + (x_D)^3
    for a given input dimension D
    """
    # Set "parameters" of toy data
    f = lambda x: x**3
    data_start, gap_start, gap_end, data_end = (-4, -2, 2, 4)

    # Construct toy data
    _x_train = np.hstack([
        np.linspace(data_start, gap_start, number_of_points),
        np.linspace(gap_end, data_end, number_of_points)
    ])
    _x_test = np.sort(np.unique(np.concatenate([
        _x_train,
        np.linspace(data_start-1, data_end+1, number_of_points*10)
    ])))
    x_train = np.hstack([_x_train.reshape(-1, 1)] * dim_in)
    x_test = np.hstack([_x_test.reshape(-1, 1)] * dim_in)
    y_train = np.sum(f(x_train), axis=1) + np.random.normal(0, noise_variance**0.5, number_of_points*2)

    return x_train.squeeze(), y_train, x_test.squeeze() # Drop single dimension from array shape
