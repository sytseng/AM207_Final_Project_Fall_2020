from autograd import numpy as np

def generate_data(number_of_points=10, noise_variance=9, input_dimension=1):
    """Generate toy regression data with function
            y = (x_1)^3 + (x_2)^3 + ... + (x_D)^3
    for a given input dimension D

    Returns:
        x_train (numpy.array): Toy training X of shape (n_param, n_obs)
        y_train (numpy.array): Toy training y of shape (n_param, n_obs)
        x_test (numpy.array): Toy test X of shape (n_param, n_obs)
    """

    # Set "parameters" of toy data
    f = lambda x: x**3
    data_start, gap_start, gap_end, data_end = (-4, -2, 2, 4)

    # Construct toy X
    _x_train = np.hstack([
        np.linspace(data_start, gap_start, number_of_points),
        np.linspace(gap_end, data_end, number_of_points)
    ])
    _x_test = np.sort(np.unique(np.concatenate([
        _x_train,
        np.linspace(data_start-1, data_end+1, number_of_points*10)
    ])))
    x_train = np.vstack([_x_train.reshape(1, -1)] * input_dimension)
    x_test = np.vstack([_x_test.reshape(1, -1)] * input_dimension)

    # Generate toy y
    _y_train = np.sum(f(x_train), axis=0, keepdims=True)
    e_train = np.random.normal(0, noise_variance**0.5, size=_y_train.shape)
    y_train = _y_train + e_train

    return x_train, y_train, x_test
