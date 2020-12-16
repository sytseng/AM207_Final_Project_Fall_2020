import datetime
from autograd import numpy as np

def generate_data(number_of_points=10, noise_variance=9., input_dimension=1, gap_start = -2., gap_end = 2., data_start = -4., data_end = 4., scale = 1.):
    """Generate toy regression data with function
            y = (x_1)^3 + (x_2)^3 + ... + (x_D)^3
    for a given input dimension D

    Returns:
        x_train (numpy.array): Toy training X of shape (n_param, n_obs)
        y_train (numpy.array): Toy training y of shape (n_param, n_obs)
        x_test (numpy.array): Toy test X of shape (n_param, n_obs)
    """

    # Set "parameters" of toy data
    f = lambda x: (x**3) * scale

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
    y_test = f(x_test)

    # Generate toy y
    _y_train = np.sum(f(x_train), axis=0, keepdims=True)
    e_train = np.random.normal(0, noise_variance**0.5, size=_y_train.shape)
    y_train = _y_train + e_train

    return x_train, y_train, x_test, y_test


def format_time(elapsed):
    """Helper function for formatting elapsed times

    Args:
        elapsed (float): Elapsed time in seconds

    Returns:
        (str): Elapsed time in "hh:mm:ss" format
    """
    elapsed_rounded = int(round(elapsed))
    return str(datetime.timedelta(seconds=elapsed_rounded))


def neg_log_likelihood(y_pred, y_true, y_noise_var, per_sample = True):
    '''y_pred: num of models/posterior samples x num of datapoints
       y_true: 1 x num of datapoints
       y_noise_var: variance of observation noise
       per_sample: if True, compute neg_log_lik per datapoint; if False, compute neg_log_lik over all datapsoints'''

    N_models = y_pred.shape[0]
    N_datapoints = y_pred.shape[1]
    assert N_datapoints == y_true.shape[1]

    noise_sd = y_noise_var**0.5

    if per_sample:
        constant = - 0.5 * np.log(2 * np.pi) - np.log(noise_sd)
        exponential = -0.5 * noise_sd**-1 * np.mean((y_pred - y_true)**2)

    else:
        constant = N_datapoints * (- 0.5 * np.log(2 * np.pi) - np.log(noise_sd))
        exponential = -0.5 * noise_sd**-1 * np.sum((y_pred - y_true)**2)/N_models

    return -constant - exponential


def epistemic_uncertainty(y_pred, take_avg = True):
    '''y_pred: num of models/posterior samples x num of datapoints
       take_avg: if True, return avg epistemic uncertainty; if False, return epistemic uncertainty for each datapoint'''

    if take_avg:
        return np.std(y_pred, axis=0).mean()
    else:
        return np.std(y_pred, axis=0)
