import datetime
from autograd import numpy as np

def generate_data(data_region=[(-4,-2),(2,4)], number_of_points=[10,10], noise_variance=9., input_dimension=1, scale=1.):
    """Generate toy regression data with function
            y = K * (x_1)^3 + K * (x_2)^3 + ... + K * (x_D)^3
    for a given scale K and input dimension D.

    Args:
        data_region (list[tuple]): List of tuples specifying data regions in input space.
        number_of_points (list(int)): List of data points to put in each data region.

    Returns:
        x_train (numpy.array): Toy training X of shape (n_param, n_obs).
        y_train (numpy.array): Toy training y of shape (n_param, n_obs); noise added.
        x_test (numpy.array): Toy test X of shape (n_param, n_obs).
        y_test (numpy.array): Toy test y of shape (n_param, n_obs); no noise.
    """

    # Ensure correct inputs
    if not hasattr(number_of_points, '__iter__'): # If a single number
        number_of_points = [int(number_of_points)] * len(list(data_region))
    if len(list(data_region)) != len(list(number_of_points)):
        raise Exception("Input Error: `data_region` and `number_of_points` should be lists of the same length.")

    # Set "parameters" of toy data
    f = lambda x: (x**3) * scale

    # Construct toy X
    _x_train = np.concatenate([np.linspace(xs[0], xs[1], n) for xs, n in zip(data_region, number_of_points)])
    _x_test = np.sort(np.unique(np.concatenate([
        _x_train,
        np.linspace(min(data_region)[0]-1, max(data_region)[1]+1, np.sum(number_of_points)*10)
    ])))
    x_train = np.vstack([_x_train.reshape(1, -1)] * input_dimension)
    x_test = np.vstack([_x_test.reshape(1, -1)] * input_dimension)

    # Generate toy y
    _y_train = np.sum(f(x_train), axis=0, keepdims=True)
    _e_train = np.random.normal(0, noise_variance**0.5, size=_y_train.shape)
    y_train = _y_train + _e_train
    y_test = np.sum(f(x_test), axis=0, keepdims=True) # Add no noise

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
