from autograd import numpy as np

def generate_data(number_of_points=10, noise_variance=9):
    """Function for generating toy regression data"""
    #training x
    x_train = np.hstack((np.linspace(-4, -2, number_of_points), np.linspace(2, 4, number_of_points)))
    #function relating x and y
    f = lambda x: x**3
    #y is equal to f(x) plus gaussian noise
    y_train = f(x_train) + np.random.normal(0, noise_variance**0.5, 2 * number_of_points)
    # x_test = np.array(list(set(list(np.hstack((np.linspace(-5, , 100), x_train))))))
    x_test = np.array(list(set(list((np.linspace(-5, 5, 100))))))
    x_test = np.sort(x_test)
    return x_train, y_train, x_test
