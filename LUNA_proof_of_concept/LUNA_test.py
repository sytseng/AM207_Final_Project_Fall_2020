import sys
from scipy.special import logsumexp
from autograd import numpy as np
from autograd import scipy as sp
from autograd import grad
from autograd.misc.optimizers import adam, sgd
from autograd.scipy.stats import multivariate_normal
import numpy
import math
import pdb
import matplotlib.pyplot as plt

from LUNA_regression import LUNA
from NLM_regression import NLM

def main():
    x, y, x_test = generate_data(number_of_points=50, noise_variance=9)
    plt.scatter(x, y)
    # plt.show()

    ###relu activation
    activation_fn_type = 'relu'
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)

    width = [50,20] # using the architecture used in the paper
    hidden_layers = len(width)
    input_dim = 1
    output_dim = 1

    M = 30

    architecture_NLM = {'width': width,
                   'hidden_layers': hidden_layers,
                   'input_dim': input_dim,
                   'output_dim': output_dim,
                   'activation_fn_type': 'relu',
                   'activation_fn_params': 'rate=1',
                   'activation_fn': activation_fn}

    architecture_LUNA = {'width': width,
                   'hidden_layers': hidden_layers,
                   'auxiliary_functions': M,
                   'input_dim': input_dim,
                   'output_dim': output_dim,
                   'activation_fn_type': 'relu',
                   'activation_fn_params': 'rate=1',
                   'activation_fn': activation_fn}

    #set random state to make the experiments replicable
    rand_state = 0
    random = np.random.RandomState(rand_state)

    ###define design choices in gradient descent
    params = {'step_size':1e-3,
              'max_iteration':5000,
              'random_restarts':1}

    #fit my neural network to minimize MSE on the given data
    reg_param = 0.0
    lambda_in = 1.0

    #instantiate a Feedforward neural network object
    nn_NLM = NLM(architecture_NLM, random=random)
    print('NLM Number of parameters =',nn_NLM.D)
    nn_LUNA = LUNA(architecture_LUNA, random=random)
    print('LUNA Number of parameters =',nn_LUNA.D)

    theta_check = nn_LUNA.weights_full[0][:nn_LUNA.D_theta]
    w_check = nn_LUNA.weights_full[0][nn_LUNA.D_theta:(nn_LUNA.D_theta + nn_LUNA.D_w)]
    theta_check2 = np.array([theta_check])
    w_all_check = np.array([np.append(theta_check, w_check) ] )
    print(np.shape(theta_check2) )
    print(np.shape(w_all_check) )


    # nn_NLM.fit(x.reshape((1, -1)), y.reshape((1, -1)), params, reg_param = reg_param)
    nn_LUNA.fit(x.reshape((1, -1)), y.reshape((1, -1)), params, lambda_in, reg_param = reg_param)

def generate_data(number_of_points=10, noise_variance=9):
    '''Function for generating toy regression data'''
    #training x
    x_train = np.hstack((np.linspace(-4, -2, number_of_points), np.linspace(2, 4, number_of_points)))
    #function relating x and y
    f = lambda x: x**3
    #y is equal to f(x) plus gaussian noise
    y_train = f(x_train) + np.random.normal(0, noise_variance**0.5, 2 * number_of_points)
    x_test = np.array(list(set(list((np.linspace(-5, 5, 100))))))
    x_test = np.sort(x_test)
    return x_train, y_train, x_test

if __name__ == "__main__":
    main()
