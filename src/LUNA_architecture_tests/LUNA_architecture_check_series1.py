import sqlite3
import sys; sys.path.insert(0, '../..')
from autograd import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from IPython.display import display

from src.models import LUNA
from src.utils import generate_data
import LUNA_database_parse as par

def run_experiments():
    x, y, _ = generate_data(number_of_points=50, noise_variance=9)

    data_base = sqlite3.connect('LUNA_trained_results_series1.sqlite')
    cursor = data_base.cursor()
    cursor.execute("DROP TABLE IF EXISTS model_params")
    cursor.execute("DROP TABLE IF EXISTS runtime")
    cursor.execute("DROP TABLE IF EXISTS train_data")
    cursor.execute("PRAGMA foreign_keys=1")
    cursor.execute('''CREATE TABLE model_params (
                id TEXT,
                ind INTEGER,
                value REAL)
                ''')
    cursor.execute('''CREATE TABLE runtime (
                id TEXT,
                runtime REAL,
                num_data INTEGER,
                num_params INTEGER)
                ''')
    cursor.execute('''CREATE TABLE train_data (
                ind INT,
                x_val REAL,
                y_val REAL)
                ''')
    data_base.commit()

    ## generate constant model parameters

    activation_fn_type = 'relu'
    activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)

    width = [50,50] # using the architecture used in the paper
    hidden_layers = len(width)
    input_dim = 1
    output_dim = 1

    architecture = {
        'width': width,
        'input_dim': input_dim,
        'output_dim': output_dim,
        'activation_fn_type': 'relu',
        'activation_fn_params': 'rate=1',
        'activation_fn': activation_fn,
        'auxiliary_functions': 30,
    }

    rand_state = 207
    random = np.random.RandomState(rand_state)

    params = {
        'step_size':1e-2,
        'max_iteration':5000,
        'random_restarts':1,
        'reg_param':100.,
        'lambda_in':100.,
    }

    par.save_training_data_to_database(data_base, x, y)

    ########################################################################
    ########################################################################
    ########################################################################

    ## generate baseline model, 'exp0Aa'
    nn0 = LUNA(architecture, random=random)
    nn0_tag = 'exp0Aa'
    t = time.time()
    nn0.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn0_tag, data_base, nn0.weights)
    par.save_runtime_to_database(nn0_tag, data_base, t2, 2*x.shape[1], nn0.weights.shape[1])

    ## generate experiment 1 model, 'exp1Aa'
    architecture['width'] = [50]
    nn1 = LUNA(architecture, random=random)
    nn1_tag = 'exp1Aa'
    t = time.time()
    nn1.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn1_tag, data_base, nn1.weights)
    par.save_runtime_to_database(nn1_tag, data_base, t2, 2*x.shape[1], nn1.weights.shape[1])

    ## generate experiment 2 model, 'exp1Ab'
    architecture['width'] = [25]
    nn2 = LUNA(architecture, random=random)
    nn2_tag = 'exp1Ab'
    t = time.time()
    nn2.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn2_tag, data_base, nn2.weights)
    par.save_runtime_to_database(nn2_tag, data_base, t2, 2*x.shape[1], nn2.weights.shape[1])

    ## generate experiment 3 model, 'exp1Ac'
    architecture['width'] = [100]
    nn3 = LUNA(architecture, random=random)
    nn3_tag = 'exp1Ac'
    t = time.time()
    nn3.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn3_tag, data_base, nn3.weights)
    par.save_runtime_to_database(nn3_tag, data_base, t2, 2*x.shape[1], nn3.weights.shape[1])

    ## generate experiment 4 model, 'exp1Bb'
    architecture['width'] = [25, 25, 25]
    nn4 = LUNA(architecture, random=random)
    nn4_tag = 'exp1Bb'
    t = time.time()
    nn4.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn4_tag, data_base, nn4.weights)
    par.save_runtime_to_database(nn4_tag, data_base, t2, 2*x.shape[1], nn4.weights.shape[1])

    ## generate experiment 5 model, 'exp1Cb'
    architecture['width'] = [25, 25, 25, 25]
    nn5 = LUNA(architecture, random=random)
    nn5_tag = 'exp1Cb'
    t = time.time()
    nn5.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn5_tag, data_base, nn5.weights)
    par.save_runtime_to_database(nn5_tag, data_base, t2, 2*x.shape[1], nn5.weights.shape[1])

    ## generate experiment 6 model, 'exp2Ab'
    architecture['width'] = [100, 50]
    nn6 = LUNA(architecture, random=random)
    nn6_tag = 'exp2Ab'
    t = time.time()
    nn6.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn6_tag, data_base, nn6.weights)
    par.save_runtime_to_database(nn6_tag, data_base, t2, 2*x.shape[1], nn6.weights.shape[1])

    ## generate experiment 7 model, 'exp2Ba'
    architecture['width'] = [50, 25]
    nn7 = LUNA(architecture, random=random)
    nn7_tag = 'exp2Ba'
    t = time.time()
    nn7.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn7_tag, data_base, nn7.weights)
    par.save_runtime_to_database(nn7_tag, data_base, t2, 2*x.shape[1], nn7.weights.shape[1])
