import sqlite3
import sys; sys.path.insert(0, '..')
from autograd import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from IPython.display import display

from src.models import LUNA
from src.utils import generate_data
import LUNA_database_parse as par

def run_experiments():
    data_base_old = sqlite3.connect('./trained_databases/LUNA_trained_results_series1.sqlite')
    cursor_old = data_base_old.cursor()
    model_train_cols = [col[1] for col in cursor_old.execute("PRAGMA table_info(train_data)")]
    query_old = '''SELECT * FROM train_data'''

    _, _, x_test = generate_data(number_of_points=50, noise_variance=9)
    x, y = par.retrieve_train_data_from_database(data_base_old, model_train_cols, query_old)

    data_base = sqlite3.connect('LUNA_trained_results_series4.sqlite')
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

    ## generate experiment 22 model, 'exp5Aa'
    architecture['width'] = [25, 50, 100]
    nn22 = LUNA(architecture, random=random)
    nn22_tag = 'exp5Aa'
    t = time.time()
    nn22.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn22_tag, data_base, nn22.weights)
    par.save_runtime_to_database(nn22_tag, data_base, t2, 2*x.shape[1], nn22.weights.shape[1])

    ## generate experiment 23 model, 'exp5Ab'
    architecture['width'] = [100, 100, 25]
    nn23 = LUNA(architecture, random=random)
    nn23_tag = 'exp5Ab'
    t = time.time()
    nn23.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn23_tag, data_base, nn23.weights)
    par.save_runtime_to_database(nn23_tag, data_base, t2, 2*x.shape[1], nn23.weights.shape[1])

    ## generate experiment 24 model, 'exp5Ac'
    architecture['width'] = [100, 100, 50]
    nn24 = LUNA(architecture, random=random)
    nn24_tag = 'exp5Ac'
    t = time.time()
    nn24.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn24_tag, data_base, nn24.weights)
    par.save_runtime_to_database(nn24_tag, data_base, t2, 2*x.shape[1], nn24.weights.shape[1])

    ## generate experiment 25 model, 'exp5Ba'
    architecture['width'] = [200, 100, 50]
    nn25 = LUNA(architecture, random=random)
    nn25_tag = 'exp5Ba'
    t = time.time()
    nn25.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn25_tag, data_base, nn25.weights)
    par.save_runtime_to_database(nn25_tag, data_base, t2, 2*x.shape[1], nn25.weights.shape[1])

    ## generate experiment 26 model, 'exp5Bb'
    architecture['width'] = [100, 50, 25]
    nn26 = LUNA(architecture, random=random)
    nn26_tag = 'exp5Bb'
    t = time.time()
    nn26.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn26_tag, data_base, nn26.weights)
    par.save_runtime_to_database(nn26_tag, data_base, t2, 2*x.shape[1], nn26.weights.shape[1])

    ## generate experiment 27 model, 'exp5Bc'
    architecture['width'] = [100, 50, 50]
    nn27 = LUNA(architecture, random=random)
    nn27_tag = 'exp5Bc'
    t = time.time()
    nn27.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn27_tag, data_base, nn27.weights)
    par.save_runtime_to_database(nn27_tag, data_base, t2, 2*x.shape[1], nn27.weights.shape[1])
