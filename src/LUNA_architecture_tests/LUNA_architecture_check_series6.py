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

    x, y = par.retrieve_train_data_from_database(data_base_old, model_train_cols, query_old)

    data_base = sqlite3.connect('LUNA_trained_results_series6.sqlite')
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

    ## generate experiment 45 model, 'exp7Aa'
    architecture['width'] = [100, 100]
    nn45 = LUNA(architecture, random=random)
    nn45_tag = 'exp7Aa'
    t = time.time()
    nn45.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn45_tag, data_base, nn45.weights)
    par.save_runtime_to_database(nn45_tag, data_base, t2, 2*x.shape[1], nn45.weights.shape[1])

    ## generate experiment 46 model, 'exp7Ab'
    architecture['width'] = [100, 100, 100]
    nn46 = LUNA(architecture, random=random)
    nn46_tag = 'exp7Ab'
    t = time.time()
    nn46.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn46_tag, data_base, nn46.weights)
    par.save_runtime_to_database(nn46_tag, data_base, t2, 2*x.shape[1], nn46.weights.shape[1])

    ## generate experiment 47 model, 'exp7Ac'
    architecture['width'] = [100, 100, 100, 100]
    nn47 = LUNA(architecture, random=random)
    nn47_tag = 'exp7Ac'
    t = time.time()
    nn47.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn47_tag, data_base, nn47.weights)
    par.save_runtime_to_database(nn47_tag, data_base, t2, 2*x.shape[1], nn47.weights.shape[1])

    ## generate experiment 48 model, 'exp7Ad'
    architecture['width'] = [100, 100, 100, 100, 100]
    nn48 = LUNA(architecture, random=random)
    nn48_tag = 'exp7Ad'
    t = time.time()
    nn48.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn48_tag, data_base, nn48.weights)
    par.save_runtime_to_database(nn48_tag, data_base, t2, 2*x.shape[1], nn48.weights.shape[1])
