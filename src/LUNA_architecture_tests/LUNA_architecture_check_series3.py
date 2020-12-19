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
    data_base_old = sqlite3.connect('./trained_databases/LUNA_trained_results_series1.sqlite')
    cursor_old = data_base_old.cursor()
    model_train_cols = [col[1] for col in cursor_old.execute("PRAGMA table_info(train_data)")]
    query_old = '''SELECT * FROM train_data'''

    x, y = par.retrieve_train_data_from_database(data_base_old, model_train_cols, query_old)

    data_base = sqlite3.connect('LUNA_trained_results_series3.sqlite')
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

    ## generate experiment 14 model, 'exp3Aa'
    architecture['width'] = [50, 200]
    nn14 = LUNA(architecture, random=random)
    nn14_tag = 'exp3Aa'
    t = time.time()
    nn14.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn14_tag, data_base, nn14.weights)
    par.save_runtime_to_database(nn14_tag, data_base, t2, 2*x.shape[1], nn14.weights.shape[1])

    ## generate experiment 15 model, 'exp3Ba'
    architecture['width'] = [25, 200]
    nn15 = LUNA(architecture, random=random)
    nn15_tag = 'exp3Ba'
    t = time.time()
    nn15.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn15_tag, data_base, nn15.weights)
    par.save_runtime_to_database(nn15_tag, data_base, t2, 2*x.shape[1], nn15.weights.shape[1])

    ## generate experiment 16 model, 'exp3Ab'
    architecture['width'] = [50, 300]
    nn16 = LUNA(architecture, random=random)
    nn16_tag = 'exp3Ab'
    t = time.time()
    nn16.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn16_tag, data_base, nn16.weights)
    par.save_runtime_to_database(nn16_tag, data_base, t2, 2*x.shape[1], nn16.weights.shape[1])

    ## generate experiment 17 model, 'exp3Bb'
    architecture['width'] = [25, 300]
    nn17 = LUNA(architecture, random=random)
    nn17_tag = 'exp3Bb'
    t = time.time()
    nn17.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn17_tag, data_base, nn17.weights)
    par.save_runtime_to_database(nn17_tag, data_base, t2, 2*x.shape[1], nn17.weights.shape[1])

    ## generate experiment 18 model, 'exp4Aa'
    architecture['width'] = [50, 50, 50, 100]
    nn18 = LUNA(architecture, random=random)
    nn18_tag = 'exp4Aa'
    t = time.time()
    nn18.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn18_tag, data_base, nn18.weights)
    par.save_runtime_to_database(nn18_tag, data_base, t2, 2*x.shape[1], nn18.weights.shape[1])

    ## generate experiment 19 model, 'exp4Ab'
    architecture['width'] = [25, 25, 25, 100]
    nn19 = LUNA(architecture, random=random)
    nn19_tag = 'exp4Ab'
    t = time.time()
    nn19.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn19_tag, data_base, nn19.weights)
    par.save_runtime_to_database(nn19_tag, data_base, t2, 2*x.shape[1], nn19.weights.shape[1])

    ## generate experiment 20 model, 'exp4Ba'
    architecture['width'] = [50, 50, 50, 50, 50]
    nn20 = LUNA(architecture, random=random)
    nn20_tag = 'exp4Ba'
    t = time.time()
    nn20.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn20_tag, data_base, nn20.weights)
    par.save_runtime_to_database(nn20_tag, data_base, t2, 2*x.shape[1], nn20.weights.shape[1])

    ## generate experiment 21 model, 'exp4Bb'
    architecture['width'] = [25, 25, 25, 25, 25]
    nn21 = LUNA(architecture, random=random)
    nn21_tag = 'exp4Bb'
    t = time.time()
    nn21.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn21_tag, data_base, nn21.weights)
    par.save_runtime_to_database(nn21_tag, data_base, t2, 2*x.shape[1], nn21.weights.shape[1])
