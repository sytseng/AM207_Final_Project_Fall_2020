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

    data_base = sqlite3.connect('LUNA_trained_results_series5.sqlite')
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

    ## generate experiment 28 model, 'exp6Aa'
    architecture['width'] = [25, 25]
    nn28 = LUNA(architecture, random=random)
    nn28_tag = 'exp6Aa'
    t = time.time()
    nn28.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn28_tag, data_base, nn28.weights)
    par.save_runtime_to_database(nn28_tag, data_base, t2, 2*x.shape[1], nn28.weights.shape[1])

    ## generate experiment 28 model, 'exp6Ba'
    architecture['width'] = [30, 10]
    nn29 = LUNA(architecture, random=random)
    nn29_tag = 'exp6Ba'
    t = time.time()
    nn29.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn29_tag, data_base, nn29.weights)
    par.save_runtime_to_database(nn29_tag, data_base, t2, 2*x.shape[1], nn29.weights.shape[1])

    ## generate experiment 28 model, 'exp6Bb'
    architecture['width'] = [30, 20]
    nn30 = LUNA(architecture, random=random)
    nn30_tag = 'exp6Bb'
    t = time.time()
    nn30.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn30_tag, data_base, nn30.weights)
    par.save_runtime_to_database(nn30_tag, data_base, t2, 2*x.shape[1], nn30.weights.shape[1])

    ## generate experiment 28 model, 'exp6Bc'
    architecture['width'] = [30, 30]
    nn31 = LUNA(architecture, random=random)
    nn31_tag = 'exp6Bc'
    t = time.time()
    nn31.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn31_tag, data_base, nn31.weights)
    par.save_runtime_to_database(nn31_tag, data_base, t2, 2*x.shape[1], nn31.weights.shape[1])

    ## generate experiment 28 model, 'exp6Bd'
    architecture['width'] = [30, 40]
    nn32 = LUNA(architecture, random=random)
    nn32_tag = 'exp6Bd'
    t = time.time()
    nn32.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn32_tag, data_base, nn32.weights)
    par.save_runtime_to_database(nn32_tag, data_base, t2, 2*x.shape[1], nn32.weights.shape[1])

    ## generate experiment 28 model, 'exp6Be'
    architecture['width'] = [30, 50]
    nn33 = LUNA(architecture, random=random)
    nn33_tag = 'exp6Be'
    t = time.time()
    nn33.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn33_tag, data_base, nn33.weights)
    par.save_runtime_to_database(nn33_tag, data_base, t2, 2*x.shape[1], nn33.weights.shape[1])

    ## generate experiment 28 model, 'exp6Bf'
    architecture['width'] = [30, 100]
    nn34 = LUNA(architecture, random=random)
    nn34_tag = 'exp6Bf'
    t = time.time()
    nn34.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn34_tag, data_base, nn34.weights)
    par.save_runtime_to_database(nn34_tag, data_base, t2, 2*x.shape[1], nn34.weights.shape[1])

    ## generate experiment 28 model, 'exp6Ca'
    architecture['width'] = [50, 10]
    nn35 = LUNA(architecture, random=random)
    nn35_tag = 'exp6Ca'
    t = time.time()
    nn35.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn35_tag, data_base, nn35.weights)
    par.save_runtime_to_database(nn35_tag, data_base, t2, 2*x.shape[1], nn35.weights.shape[1])

    ## generate experiment 28 model, 'exp6Cb'
    architecture['width'] = [50, 20]
    nn36 = LUNA(architecture, random=random)
    nn36_tag = 'exp6Cb'
    t = time.time()
    nn36.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn36_tag, data_base, nn36.weights)
    par.save_runtime_to_database(nn36_tag, data_base, t2, 2*x.shape[1], nn36.weights.shape[1])

    ## generate experiment 28 model, 'exp6Cc'
    architecture['width'] = [50, 30]
    nn37 = LUNA(architecture, random=random)
    nn37_tag = 'exp6Cc'
    t = time.time()
    nn37.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn37_tag, data_base, nn37.weights)
    par.save_runtime_to_database(nn37_tag, data_base, t2, 2*x.shape[1], nn37.weights.shape[1])

    ## generate experiment 28 model, 'exp6Da'
    architecture['width'] = [50, 40]
    nn38 = LUNA(architecture, random=random)
    nn38_tag = 'exp6Da'
    t = time.time()
    nn38.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn38_tag, data_base, nn38.weights)
    par.save_runtime_to_database(nn38_tag, data_base, t2, 2*x.shape[1], nn38.weights.shape[1])

    ## generate experiment 28 model, 'exp6Da'
    architecture['width'] = [20, 10]
    nn39 = LUNA(architecture, random=random)
    nn39_tag = 'exp6Da'
    t = time.time()
    nn39.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn39_tag, data_base, nn39.weights)
    par.save_runtime_to_database(nn39_tag, data_base, t2, 2*x.shape[1], nn39.weights.shape[1])

    ## generate experiment 28 model, 'exp6Db'
    architecture['width'] = [20, 20]
    nn40 = LUNA(architecture, random=random)
    nn40_tag = 'exp6Db'
    t = time.time()
    nn40.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn40_tag, data_base, nn40.weights)
    par.save_runtime_to_database(nn40_tag, data_base, t2, 2*x.shape[1], nn40.weights.shape[1])

    ## generate experiment 28 model, 'exp6Dc'
    architecture['width'] = [20, 30]
    nn41 = LUNA(architecture, random=random)
    nn41_tag = 'exp6Dc'
    t = time.time()
    nn41.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn41_tag, data_base, nn41.weights)
    par.save_runtime_to_database(nn41_tag, data_base, t2, 2*x.shape[1], nn41.weights.shape[1])

    ## generate experiment 28 model, 'exp6Dd'
    architecture['width'] = [20, 40]
    nn42 = LUNA(architecture, random=random)
    nn42_tag = 'exp6Dd'
    t = time.time()
    nn42.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn42_tag, data_base, nn42.weights)
    par.save_runtime_to_database(nn42_tag, data_base, t2, 2*x.shape[1], nn42.weights.shape[1])

    ## generate experiment 28 model, 'exp6De'
    architecture['width'] = [20, 50]
    nn43 = LUNA(architecture, random=random)
    nn43_tag = 'exp6De'
    t = time.time()
    nn43.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn43_tag, data_base, nn43.weights)
    par.save_runtime_to_database(nn43_tag, data_base, t2, 2*x.shape[1], nn43.weights.shape[1])

    ## generate experiment 28 model, 'exp6Df'
    architecture['width'] = [20, 100]
    nn44 = LUNA(architecture, random=random)
    nn44_tag = 'exp6Df'
    t = time.time()
    nn44.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    t2 = time.time() - t
    par.save_params_to_database(nn44_tag, data_base, nn44.weights)
    par.save_runtime_to_database(nn44_tag, data_base, t2, 2*x.shape[1], nn44.weights.shape[1])
