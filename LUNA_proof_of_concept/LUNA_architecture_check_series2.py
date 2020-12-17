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

    print(x.shape)
    print(y.shape)

    # data_base = sqlite3.connect('LUNA_trained_results_series2.sqlite')
    # cursor = data_base.cursor()
    # cursor.execute("DROP TABLE IF EXISTS model_params")
    # cursor.execute("DROP TABLE IF EXISTS runtime")
    # cursor.execute("DROP TABLE IF EXISTS train_data")
    # cursor.execute("PRAGMA foreign_keys=1")
    # cursor.execute('''CREATE TABLE model_params (
    #             id TEXT,
    #             ind INTEGER,
    #             value REAL)
    #             ''')
    # cursor.execute('''CREATE TABLE runtime (
    #             id TEXT,
    #             runtime REAL,
    #             num_data INTEGER,
    #             num_params INTEGER)
    #             ''')
    # cursor.execute('''CREATE TABLE train_data (
    #             ind INT,
    #             x_val REAL,
    #             y_val REAL)
    #             ''')
    # data_base.commit()
    #
    # ## generate constant model parameters
    #
    # activation_fn_type = 'relu'
    # activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)
    #
    # width = [50,50] # using the architecture used in the paper
    # hidden_layers = len(width)
    # input_dim = 1
    # output_dim = 1
    #
    # architecture = {
    #     'width': width,
    #     'input_dim': input_dim,
    #     'output_dim': output_dim,
    #     'activation_fn_type': 'relu',
    #     'activation_fn_params': 'rate=1',
    #     'activation_fn': activation_fn,
    #     'auxiliary_functions': 30,
    # }
    #
    # rand_state = 207
    # random = np.random.RandomState(rand_state)
    #
    # params = {
    #     'step_size':1e-2,
    #     'max_iteration':5000,
    #     'random_restarts':1,
    #     'reg_param':100.,
    #     'lambda_in':100.,
    # }
    #
    # par.save_training_data_to_database(data_base, x, y)
    #
    # model_train_cols = [col[1] for col in cursor.execute("PRAGMA table_info(train_data)")]
    # query1 = '''SELECT * FROM train_data'''
    #
    # ########################################################################
    # ########################################################################
    # ########################################################################
    #
    # ## generate experiment 8 model, 'exp1Ba'
    # architecture['width'] = [50, 50, 50]
    # nn8 = LUNA(architecture, random=random)
    # nn8_tag = 'exp1Ba'
    # t = time.time()
    # nn8.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    # t2 = time.time() - t
    # par.save_params_to_database(nn8_tag, data_base, nn8.weights)
    # par.save_runtime_to_database(nn8_tag, data_base, t2, 2*x.shape[1], nn8.weights.shape[1])
    #
    # ## generate experiment 9 model, 'exp1Ca'
    # architecture['width'] = [50, 50, 50, 50]
    # nn9 = LUNA(architecture, random=random)
    # nn9_tag = 'exp1Ca'
    # t = time.time()
    # nn9.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    # t2 = time.time() - t
    # par.save_params_to_database(nn9_tag, data_base, nn9.weights)
    # par.save_runtime_to_database(nn9_tag, data_base, t2, 2*x.shape[1], nn9.weights.shape[1])
    #
    # ## generate experiment 10 model, 'exp2Aa'
    # architecture['width'] = [25, 50]
    # nn10 = LUNA(architecture, random=random)
    # nn10_tag = 'exp2Aa'
    # t = time.time()
    # nn10.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    # t2 = time.time() - t
    # par.save_params_to_database(nn10_tag, data_base, nn10.weights)
    # par.save_runtime_to_database(nn10_tag, data_base, t2, 2*x.shape[1], nn10.weights.shape[1])
    #
    # ## generate experiment 11 model, 'exp2Bb'
    # architecture['width'] = [50, 100]
    # nn11 = LUNA(architecture, random=random)
    # nn11_tag = 'exp2Bb'
    # t = time.time()
    # nn11.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    # t2 = time.time() - t
    # par.save_params_to_database(nn11_tag, data_base, nn11.weights)
    # par.save_runtime_to_database(nn11_tag, data_base, t2, 2*x.shape[1], nn11.weights.shape[1])
    #
    # ## generate experiment 12 model, 'exp2Ca'
    # architecture['width'] = [25, 100]
    # nn12 = LUNA(architecture, random=random)
    # nn12_tag = 'exp2Ca'
    # t = time.time()
    # nn12.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    # t2 = time.time() - t
    # par.save_params_to_database(nn12_tag, data_base, nn12.weights)
    # par.save_runtime_to_database(nn12_tag, data_base, t2, 2*x.shape[1], nn12.weights.shape[1])
    #
    # ## generate experiment 13 model, 'exp2Cb'
    # architecture['width'] = [100, 25]
    # nn13 = LUNA(architecture, random=random)
    # nn13_tag = 'exp2Cb'
    # t = time.time()
    # nn13.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)
    # t2 = time.time() - t
    # par.save_params_to_database(nn13_tag, data_base, nn13.weights)
    # par.save_runtime_to_database(nn13_tag, data_base, t2, 2*x.shape[1], nn13.weights.shape[1])
