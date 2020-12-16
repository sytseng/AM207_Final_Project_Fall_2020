import sqlite3
import sys; sys.path.insert(0, '..')
from autograd import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from IPython.display import display

from src.models import LUNA
from src.utils import generate_data

def save_params_to_database(model_id, db, params):
    cursor = db.cursor()

    for i, val in enumerate(params[0]):
        cursor.execute('''INSERT INTO model_params
                    (id, ind, value)
                    VALUES (?, ?, ?)''',
                    (model_id, i, val.item()))

    db.commit()
    print("saved trained parameters for {}".format(model_id))

def retrieve_params_from_database(cols, query, model_id):
    q = cursor.execute(query).fetchall()
    framelist = dict()
    for i, col_name in enumerate(cols):
        framelist[col_name] = [row[i] for row in q]
    print(framelist)
    values = []
    for i, id in enumerate(framelist['id']):
        if id == model_id:
            values.append(framelist['value'][i])

    return np.array([values])

###### begin experiments

x, y, x_test = generate_data(number_of_points=50, noise_variance=9)
# plt.scatter(x, y)
# plt.show()

data_base = sqlite3.connect('LUNA_trained_results')
cursor = data_base.cursor()
# cursor.execute("DROP TABLE IF EXISTS model_params")
# data_base.commit()

cursor.execute("PRAGMA foreign_keys=1")
# cursor.execute('''CREATE TABLE model_params (
#             id TEXT,
#             ind INTEGER,
#             value REAL)
#             ''')

data_base.commit()


## generate baseline model, 'exp0Aa'

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
    'auxiliary_functions': 50,
}

#set random state to make the experiments replicable
rand_state = 207
random = np.random.RandomState(rand_state)

#instantiate a Feedforward neural network object
nn0Aa = LUNA(architecture, random=random)
nn0Aa_tag = 'exp0Aa'

### define design choices in gradient descent
params = {
    'step_size':1e-2,
    'max_iteration':5000,
    'random_restarts':1,
    'reg_param':1.,
    'lambda_in':1.,
}

# fit LUNA
nn0Aa.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)

save_params_to_database(nn0Aa_tag, data_base, nn0Aa.weights)
model_params_cols = [col[1] for col in cursor.execute("PRAGMA table_info(model_params)")]
query1 = '''SELECT * FROM model_params'''
