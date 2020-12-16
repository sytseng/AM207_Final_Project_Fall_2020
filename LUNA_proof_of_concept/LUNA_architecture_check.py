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

data_base = sqlite3.connect('LUNA_trained_results_linux')
cursor = data_base.cursor()
cursor.execute("DROP TABLE IF EXISTS model_params")
data_base.commit()

cursor.execute("PRAGMA foreign_keys=1")
cursor.execute('''CREATE TABLE model_params (
            id TEXT,
            ind INTEGER,
            value REAL)
            ''')

data_base.commit()


## generate baseline model, 'exp0Aa'

activation_fn_type = 'relu'
activation_fn = lambda x: np.maximum(np.zeros(x.shape), x)

width = [50,50] # using the architecture used in the paper
hidden_layers = len(width)
input_dim = 1
output_dim = 1

architecture0 = {
    'width': width,
    'input_dim': input_dim,
    'output_dim': output_dim,
    'activation_fn_type': 'relu',
    'activation_fn_params': 'rate=1',
    'activation_fn': activation_fn,
    'auxiliary_functions': 30,
}

#set random state to make the experiments replicable
rand_state = 207
random = np.random.RandomState(rand_state)

#instantiate a Feedforward neural network object
nn0Aa = LUNA(architecture0, random=random)
nn0Aa_tag = 'exp0Aa'

### define design choices in gradient descent
params = {
    'step_size':1e-2,
    'max_iteration':5000,
    'random_restarts':1,
    'reg_param':100.,
    'lambda_in':100.,
}

# fit LUNA
nn0Aa.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)

save_params_to_database(nn0Aa_tag, data_base, nn0Aa.weights)
# model_params_cols = [col[1] for col in cursor.execute("PRAGMA table_info(model_params)")]
# query1 = '''SELECT * FROM model_params'''

## generate experiment 4 model, 'exp1Bb'

width = [25,25,25] # using the architecture used in the paper
hidden_layers = len(width)
input_dim = 1
output_dim = 1

architecture4 = {
    'width': [25,25,25],
    'input_dim': input_dim,
    'output_dim': output_dim,
    'activation_fn_type': 'relu',
    'activation_fn_params': 'rate=1',
    'activation_fn': activation_fn,
    'auxiliary_functions': 30,
}

#set random state to make the experiments replicable
rand_state = 207
random = np.random.RandomState(rand_state)

#instantiate a Feedforward neural network object
nn1Bb = LUNA(architecture4, random=random)
nn1Bb_tag = 'exp1Bb'

# fit LUNA
nn1Bb.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)

save_params_to_database(nn1Bb_tag, data_base, nn1Bb.weights)

## generate experiment 5 model, 'exp1Cb'

width = [25,25,25,25] # using the architecture used in the paper
hidden_layers = len(width)
input_dim = 1
output_dim = 1

architecture5 = {
    'width': [25,25,25,25],
    'input_dim': input_dim,
    'output_dim': output_dim,
    'activation_fn_type': 'relu',
    'activation_fn_params': 'rate=1',
    'activation_fn': activation_fn,
    'auxiliary_functions': 30,
}

#set random state to make the experiments replicable
rand_state = 207
random = np.random.RandomState(rand_state)

#instantiate a Feedforward neural network object
nn1Cb = LUNA(architecture5, random=random)
nn1Cb_tag = 'exp1Cb'

# fit LUNA
nn1Cb.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)

save_params_to_database(nn1Cb_tag, data_base, nn1Cb.weights)

## generate experiment 6 model, 'exp2Ab'

width = [100,50] # using the architecture used in the paper
hidden_layers = len(width)
input_dim = 1
output_dim = 1

architecture6 = {
    'width': [100,50],
    'input_dim': input_dim,
    'output_dim': output_dim,
    'activation_fn_type': 'relu',
    'activation_fn_params': 'rate=1',
    'activation_fn': activation_fn,
    'auxiliary_functions': 30,
}

#set random state to make the experiments replicable
rand_state = 207
random = np.random.RandomState(rand_state)

#instantiate a Feedforward neural network object
nn2Ab = LUNA(architecture6, random=random)
nn2Ab_tag = 'exp2Ab'

# fit LUNA
nn2Ab.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)

save_params_to_database(nn2Ab_tag, data_base, nn2Ab.weights)

## generate experiment 7 model, 'exp2Ba'

width = [50,25] # using the architecture used in the paper
hidden_layers = len(width)
input_dim = 1
output_dim = 1

architecture7 = {
    'width': [50,25],
    'input_dim': input_dim,
    'output_dim': output_dim,
    'activation_fn_type': 'relu',
    'activation_fn_params': 'rate=1',
    'activation_fn': activation_fn,
    'auxiliary_functions': 30,
}

#set random state to make the experiments replicable
rand_state = 207
random = np.random.RandomState(rand_state)

#instantiate a Feedforward neural network object
nn2Ba = LUNA(architecture7, random=random)
nn2Ba_tag = 'exp2Ba'

# fit LUNA
nn2Ba.fit(x.reshape((1, -1)), y.reshape((1, -1)), params)

save_params_to_database(nn2Ba_tag, data_base, nn2Ba.weights)

data_base.close()
