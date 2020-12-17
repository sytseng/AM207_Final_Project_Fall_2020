import sqlite3
import sys; sys.path.insert(0, '..')
from autograd import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
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

def save_training_data_to_database(db, x_train, y_train):
    cursor = db.cursor()

    for i, val in enumerate(x_train[0]):
        cursor.execute('''INSERT INTO train_data
                    (ind, x_val, y_val)
                    VALUES (?, ?, ?)''',
                    (i, val.item(), y_train[0][i].item() ) )

    db.commit()
    print("saved training data")

def save_runtime_to_database(model_id, db, train_time, num_data, num_params):
    cursor = db.cursor()
    cursor.execute('''INSERT INTO runtime
                (id, runtime, num_data, num_params)
                VALUES (?, ?, ?, ?)''',
                (model_id, train_time, num_data, num_params))

    db.commit()
    print("saved training runtime for {}".format(model_id))

def retrieve_params_from_database(db, cols, query, model_id):
    cursor = db.cursor()

    q = cursor.execute(query).fetchall()
    framelist = dict()
    for i, col_name in enumerate(cols):
        framelist[col_name] = [row[i] for row in q]
    values = []
    for i, id in enumerate(framelist['id']):
        if id == model_id:
            values.append(framelist['value'][i])

    return np.array([values])

def retrieve_train_data_from_database(db, cols, query):
    cursor = db.cursor()

    q = cursor.execute(query).fetchall()
    framelist = dict()
    for i, col_name in enumerate(cols):
        framelist[col_name] = [row[i] for row in q]
    x_values = []
    y_values = []
    for i, xval in enumerate(framelist['x_val']):
        x_values.append(xval)
        y_values.append(framelist['y_val'][i])

    return np.array([x_values]), np.array([y_values])

def retrieve_runtimes_from_database(db, cols, query, model_id):
    cursor = db.cursor()

    q = cursor.execute(query).fetchall()
    framelist = dict()
    for i, col_name in enumerate(cols):
        framelist[col_name] = [row[i] for row in q]
    runtimes = []
    data_num = []
    param_num = []
    for i, id in enumerate(framelist['id']):
        if id == model_id:
            runtimes.append(framelist['runtime'])
            data_num.append(framelist['num_data'])
            param_num.append(framelist['num_params'])

    return runtimes, data_num, param_num
