import sqlite3
import sys; sys.path.insert(0, '..')
from autograd import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from IPython.display import display
from src.models import LUNA
from src.utils import generate_data

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
