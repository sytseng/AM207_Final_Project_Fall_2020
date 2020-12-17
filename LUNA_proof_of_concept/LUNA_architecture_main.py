import sqlite3
import sys; sys.path.insert(0, '..')
from autograd import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import time
from IPython.display import display
from src.models import LUNA
from src.utils import generate_data

import LUNA_architecture_check_series1 as arch1

arch1.run_experiments()
