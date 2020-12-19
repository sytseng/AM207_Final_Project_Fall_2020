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
import LUNA_architecture_check_series2 as arch2
import LUNA_architecture_check_series3 as arch3
import LUNA_architecture_check_series4 as arch4
import LUNA_architecture_check_series5 as arch5
import LUNA_architecture_check_series6 as arch6
import LUNA_architecture_check_series7 as arch7


# arch1.run_experiments()
# arch2.run_experiments()
# arch3.run_experiments()
# arch4.run_experiments()
# arch5.run_experiments()
# arch6.run_experiments()
arch7.run_experiments()
