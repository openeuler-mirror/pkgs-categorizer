import numpy as np
import pandas as pd
from sklearn.utils import shuffle
import re

def get_idx_name_map(names):
    """
    输入包名列表，输出 name_idx, idx_name Map
    """
    name_list = list(names)
    name_idx_map = {}
    idx_name_map = {}
    for i, name in enumerate(name_list):
        name_idx_map[name] = i
        idx_name_map[i] = name

