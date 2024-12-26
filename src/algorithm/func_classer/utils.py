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

def get_num_label_map(labels):
    """
    输入标签列表，输出 name_idx, idx_name Map
    """

    print(labels)
    print('labels.........')
    label_list = []
    # 为了保持每次的顺序（有无更好的方法）
    for label in list(labels):
        if not label in label_list:
            label_list.append(label)
    # label_list = list(set(list(labels)))
    label_num_map = {}
    num_label_map = {}
    for i, label in enumerate(label_list):
        label_num_map[label] = i
        num_label_map[i] = label

    return num_label_map, label_num_map

