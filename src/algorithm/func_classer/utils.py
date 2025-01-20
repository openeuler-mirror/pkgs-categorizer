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


def build_edge_text_v1(name, df_data, df_edges):
    label_map = {
        "核心工具":"core tool",
        "核心服务":"core service",
        "核心库":"core library",
        "系统库":"system library",
        "系统服务":"system service",
        "系统工具":"system tool",
        "应用服务":"application service",
        "应用库":"application library",
        "应用工具":"application tool",
        "其它":"others",
    }

    edges = df_edges.values.tolist()
    out_label_set = set([])
    in_label_set = set([])
    for edge in edges:
        if edge[0] == name:
            label = df_data.loc[df_data["rpm_name"] == edge[1]]["all_label"].values.tolist()[0]
            if df_data.loc[df_data["rpm_name"] == edge[1]]["is_train"].values.tolist()[0] == 1:
                out_label_set.add(label_map[label])
        elif edge[1] == name:
            label = df_data.loc[df_data["rpm_name"] == edge[0]]["all_label"].values.tolist()[0]
            if df_data.loc[df_data["rpm_name"] == edge[0]]["is_train"].values.tolist()[0] == 1:
                in_label_set.add(label_map[label])

    s = "depending {}, depended by {}".format(" ".join(list(out_label_set)), " ".join(list(in_label_set)))
    return s

