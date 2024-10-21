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
    
    return idx_name_map, name_idx_map

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

def build_edge_text_v2(name, df_data, df_edges):
    # 一阶邻居标签特征
    label_text_list_in = [["core", 0], ["system", 0], ["application", 0], ["other layer", 0], ["unknown layer", 0],
                       ["library", 0],["tool", 0], ["service", 0], ["other class", 0], ["unknown class", 0]]
    label_text_list_out = [["core", 0], ["system", 0], ["application", 0], ["other layer", 0], ["unknown layer", 0],
                       ["library", 0],["tool", 0], ["service", 0], ["other class", 0], ["unknown class", 0]]
    
#     label_text_list_in = [["library", 0],["tool", 0], ["service", 0], ["other class", 0], ["unknown class", 0]]
#     label_text_list_out = [["library", 0],["tool", 0], ["service", 0], ["other class", 0], ["unknown class", 0]]
    
#     label_text_list_in = [["core", 0], ["system", 0], ["application", 0], ["other layer", 0], ["unknown layer", 0]]
#     label_text_list_out = [["core", 0], ["system", 0], ["application", 0], ["other layer", 0], ["unknown layer", 0]]
    trans_map = {
        "核心":"core",
        "系统":"system",
        "应用":"application",
        "库":"library",
        "工具":"tool",
        "服务":"service",
        "其它":"other",
        "unknown class":"unknown class",
        "unknown layer":"unknown layer",
    }
    
    def triple2str(tup):
        if tup[1] == 0:
            return "[PAD]"
        else:
            return "{}".format(tup[0])
    
    this_data = df_data.loc[df_data["name"] == name]
    
    edges = df_edges.values.tolist()
    for edge in edges:
        if edge[0] == name:
            that_data = df_data.loc[df_data["name"] == edge[1]]
            if that_data["is_train"].values.tolist()[0] == 1:
                that_class_label = trans_map[that_data["class_label"].values.tolist()[0]]
                that_layer_label = trans_map[that_data["layer_label"].values.tolist()[0]]
            else:
                that_class_label = trans_map["unknown class"]
                that_layer_label = trans_map["unknown layer"]
            for tup in label_text_list_out:
                if tup[0] == that_class_label:
                    tup[1] = tup[1] + 1
                if tup[0] == that_layer_label:
                    tup[1] = tup[1] + 1
        elif edge[1] == name:
            that_data = df_data.loc[df_data["name"] == edge[0]]
            if that_data["is_train"].values.tolist()[0] == 1:
                that_class_label = trans_map[that_data["class_label"].values.tolist()[0]]
                that_layer_label = trans_map[that_data["layer_label"].values.tolist()[0]]
            else:
                that_class_label = trans_map["unknown class"]
                that_layer_label = trans_map["unknown layer"]
            for tup in label_text_list_in:
                if tup[0] == that_class_label:
                    tup[1] = tup[1] + 1
                if tup[0] == that_layer_label:
                    tup[1] = tup[1] + 1
                    
        
    s_list = []
    for tup in label_text_list_in:
        if tup[1] > 0:
            s_list.append(tup[0])
    in_res = " ".join(s_list)
    s_list = []
    for tup in label_text_list_out:
        if tup[1] > 0:
            s_list.append(tup[0])
    out_res = " ".join(s_list)
    
    
    def find_label(name, t):
        info = df_data.loc[df_data["name"] == name]
        if info["is_train"].values.tolist()[0] == 0:
            return "unknown"
        if t == "class":
            return trans_map[info["class_label"].values.tolist()[0]]
        if t == "layer":
            return trans_map[info["layer_label"].values.tolist()[0]]
        
    # 相似节点标签特征
    # sim_nodes = top_nodes[name]
    
    # sim_nodes_layer_label = [find_label(node, "layer") for node in sim_nodes]
    # sim_nodes_class_label = [find_label(node, "class") for node in sim_nodes]
    # sim_nodes_label = sim_nodes_layer_label + sim_nodes_class_label
    
    # sim_res = " ".join(sim_nodes_label)
    
    # final_res = "depended by: {}, depending on: {}, similar: {}".format(in_res, out_res, sim_res)
#     final_res = "depended by: {}, depending on: {}, similar: {}".format("", "", sim_res)
    final_res = "depended by: {}, depending on: {}, similar: {}".format(in_res, out_res, "")
#     final_res = "depended by: {}, depending on: {}, similar: {}".format("", "", "")
#     print(final_res)
    return final_res

def split_dataset_by_label_ratio(_data, label_name):
    print("label_name" , label_name)
    data = shuffle(_data)
    # 计算每个标签的比例
    label_counts = data[label_name].value_counts(normalize=True)

    # 计算每组应该包含的每个标签的样本数量
    num_groups = 10
    group_sizes = {label: round(label_counts[label] * len(data) / num_groups) for label in label_counts.index}

    # 对每个标签的样本进行排序
    sorted_data = data.sort_values(label_name)

    # 将样本分配到各个组中
    groups = []
    for i in range(num_groups):
        group = pd.DataFrame(columns=data.columns)
        for label in label_counts.index:
            # 选择该标签下尚未被分配到组中的样本
            label_data = sorted_data[sorted_data[label_name] == label].iloc[group_sizes[label]*i:group_sizes[label]*(i+1)]
            # group = group.append(label_data)
            group = pd.concat([group, pd.DataFrame(label_data)], ignore_index=True)
        groups.append(set(group["name"]))
 
    print(groups)
    print(len(groups))
    return groups

def get_group_num(name, groups):
    for i in range(10):
        if name in groups[i]:
            return i
    return 0

def encode_onehot(label_num_map, labels):
    classes_dict = {label: np.identity(len(label_num_map))[label_num_map[label], :] for label in label_num_map}
    labels_onehot = np.array(list(map(classes_dict.get, labels)), dtype=np.int32)
    print("labels  onehot ......")
    print(labels_onehot.shape)

    return labels_onehot

def divideDataSet(csv_name):
    df = pd.read_csv(csv_name)
    df["text"] = df.apply(lambda x: x["summary"] + " " + x["description"], axis=1)
    contents = df["text"]
    labels = df["class"]

    df["class_num"] = df.apply( lambda row: len(re.compile(r"'(.*?)'").findall(row['class'])), axis=1)
    label_name = "class_num"
    print("label_name" , label_name)
    data = shuffle(df)
    # 计算每个标签的比例

    label_counts = data[label_name].value_counts(normalize=True)
    # 计算每组应该包含的每个标签的样本数量
    num_groups = 10
    group_sizes = {label: round(label_counts[label] * len(data) / num_groups) for label in label_counts.index}

    # 对每个标签的样本进行排序
    sorted_data = data.sort_values(label_name)
    # 将样本分配到各个组中
    groups = []
    for i in range(num_groups):
        group = pd.DataFrame(columns=data.columns)
        for label in label_counts.index:
            # 选择该标签下尚未被分配到组中的样本
            label_data = sorted_data[sorted_data[label_name] == label].iloc[group_sizes[label]*i:group_sizes[label]*(i+1)]
            # group = group.append(label_data)
            group = pd.concat([group, pd.DataFrame(label_data)], ignore_index=True)
        groups.append(set(group["name"]))
    #print(groups)
    print(len(groups))
    return groups
