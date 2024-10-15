# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np
from transformers import BertTokenizer
import models
import torch
from torch import nn
from transformers import AutoTokenizer

from sklearn.metrics import classification_report
# from global_variable import * 
# config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
# session = tf.compat.v1.Session(config=config)
import argparse
from utils import *
from models import *
# 解析命令行参数
parser = argparse.ArgumentParser()
#parser.add_argument('--mode', type=str, default="all")
parser.add_argument('--use_graph_feat', type=str, default="false")
args = parser.parse_args()

# parameters
maxlen = 192   # BERT输入文本最大长度，超出截断，不足使用 [PAD] 补齐
batch_size = 1 # batch 大小
epochs = 10
learning_rate = 1e-5 # 训练轮次与学习率经过实验，推荐使用这两个值

#classes = ['network', 'graph', 'muilti-media', 'management', 'device', 'storage', 'virtual', 'DC', 'doc', 'general','programming','test/Debug','math']
classes = ['network', 'graph', 'multi-media', 'management', 'device', 'storage', 'virtual', 'Data_Convert', 'doc', 'general','programming','Debug','math','other']

# 结果输出目录
save_path = f'./functions.csv'
save_test_path = f'./0401.csv'
# BERT模型，根据名称从 Huggingface 加载
bert_path = "/home/huan/workspce/data/bert-base-uncased"
# 加载BERTTokenizer，与模型对应
tokenizer = AutoTokenizer.from_pretrained(bert_path)

# 节点数据预处理，将 summary 和 description 拼合，并将标签分为分类标签、分层标签、分层分类标签
# 边数据预处理，将 dot 文件表示的依赖关系转化为 csv 表示
#preprocess_data(save_path)

# 数据读取
print("开始读取数据")
df_train = pd.read_csv(save_path)
df_test_info = pd.read_csv(save_test_path)
#df_test_info = pd.concat([df_train,df_test],axis=0)
#print(df_test_info)
#df_edges = pd.read_csv(data_config.train_dot_data)

#mode = args.mode
label_str = "labels"
#if mode == "class":
#label_str = "class_label"
#if mode == "layer":
#label_str = "layer_label"

# 将数据分为 10 组，采用 10 折交叉验证
#groups = split_dataset_by_label_ratio(df_test_info, label_str)
#df_test_info["group_num"] = df_test["rpm_name"].apply(lambda x: get_group_num(x, groups))

# 在训练过程中用于将 index 与 name 和 label 进行对应的字典
idx_name_map, name_idx_map = get_idx_name_map(df_test_info["rpm_name"])
num_label_map, label_num_map = get_num_label_map(classes)

num_classes = len(classes)

#onehot_labels = encode_onehot(label_num_map, list(df_test_info[label_str]))
#print(onehot_labels)
#print(onehot_labels.shape)
df_test_info["ori_text"] = df_test_info["text"].apply(lambda x: x)

def mask2onehot(data):
    name_label_dict = {}
    
    for ind in data.index :
        labelsss = set()
        for label in classes :
            if label in  data[label_str][ind]:
                labelsss.add(label)
        name_label_dict[ind] = labelsss
    
    label_onehot = []
    for item in name_label_dict:
        y = [0] * len(classes)
        #data['labels'][item]
        for label in name_label_dict[item]:
            y[label_num_map[label]] = 1
        label_onehot.append(y)
    return label_onehot

name_pred_map = {}

print("ori_text：",df_test_info["ori_text"])
df_test_info["text"] = df_test_info.apply(lambda x: str("[CLS]") + str(x["ori_text"]) + str("[SEP]"), axis=1)
print("text：",df_test_info["text"])
def tokenize(df):
        # 使用 huggingface 提供的 API 构造 BERT 需要的输入特征，包括 Token Embedding、Segment Embedding、Position Embedding、Attention Mask
        #t = time.time()
        
    X_name = []
    X_input_ids = []
    X_token_type_ids = []
    X_attention_mask = []
    y = mask2onehot(df)
        
    for content,name in zip(list(df.text),list(df["rpm_name"])):
        bert_input = tokenizer.encode_plus(content,
            add_special_tokens = False, # add [CLS], [SEP]
            max_length = maxlen, # max length of the text that can go to BERT
            pad_to_max_length = True, # add [PAD] tokens
            return_attention_mask = True, # add attention mask to not focus on pad tokens
            truncation=True)
            #print(bert_input)
            #print("input_ids",bert_input['input_ids'].shape)
        X_input_ids.append(bert_input['input_ids'])
        #print("token_type_ids",bert_input['token_type_ids'].shape)
        X_token_type_ids.append(bert_input['token_type_ids'])
        X_attention_mask.append(bert_input['attention_mask'])
        #print("attention_mask",bert_input['attention_mask'].shape)
        #labels.append(label_num_map[ii]) 
        
        X_name.append(name)
    X_input_ids = np.array(X_input_ids)
    X_token_type_ids = np.array(X_token_type_ids)
    X_attention_mask = np.array(X_attention_mask)
    #y = encode_onehot(label_num_map, list(df[label_str]))
    # print('tokenizing time cost:',time.time()-t,'s.')
    return X_input_ids, X_token_type_ids, X_attention_mask, X_name

# ========== model traing: ==========
# 构造训练、验证、测试集 BERT 需要的输入特征
X_input_ids_train, X_token_type_ids_train, X_attention_mask_train,  name_train = tokenize(df_test_info)

# 调用模型并训练
model = BERT(bert_path, maxlen, num_classes, learning_rate)

# 使用测试集测试，保存预测结果
model_input = [X_input_ids_train, X_attention_mask_train, X_token_type_ids_train]
output = model.predict(model_input)[:,:num_classes]
names_list = []
pres_list = []
name_pred_map = {}
for i, name in enumerate(name_train):
    pres = torch.tensor(output[i])
    sigmoid = nn.Sigmoid()
    probs = sigmoid(pres)
    pred_classes = []
    predictions = np.zeros(probs.shape)
    predictions[np.where(probs > 0.55)] = 1
    
    max_value=max(probs)
    for j in range(len(predictions)):
        if max_value == probs[j] or predictions[j] == 1:
            pred_classes.append(classes[j])
            predictions[j]=1

    names_list.append(name)
    pres_list.append(pred_classes)
    #name_pred_map[name] = pred_classes

#print(name_pred_map)
df = pd.DataFrame({'rpm_name':names_list , 'labels':pres_list})
df.to_csv("./result_0401.csv", index=False)
    #print(predictions)
    #L_label = np.array(range(num_classes))
    #model_input = [np.array([X_input_ids_train[i]]), np.array([X_attention_mask_train[i]]), np.array([X_token_type_ids_train[i]])]
    #pred_probs = output[i]
    #y_pred = output
    #print("sigmoid:",tf.sigmoid(output))
    #pred_classes = []
    #max_value=max(y_pred[i])
    #for j in range(len(y_pred[i])):
            #if tf.sigmoid(y_pred[i])[j] > 0.3:
        #if max_value == y_pred[i][j]:
            #pred_classes.append(classes[j])
            #y_pred[i][j]=1
        #else:
            #y_pred[i][j]=0
    #labless = df_test_info[df_test['rpm_name'] == name]
    #if len(labless[label_str]) == 0:
    #name_pred_map[name] = pred_classes
    #else :
    #names_list.append(name)
    #pres_list.append(pred_classes)
    #name_pred_map[name] = pred_classes

#df2 = df_train[['rpm_name','labels']]
#pre_df = pd.concat([df,df2],axis=0)
#print(pre_df)

# 评价指标计算及结果输出
#df_test_info["pred"] = df_test["rpm_name"].apply(lambda x: name_pred_map[x])
#report = classification_report(df_test_info[label_str], df_test["pred"], digits=3)
#print(report)
#with open('{}/result.txt'.format(save_path), 'a+', encoding='utf-8') as f:
#print(report, file=f)
#df_print = df_test_info[["rpm_name", "pred", "class_label", "layer_label", "all_label"]]
#df_print.to_csv("{}/result.csv".format(save_path), index=False)

