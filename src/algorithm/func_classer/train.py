# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np
import transformers as tf
from transformers import BertTokenizer

import models
from transformers import AutoTokenizer

from sklearn.metrics import classification_report
# from global_variable import * 
#config = tf.compat.v1.ConfigProto(gpu_options=gpu_options)
#session = tf.compat.v1.Session(config=config)
import argparse
from utils import *
from models import *
import sys
import os
# 获取当前脚本所在的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算上两级目录的路径
parent_dir = os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))
# 添加上两级目录到系统路径中
sys.path.append(parent_dir)
# 导入需要调用的包或模块
import data_config

# 解析命令行参数
#tf.config.experimental.set_memory_growth('GPU-04edfd4f-8164-66b6-476d-9db80d794f53', True)
parser = argparse.ArgumentParser()
#parser.add_argument('--mode', type=str, default="all")
parser.add_argument('--use_graph_feat', type=str, default="false")
args = parser.parse_args()

# parameters
maxlen = 192   # BERT输入文本最大长度，超出截断，不足使用 [PAD] 补齐
batch_size = 1 # batch 大小
epochs = 20
learning_rate = 1e-5 # 训练轮次与学习率经过实验，推荐使用这两个值

classes = ['network', 'graph', 'multi-media', 'management', 'device', 'storage', 'virtual', 'Data_Convert', 'doc', 'general','programming','Debug','math','other']

# 结果输出目录
save_path = f'./human_labels.csv'
# BERT模型，根据名称从 Huggingface 加载
#bert_path = "/home/huan/workspce/data/bert-base-uncased"
bert_path = data_config.model_path

# 加载BERTTokenizer，与模型对应
tokenizer = AutoTokenizer.from_pretrained(bert_path)

# 节点数据预处理，将 summary 和 description 拼合，并将标签分为分类标签、分层标签、分层分类标签
# 边数据预处理，将 dot 文件表示的依赖关系转化为 csv 表示
#preprocess_data(save_path)

# 数据读取
print("开始读取数据")
df_data = pd.read_csv(save_path)
#print(df_data)
#df_edges = pd.read_csv(data_config.train_dot_data)

#mode = args.mode
label_str = "classes"
#if mode == "class":
#label_str = "class_label"
#if mode == "layer":
#label_str = "layer_label"

# 将数据分为 10 组，采用 10 折交叉验证
groups = split_dataset_by_label_ratio(df_data, label_str)
df_data["group_num"] = df_data["rpm_name"].apply(lambda x: get_group_num(x, groups))

# 在训练过程中用于将 index 与 name 和 label 进行对应的字典
idx_name_map, name_idx_map = get_idx_name_map(df_data["rpm_name"])
num_label_map, label_num_map = get_num_label_map(classes)

num_classes = len(classes)

#onehot_labels = encode_onehot(label_num_map, list(df_data[label_str]))
#print(onehot_labels)
#print(onehot_labels.shape)
df_data["ori_text"] = df_data["text"].apply(lambda x: x)
print("ori_text   ：",df_data["ori_text"])

def mask2onehot(data):
    name_label_dict = {}
    
    for ind in data.index :
        labelsss = set()
        for label in classes :
            if label in data[label_str][ind]:
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

for i in range(10):
    print("正在运行第 {} / 10 组训练集&测试集".format(i + 1))
    #df_data["is_train"] = df_data["group_num"].apply(lambda x: 0 if x == i else (2 if x == (i+1)%10 else 1))
    #df_data["is_train"] = df_data["group_num"].apply(lambda x: 0 if x == i else 1)
    df_data["is_train"] = 1

    if args.use_graph_feat == "true":
         print("正在构造额外特征，此步骤无GPU加速，预计耗时三分钟")
         df_data["text"] = df_data.apply(lambda x: "[CLS]" + build_edge_text_v2(x["rpm_name"], df_data, df_edges) + "[SEP]" + x["ori_text"] + "[SEP]", axis=1)
    else:
        df_data["text"] = df_data.apply(lambda x: "[CLS]" + x["ori_text"] + "[SEP]", axis=1)

    df_train = df_data.loc[df_data["is_train"] == 1]
    df_val = df_data.loc[df_data["is_train"] == 0]
    #df_test = df_data.loc[df_data["is_train"] == 0]

    def tokenize(df):
        # 使用 huggingface 提供的 API 构造 BERT 需要的输入特征，包括 Token Embedding、Segment Embedding、Position Embedding、Attention Mask
        #t = time.time()
        
        X_name = []
        X_input_ids = []
        X_token_type_ids = []
        X_attention_mask = []
        y = mask2onehot(df)
        
        for content,label,name in zip(list(df.text),list(df[label_str]),list(df["rpm_name"])):
            bert_input = tokenizer.encode_plus(content,
                add_special_tokens = False, # add [CLS], [SEP]
                max_length = maxlen, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
                truncation=True)
            #print(bert_input)
            #print("input_ids",bert_input['input_ids'].shape)
            X_input_ids.append(bert_input['input_ids'])
            X_token_type_ids.append(bert_input['token_type_ids'])
            X_attention_mask.append(bert_input['attention_mask'])
            #labels.append(label_num_map[ii]) 
        
            #print(np.array(labels))
            #y.append(label_num_map[label])
            #print("name:" , name , "label :",label_num_map[label])
            X_name.append(name)
        X_input_ids = np.array(X_input_ids)
        X_token_type_ids = np.array(X_token_type_ids)
        X_attention_mask = np.array(X_attention_mask)
        y = np.array(y)
        # y = encode_onehot(label_num_map, list(df[label_str]))
        # print('tokenizing time cost:',time.time()-t,'s.')
        return X_input_ids, X_token_type_ids, X_attention_mask, y, X_name

    # ========== model traing: ==========
    # 构造训练、验证、测试集 BERT 需要的输入特征
    X_input_ids_train, X_token_type_ids_train, X_attention_mask_train, y_train, name_train = tokenize(df_train)
    print("X_input_ids_train:",X_input_ids_train.shape)
    print("X_attention_mask_train:",X_attention_mask_train.shape)
    print("X_token_type_ids_train:",X_token_type_ids_train.shape)
    print("y_train:",y_train.shape)
    #X_input_ids_test, X_token_type_ids_test, X_attention_mask_test, y_test, name_test = tokenize(df_test)
    X_input_ids_val, X_token_type_ids_val, X_attention_mask_val, y_val, name_val = tokenize(df_val)

    data_package = [X_input_ids_train, X_token_type_ids_train, X_attention_mask_train, y_train,  X_input_ids_val, X_token_type_ids_val, X_attention_mask_val, y_val]

    # 调用模型并训练
    model = BERT(bert_path, maxlen, num_classes, learning_rate)
    model.train_val(data_package, batch_size, epochs, save_best=True)

    #print('test acc:', str(test_score))
    #print('best val acc:', str(best_val_score))
    #print('train acc list:\n', str(train_score_list))
    #print('val acc list:\n', str(val_socre_list), '\n')

    # 使用测试集测试，保存预测结果
    #model_input = [X_input_ids_test, X_attention_mask_test, X_token_type_ids_test]
    #output = model.predict(model_input)[:,:num_classes]
    #for i, name in enumerate(name_test):
        #print('ssssssssssssss')
        #L_label = np.array(range(num_classes))
        #model_input = [np.array([X_input_ids_test[i]]), np.array([X_attention_mask_test[i]]), np.array([X_token_type_ids_test[i]])]
        #pred_probs = output[i]
        #y_pred = pred_probs
        #for i in range(len(y_pred)):
            #max_value=max(y_pred[i])
            #for j in range(len(y_pred[i])):
                #if max_value==y_pred[i][j]:
                    #y_pred[i][j]=1
                #else:
                    #y_pred[i][j]=0
        #print(classification_report(y_train, y_pred))
        #print(pred_probs)
        #name_pred_map[name] = num_label_map[np.argmax(pred_probs)]

# 评价指标计算及结果输出
#df_data["pred"] = df_data["rpm_name"].apply(lambda x: name_pred_map[x])
#report = classification_report(df_data[label_str], df_data["pred"], digits=3)
#print(report)
#with open('{}/result.txt'.format(save_path), 'a+', encoding='utf-8') as f:
    #print(report, file=f)
#df_print = df_data[["rpm_name", "pred", "class_label", "layer_label", "all_label"]]
#df_print.to_csv("{}/result.csv".format(save_path), index=False)

