# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

#-*-coding:utf-8-*-

import csv
import json
from pyexpat import features
import random
import collections
import nltk
import torch.nn as nn
import torch
import re
from tqdm.auto import tqdm
from transformers import BertModel, BertTokenizer, BertForSequenceClassification

import requests
from bs4 import BeautifulSoup




import sys
sys.path.append('..')
from .set import node_set
from .set import load_data

def csvdict_to_all_set_bin(features, dot_path, test_set_json):
    dic = {"内核": 1 ,"内核服务": 1 ,
        "核心工具": 1 ,"核心库": 1 ,"核心服务": 1 ,"基础环境": 1 ,
        "系统工具": 2 ,"系统服务": 2 ,"系统库": 2 ,"系统应用":2,
        "虚拟化": 3 ,"应用库": 3 ,"应用工具":3,"应用服务":3,"字体":3,
        "数据库":3,"云计算":3,"大数据":3,"桌面":3,"云原生":3,
        "编程语言":2}
    #features = []
    name_to_index = {}

    # 从文件中读入依赖关系对list
    dependency_relationship = load_data(dot_path) #[before, after] == before -> after

    # s 是存储所有节点的set
    s = node_set()
    # 将依赖关系对list添加到点集中,name2node是个字典，存名和pre，名和post，每个节点都存了in和out
    s.add_pairs(dependency_relationship)

    # 找到最大的出度和入度，用于补全
    max_indegree = 0
    max_outdegree = 0
    # features是列表, 每一项是字典
    for feature in features:
        if feature['src_name'] not in s.name2node:
            continue
        if len(s.name2node[feature['src_name']].in_node) > max_indegree:
            max_indegree = len(s.name2node[feature['src_name']].in_node)
        if len(s.name2node[feature['src_name']].out_node) > max_outdegree:
            max_outdegree = len(s.name2node[feature['src_name']].out_node)
    
    print('最大的入度：', max_indegree)
    print('最大的出度：', max_outdegree)


    # features是列表
    for feature in features:
        feature['indegree'] = []
        feature['indegree_description'] = []
        feature['indegree_summary'] = []
        feature['outdegree'] = []
        feature['outdegree_description'] = []
        feature['outdegree_summary'] = []

        if feature['src_name'] not in s.name2node:
            continue
        for node in s.name2node[feature['src_name']].in_node:
            if node.packages[0] not in name_to_index:
                continue
            if features[name_to_index[node.packages[0]]]['src_name'] != '':
                feature['indegree'].append(features[name_to_index[node.packages[0]]]['src_name'])
            if features[name_to_index[node.packages[0]]]['description'] != '':
                feature['indegree_description'].append(features[name_to_index[node.packages[0]]]['description'])
            if features[name_to_index[node.packages[0]]]['summary'] != '':
                feature['indegree_summary'].append(features[name_to_index[node.packages[0]]]['summary'])


        for node in s.name2node[feature['src_name']].out_node:
            if node.packages[0] not in name_to_index:
                continue
            if features[name_to_index[node.packages[0]]]['src_name'] != '':
                feature['outdegree'].append(features[name_to_index[node.packages[0]]]['src_name'])
            if features[name_to_index[node.packages[0]]]['description'] != '':
                feature['outdegree_description'].append(features[name_to_index[node.packages[0]]]['description'])
            if features[name_to_index[node.packages[0]]]['summary'] != '':
                feature['outdegree_summary'].append(features[name_to_index[node.packages[0]]]['summary'])

    for feature in features:
        if len(feature['indegree']) == 0:
            feature['indegree'].append('')
        if len(feature['indegree_description']) == 0:
            feature['indegree_description'].append('')
        if len(feature['indegree_summary']) == 0:
            feature['indegree_summary'].append('')
        
        if len(feature['outdegree']) == 0:
            feature['outdegree'].append('')
        if len(feature['outdegree_description']) == 0:
            feature['outdegree_description'].append('')
        if len(feature['outdegree_summary']) == 0:
            feature['outdegree_summary'].append('')

    random.shuffle(features)
    l = len(features)
    test_set = features

    with open(test_set_json, 'w') as f:
        json.dump(features,f)

    print("csvdict_to_all_set_bin run over")


def csv_and_dot_to_json(csv_path, dot_path, test_set_json):
    dic = {"内核": 1 ,"内核服务": 1 ,
        "核心工具": 1 ,"核心库": 1 ,"核心服务": 1 ,"基础环境": 1 ,
        "系统工具": 2 ,"系统服务": 2 ,"系统库": 2 ,"系统应用":2,
        "虚拟化": 3 ,"应用库": 3 ,"应用工具":3,"应用服务":3,"字体":3,
        "数据库":3,"云计算":3,"大数据":3,"桌面":3,"云原生":3,
        "编程语言":2}
    features = []
    name_to_index = {}

    # 读取文件
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        #feature = {'label': []}
        for row in reader:
            feature ={'src_name' : row[0]}
            feature['summary'] = row[4].replace("\n"," ",-1)
            feature['description'] = row[1].replace("\n"," ",-1)
            name_to_index[feature['src_name']] = len(features) #计算字典在放入当前项时的长度
            features.append(feature)
    #上述代码：把csv存放到features中去

    # 从文件中读入依赖关系对list
    dependency_relationship = load_data(dot_path) #[before, after] == before -> after

    # s 是存储所有节点的set
    s = node_set()
    # 将依赖关系对list添加到点集中,name2node是个字典，存名和pre，名和post，每个节点都存了in和out
    s.add_pairs(dependency_relationship)

    # 找到最大的出度和入度，用于补全
    max_indegree = 0
    max_outdegree = 0
    # features是列表, 每一项是字典
    for feature in features:
        if feature['src_name'] not in s.name2node:
            continue
        if len(s.name2node[feature['src_name']].in_node) > max_indegree:
            max_indegree = len(s.name2node[feature['src_name']].in_node)
        if len(s.name2node[feature['src_name']].out_node) > max_outdegree:
            max_outdegree = len(s.name2node[feature['src_name']].out_node)
    print('最大的入度：', max_indegree)
    print('最大的出度：', max_outdegree)


    # features是列表
    for feature in features:
        feature['indegree'] = []
        feature['indegree_description'] = []
        feature['indegree_summary'] = []
        feature['outdegree'] = []
        feature['outdegree_description'] = []
        feature['outdegree_summary'] = []

        if feature['src_name'] not in s.name2node:
            continue
        # 遍历每个节点的in_node
        for node in s.name2node[feature['src_name']].in_node:
            if node.packages[0] not in name_to_index:
                continue
            if features[name_to_index[node.packages[0]]]['src_name'] != '':
                feature['indegree'].append(features[name_to_index[node.packages[0]]]['src_name'])
            if features[name_to_index[node.packages[0]]]['description'] != '':
                feature['indegree_description'].append(features[name_to_index[node.packages[0]]]['description'])
            if features[name_to_index[node.packages[0]]]['summary'] != '':
                feature['indegree_summary'].append(features[name_to_index[node.packages[0]]]['summary'])


        for node in s.name2node[feature['src_name']].out_node:
            if node.packages[0] not in name_to_index:
                continue
            if features[name_to_index[node.packages[0]]]['src_name'] != '':
                feature['outdegree'].append(features[name_to_index[node.packages[0]]]['src_name'])
            if features[name_to_index[node.packages[0]]]['description'] != '':
                feature['outdegree_description'].append(features[name_to_index[node.packages[0]]]['description'])
            if features[name_to_index[node.packages[0]]]['summary'] != '':
                feature['outdegree_summary'].append(features[name_to_index[node.packages[0]]]['summary'])

    for feature in features:
        if len(feature['indegree']) == 0:
            feature['indegree'].append('')
        if len(feature['indegree_description']) == 0:
            feature['indegree_description'].append('')
        if len(feature['indegree_summary']) == 0:
            feature['indegree_summary'].append('')
        
        if len(feature['outdegree']) == 0:
            feature['outdegree'].append('')
        if len(feature['outdegree_description']) == 0:
            feature['outdegree_description'].append('')
        if len(feature['outdegree_summary']) == 0:
            feature['outdegree_summary'].append('')

    random.shuffle(features)
    l = len(features)
    test_set = features

    print(f'总长度：{l}, 测试集长度：{len(test_set)}')

    with open(test_set_json, 'w') as f:
        json.dump(features,f)

    print("to json run over")





def check_label_partition(features):
    count = collections.defaultdict(int)
    mcount = collections.defaultdict(int)

    total = 0
    # 每个label的计数和总计数
    for feature in features:
        for label in feature['label']:
            count[label] += 1
            total += 1
    
    print('single label：')
    for key in count:
        print(f'{key} : {count[key] * 100/total}%')

    for feature in features:
        t = '-'.join([f'{l}' for l in feature['label'] ])
        mcount[t] += 1
    
    print('multi label：')
    for key in mcount:
        print(f'{key} : {mcount[key] * 100/len(features)}%')

# nltk 处理
def nltk_process(s):
    # 分词
    words = nltk.word_tokenize(s)
    # 去除符号 和 停用词
    interpunctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
    stops = set(nltk.corpus.stopwords.words("english"))
    words = [word for word in words if word not in interpunctuations]
    words = [word for word in words if word not in stops]
    # 词干提取 
    #words = [nltk.stem.PorterStemmer().stem(word) for word in words]
    # 词性还原 
    #words = [nltk.stem.WordNetLemmatizer().lemmatize(word) for word in words]

    #print(words)
    return " ".join(words).replace("\n"," ",-1)    


def generate_vector_map():
    dic = {"内核": 1 ,"内核服务": 1 ,
        "核心工具": 1 ,"核心库": 1 ,"核心服务": 1 ,"基础环境": 1 ,
        "系统工具": 2 ,"系统服务": 2 ,"系统库": 2 ,"系统应用":2,
        "虚拟化": 3 ,"应用库": 3 ,"应用工具":3,"应用服务":3,"字体":3,
        "数据库":3,"云计算":3,"大数据":3,"桌面":3,"云原生":3,
        "编程语言":2}
    features = []
    name_to_index = {}



    # 读取文件
    with open(f'./data/1228_bin.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)

        for row in reader:
            feature = {'label': []}
            if row[6] == '':
                continue
            single_package_class = row[6].split('/')

            is_labeled = False

            for classes in single_package_class:
                layer = dic[classes]
                if layer != '':
                    is_labeled = True
                else:
                    continue

                repeat = False
                for label in feature['label']:
                    if label == layer:
                        repeat = True
                if not repeat:
                    feature['label'].append(layer)

            if not is_labeled:
                continue

            feature['label'].sort()
            feature['src_name'] = row[0]
            feature['summary'] = row[4].replace("\n"," ",-1)
            feature['description'] = row[1].replace("\n"," ",-1)

            name_to_index[feature['src_name']] = len(features)
            if len(feature['label']) == 1:
                features.append(feature)

    data = ''
    for feature in features:
        data += f'{feature["src_name"]}\t{feature["label"][0]}\n'

    with open('./data/1228_bin_single_label.txt', 'w') as f:
        f.write(data)

def generate_label_partition():
    dic = {"内核": 1 ,"内核服务": 1 ,
        "核心工具": 1 ,"核心库": 1 ,"核心服务": 1 ,"基础环境": 1 ,
        "系统工具": 2 ,"系统服务": 2 ,"系统库": 2 ,"系统应用":2,
        "虚拟化": 3 ,"应用库": 3 ,"应用工具":3,"应用服务":3,"字体":3,
        "数据库":3,"云计算":3,"大数据":3,"桌面":3,"云原生":3,
        "编程语言":2}
    label1 = collections.defaultdict(int)
    label2 = collections.defaultdict(int)

    # 读取文件
    with open(f'./data/1228_bin.csv', 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
        for row in reader:
            feature = {'label': []}
            if row[6] == '':
                continue
            single_package_class = row[6].split('/')

            for classes in single_package_class:
                label1[classes] += 1
                label2[dic[classes]] += 1
    print('label1', len(label1))
    for key in label1:
        print(f'{key}|{label1[key]}')

    print('label2', len(label2))
    for key in label2:
        print(f'{key}|{label2[key]}')

def generate_fasttext_dataset():
    with open('./data/1228_bin_set.json', 'r') as f:
        features = json.load(f)

        test_desc = "\n".join([" ".join(["__label__" + str(l) for l in feat['label']]) + " " + nltk_process(feat['description']) for feat in features if feat['description_nltk'] != ''])
        test_sum = "\n".join([" ".join(["__label__" + str(l) for l in feat['label']]) + " " + nltk_process(feat['summary_nltk']) for feat in features if feat['summary_nltk'] != ''])

        test_desc = "\n".join([" ".join(["__label__" + str(l) for l in feat['label']]) + " " + nltk_process(feat['description']) for feat in features if feat['description_nltk'] != ''])
        test_sum = "\n".join([" ".join(["__label__" + str(l) for l in feat['label']]) + " " + nltk_process(feat['summary']) for feat in features if feat['summary_nltk'] != ''])


