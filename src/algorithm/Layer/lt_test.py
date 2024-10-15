# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import os
import csv
import torch
import json
import sys
import .nltk
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader)
import torch.optim as optim
from tqdm.auto import tqdm
from bert_pkg_emb import do_layer;

class Args():
    def __init__(self):
        self.model_dir = f'./data/layer_best_model.bin'
        self.train_dir = f'./data/1228_bin_train_set.json'
        self.test_dir = f'./data/1228_bin_test_set.json'
        self.loss_dir = f'./data/1228_bin_loss_bert-no-cnn-all-new.json'

        self.epochs = 2
        self.lr = 0.000001
        
        self.num_labels = 3

        self.dataloader_batch_size = 2

        self.dropout_rate = 0.1

        self.device = "cuda" if torch.cuda.is_available() else "cpu"    # 计算设备

        self.best_accuracy = 0.0

        #self.package_vector = get_package_vector()
        self.package_emb = get_package_emb()
        self.label_index_dict = {
            '1': 0,
            '2': 1,
            '3': 2,
            '4': 3,
            '2-3': 4,
            '2-4': 5,
            '3-4': 6,
        }
        self.loss_record = []
    
    def label_index(self, label):
        return self.label_index_dict[label]


def get_package_emb():
    f =  open('./data/pkg_embedding_bidirect_padding.json', 'r')
    package_emb = json.load(f)
    return package_emb

def get_dataset(filename, args):
    f = open(filename, 'r')
    features = json.load(f)
    return LayerDataset(features, args)

def csv_to_set_bin(csv_path, dot_path, train_set_json, test_set_json):
    dic = {"内核": "1" ,"内核服务": "1" ,
        "核心工具": 1 ,"核心库": 1 ,"核心服务": 1 ,"基础环境": 1 ,
        "系统工具": 2 ,"系统服务": 2 ,"系统库": 2 ,"系统应用":2,
        "虚拟化": 3 ,"应用库": 3 ,"应用工具":3,"应用服务":3,"字体":3,
        "数据库":4,"云计算":4,"大数据":4,"桌面":4,"云原生":4,
        "编程语言":"2"}

    features = []
    name_to_index = {}

    # 读取文件
    with open(csv_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        for row in reader:
            feature = {'label': []}
            if row[6] == '':
                continue
            single_package_class = row[6].split('/')
            is_labeled = False

            # 因为labe存在”系统工具/虚拟化“这样的情况，此前先用/隔开
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
            feature['summary_nltk'] = nltk_process(feature['summary'])
            feature['description_nltk'] = nltk_process(feature['description'])

            name_to_index[feature['src_name']] = len(features) #计算字典在放入当前项时的长度
            features.append(feature)

    #上述代码：把csv存放到features中去
    # 从文件中读入依赖关系对list
    dependency_relationship = load_data(dot_path)
    s = node_set()

    s.add_pairs(dependency_relationship)

    # 找到最大的出度和入度，用于补全
    max_indegree = 0
    max_outdegree = 0

    # features是列表
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
        feature['indegree_description_nltk'] = []
        feature['indegree_summary'] = []
        feature['indegree_summary_nltk'] = []
        feature['outdegree'] = []
        feature['outdegree_description'] = []
        feature['outdegree_description_nltk'] = []
        feature['outdegree_summary'] = []
        feature['outdegree_summary_nltk'] = []


        if feature['src_name'] not in s.name2node:
            continue

        # 遍历每个节点的in_node
        for node in s.name2node[feature['src_name']].in_node:
            # name_to_index 是 每个节点在字典feature的index {name:index}
            if node.packages[0] not in name_to_index:
                continue
            if features[name_to_index[node.packages[0]]]['src_name'] != '':
                feature['indegree'].append(features[name_to_index[node.packages[0]]]['src_name'])
            if features[name_to_index[node.packages[0]]]['description'] != '':
                feature['indegree_description'].append(features[name_to_index[node.packages[0]]]['description'])
            if features[name_to_index[node.packages[0]]]['description_nltk'] != '':
                feature['indegree_description_nltk'].append(features[name_to_index[node.packages[0]]]['description_nltk'])
            if features[name_to_index[node.packages[0]]]['summary'] != '':
                feature['indegree_summary'].append(features[name_to_index[node.packages[0]]]['summary'])
            if features[name_to_index[node.packages[0]]]['summary_nltk'] != '':
                feature['indegree_summary_nltk'].append(features[name_to_index[node.packages[0]]]['summary_nltk'])

        for node in s.name2node[feature['src_name']].out_node:
            if node.packages[0] not in name_to_index:
                continue

            if features[name_to_index[node.packages[0]]]['src_name'] != '':
                feature['outdegree'].append(features[name_to_index[node.packages[0]]]['src_name'])

            if features[name_to_index[node.packages[0]]]['description'] != '':
                feature['outdegree_description'].append(features[name_to_index[node.packages[0]]]['description'])

            if features[name_to_index[node.packages[0]]]['description_nltk'] != '':
                feature['outdegree_description_nltk'].append(features[name_to_index[node.packages[0]]]['description_nltk'])

            if features[name_to_index[node.packages[0]]]['summary'] != '':
                feature['outdegree_summary'].append(features[name_to_index[node.packages[0]]]['summary'])

            if features[name_to_index[node.packages[0]]]['summary_nltk'] != '':
                feature['outdegree_summary_nltk'].append(features[name_to_index[node.packages[0]]]['summary_nltk'])


    for feature in features:
        if len(feature['indegree']) == 0:
            feature['indegree'].append('')

        if len(feature['indegree_description']) == 0:
            feature['indegree_description'].append('')

        if len(feature['indegree_description_nltk']) == 0:
            feature['indegree_description_nltk'].append('')

        if len(feature['indegree_summary']) == 0:
            feature['indegree_summary'].append('')

        if len(feature['indegree_summary_nltk']) == 0:
            feature['indegree_summary_nltk'].append('')

        if len(feature['outdegree']) == 0:
            feature['outdegree'].append('')

        if len(feature['outdegree_description']) == 0:
            feature['outdegree_description'].append('')

        if len(feature['outdegree_description_nltk']) == 0:
            feature['outdegree_description_nltk'].append('')

        if len(feature['outdegree_summary']) == 0:
            feature['outdegree_summary'].append('')

        if len(feature['outdegree_summary_nltk']) == 0:
            feature['outdegree_summary_nltk'].append('')

    # 划分测试集和训练集

    random.shuffle(features)
    l = len(features)
    train_set = features[int(l * (2/10)):]
    test_set = features[:int(l * (2/10))]  

    with open(train_set_json, 'w') as f:
        json.dump(train_set, f)
    with open(test_set_json, 'w') as f:
        json.dump(test_set,f)

    print("csv_to_set_bin run over")




class LayerDataset(Dataset):
    def __init__(self, features, args):
        self.data = features
        self.args = args

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        src_name = self.data[index]['src_name']
        description = self.data[index]['description']
        summary = self.data[index]['summary']

        indegree_description = [''] # self.data[index]['indegree_description']
        indegree_summary = self.data[index]['indegree_summary'][:20]

        outdegree_description = [''] # self.data[index]['outdegree_description']
        outdegree_summary = [''] # self.data[index]['outdegree_summary']

        return src_name, description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary 

def collate_fn(data):
    src_names, descriptions, summarys, indegree_descriptions, indegree_summarys, outdegree_descriptions, outdegree_summarys, labels = [], [], [], [], [], [], [], []

    for src_name, description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary in data:
        src_names.append(src_name)
        descriptions.append(description)
        summarys.append(summary)
        indegree_descriptions.append(indegree_description)
        indegree_summarys.append(indegree_summary)
        outdegree_descriptions.append(outdegree_description)
        outdegree_summarys.append(outdegree_summary)

    return src_names, descriptions, summarys, indegree_descriptions, indegree_summarys, outdegree_descriptions, outdegree_summarys



class Bertcls(nn.Module):
    def __init__(self, args):
        # args 为各种参数
        super().__init__()
        # 载入预训练 bert 模型, args.bert_dir 为预训练模型位置
        self.device = args.device
        self.bert = BertModel.from_pretrained(data_config.model_path).to(args.device)
        self.drop = nn.Dropout(args.dropout_rate)
        self.bertForSequenceClassification = BertForSequenceClassification.from_pretrained(data_config.model_path, num_labels=args.num_labels, ignore_mismatched_sizes=True).to(args.device)

        self.l1 = nn.Linear(768 * 5, 512)
        #self.l1 = nn.Linear(768 * 4 + 512 , 512)
        self.l2 = nn.Linear(2048 , 1024)
        self.l3 = nn.Linear(1024 , 512)
        # self.l1 = nn.Linear(768 * 3 + 4 + 512, 512)
        # num_labels 为标签数量 
        self.cls = nn.Linear(512, args.num_labels)
        #self.cls = nn.Linear(768, args.num_labels)

        self.bertTokenizer = BertTokenizer.from_pretrained(data_config.model_path)


    def forward(self, description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary, src_name):
        # bert 得到句子编码
        token_ids_pkg = self.bertTokenizer(src_name, return_tensors='pt', padding=True, truncation=True).to(self.device)
        emb_pkg_emb = (self.bert(**token_ids_pkg).last_hidden_state)[:, 0, :]

        token_ids_description = self.bertTokenizer(description, return_tensors='pt', padding=True, truncation=True).to(self.device)
        emb_description = (self.bert(**token_ids_description).last_hidden_state)[:, 0, :]
        
        token_ids_summary = self.bertTokenizer(summary, return_tensors='pt', padding=True, truncation=True).to(self.device)
        emb_summary = (self.bert(**token_ids_summary).last_hidden_state)[:, 0, :]

        emb_in_descs_total = []
        emb_in_summary_total = []

        for descs in indegree_description:

            emb_accumulate = torch.zeros([768]).to(self.device)
            # print('emb_accumulate.shape ',emb_accumulate.shape)

            for i in range(0, len(descs), 20):
                descs_segment = descs[i:i+20]
                token_ids_descs_segment = self.bertTokenizer(descs_segment, return_tensors='pt', padding=True, truncation=True).to(self.device)
                emb_segment = self.bert(**(token_ids_descs_segment)).last_hidden_state
                emb_segment = emb_segment[:, 0, :] # 取0列中所有的向量，第三个参数代表0列是向量时取向量第几个值。
                emb_segment = torch.sum(emb_segment, 0) / (len(indegree_description) * 4)
                emb_accumulate += emb_segment

            # print('emb_accumulate.shape ',emb_accumulate.shape)


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

def train(dataloader, model, loss_fn, optimizer, args):
    model.train()

    progress_bar = tqdm(range(len(dataloader)))
    acc = 0.0
    for src_name, description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary, label in dataloader:

        label = label.to(args.device)

        # 计算损失
        pred = model(description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary, src_name).to(args.device)
        loss = loss_fn(pred, label.float())

        # 反向传播
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        progress_bar.update(1)
        args.loss_record.append(loss.item())

        if progress_bar.n % 300 == 0:
            acc = test_multi(args.test_dataloader, model, loss_fn, args)
            print("now acc=%f, best_acc=%f" %(acc, args.best_accuracy))
            if acc > args.best_accuracy:
                args.best_accuracy = acc
                model_to_save = model.module if hasattr(model, 'module') else model
                torch.save(model_to_save.state_dict(), args.model_dir)
    return acc

def train1(train_dir, test_dir):
    args = Args()
    args.train_dir = train_dir
    args.test_dir  = test_dir

    # 返回：description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary, vector, package_emb
    # vector 是ONE-HOT
    # package_embs 是词嵌入
    train_dataset = get_dataset(args.train_dir, args)
    test_dataset  = get_dataset(args.test_dir, args)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.dataloader_batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.dataloader_batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )

    args.train_dataloader = train_dataloader
    args.test_dataloader = test_dataloader

    model = Bertcls(args).to(args.device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    #for t in range(args.epochs):
    for t in range(3):
        print(f"Epoch {t+1}\n-------------------------------")
        a=train(train_dataloader, model, loss_fn, optimizer, args)
        print("acc========%d" %a)

    print("loss_dir =%s" %args.loss_dir)
    with open(args.loss_dir, 'w') as fw:
        json.dump(args.loss_record, fw)
    print(f"test {t+1}\n-------------------------------")
    return 0
csv_to_set_bin('./data/1228_bin.csv', './data/rpm_all.dot', './data/1228_bin_train_set.json', './data/1228_bin_test_set.json')
