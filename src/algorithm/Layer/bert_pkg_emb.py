# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

#-*-coding:utf-8-*-

import os
import torch
import json
import sys
import torch.nn.functional as F
from transformers import BertModel, BertTokenizer, BertForSequenceClassification
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader)
import torch.optim as optim
from tqdm.auto import tqdm
#from .tools import *

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"

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

class Args():
    def __init__(self):
        #self.model_dir = f'./data/best_model.bin'
        self.model_dir = data_config.model_dir
        self.train_dir = data_config.train_dir
        self.test_dir = data_config.test_dir
        self.loss_dir = data_config.loss_dir

        self.epochs = 1
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
            emb_in_descs_total.append(emb_accumulate)


        for descs in indegree_summary:
            emb_accumulate = torch.zeros([768]).to(self.device)
            # print('emb_accumulate.shape ',emb_accumulate.shape)

            for i in range(0, len(descs), 20):
                descs_segment = descs[i:i+20]
                token_ids_descs_segment = self.bertTokenizer(descs_segment, return_tensors='pt', padding=True, truncation=True).to(self.device)
                emb_segment = self.bert(**(token_ids_descs_segment)).last_hidden_state
                emb_segment = emb_segment[:, 0, :]
                emb_segment = torch.sum(emb_segment, 0) / (len(indegree_description) * 4)
                emb_accumulate += emb_segment

            # print('emb_accumulate.shape ',emb_accumulate.shape)
            emb_in_summary_total.append(emb_accumulate)

        emb_in_descs_total = torch.stack(emb_in_descs_total, 0)
        emb_in_summary_total = torch.stack(emb_in_summary_total, 0)
        emb = torch.cat([emb_description, emb_summary, emb_in_descs_total, emb_in_summary_total, emb_pkg_emb], 1)
        emb = self.drop(emb)
        # 选取 [CLS] 特征
        # sent_rep = emb[:, 0, :]

        #上面定义：self.l1 = nn.Linear(768 * 5, 512)
        emb = self.l1(emb)

        # 全连接层分类
        logits = self.cls(emb)
        #self.cls = nn.Linear(512, args.num_labels)
        # logits = self.cls(sent_rep)

        # 返回分类 logits
        return logits


#训练过程
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

# 测试过程
def test(dataloader, model, loss_fn, args):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        progress_bar = tqdm(range(len(dataloader)))
        for src_name, description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary, label in dataloader:
            label = label.to(args.device)
            pred  = model(description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary, src_name).to(args.device)
            test_loss += loss_fn(pred, label).item()
            correct   += (pred.argmax(1) == label).type(torch.float).sum().item()
            progress_bar.update(1)
    test_loss /= size
    correct   /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.7f}%, Avg loss: {test_loss:>8f} \n")
    return correct

    # 测试过程
def predect(dataloader, model, loss_fn, args):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        progress_bar = tqdm(range(len(dataloader)))
        for src_name, description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary in dataloader:
            pred  = model(description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary, src_name).to(args.device)
            test_loss += loss_fn(pred, label).item()
            correct   += (pred.argmax(1) == label).type(torch.float).sum().item()
            progress_bar.update(1)
    test_loss /= size
    correct   /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.7f}%, Avg loss: {test_loss:>8f} \n")
    return correct


def transLabel(predicted, args):
    l1=torch.tensor([ False,  False, True]).to(args.device)
    l2=torch.tensor([ False,  True, False]).to(args.device)
    l3=torch.tensor([ True,   False, False]).to(args.device)
    if torch.equal(predicted , l1) :return 3
    if torch.equal(predicted , l2) :return 2
    if torch.equal(predicted , l3) :return 1
    return 0


def test_multi2(dataloader, model, loss_fn, args):
    size = len(dataloader.dataset)
    model.eval()
    predict_true_num = 0
    result_label_dict = {}
    with torch.no_grad():
        progress_bar = tqdm(range(len(dataloader)))
        for src_name, description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary in dataloader:
            pred = model(description, summary, indegree_description, indegree_summary, outdegree_description, outdegree_summary, src_name).to(args.device)
            predicted = torch.sigmoid(pred) > 0.5
            tuplelist = zip(src_name, predicted)
            for (n, l) in tuplelist:
                lstr = transLabel(l, args)
                result_label_dict[n] = lstr
            predict_true_num += predicted.sum().item()
            progress_bar.update(1)
    print('predict_true_num:', predict_true_num)
    return result_label_dict

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

def get_dataset(filename, args):
    #print("=========filename==============%s" %filename)
    f = open(filename, 'r')
    features = json.load(f)
    #print("=========len filename json=====%d" %len(features))
    return LayerDataset(features, args)

def get_package_vector():
    f =  open(data_config.pkg_vector)
    package_vector = json.load(f)
    return package_vector

def get_package_emb():
    
    f =  open(data_config.pkg_emb)
    package_emb = json.load(f)
    return package_emb


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


def do_layer(test_dir):
    args = Args()
    args.test_dir = test_dir
    test_dataset = get_dataset(args.test_dir, args)

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.dataloader_batch_size,
        num_workers=0,
        collate_fn=collate_fn,
    )

    args.test_dataloader = test_dataloader

    model = Bertcls(args).to(args.device)
    loss_fn = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    t = torch.load(args.model_dir)
    model.load_state_dict(t)
    return test_multi2(test_dataloader, model, loss_fn, args)



# 功能case：print(labels_to_one_hot([2, 3]))
# output：[0, 1, 1, 0]
def labels_to_one_hot(a):
    #print("in one_hot: %s" %a)
    b = [0, 0, 0]
    for i in a:
        b[i-1] = 1
    return b

