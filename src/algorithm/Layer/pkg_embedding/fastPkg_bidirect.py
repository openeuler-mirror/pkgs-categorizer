# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

from set import package
from set import node
from set import node_set
from set import load_data

import csv
import json
import torch
from torch.autograd._functions import tensor
from torch.utils.data import (Dataset, DataLoader)
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

##配置信息
class Args():
    def __init__(self):
        self.epochs = 500
        self.lr = 0.0001
        self.dataloader_batch_size = 4

        self.vocab_size = 6533   # 词表size，这里和包的数量一致
        self.emb_size = 512      # embedding维度，这里取512

        self.device = "cuda" if torch.cuda.is_available() else "cpu"    # 计算设备
        print('计算设备为：',self.device)




class fastPkg(nn.Module):
    def __init__(self, args):
        super(fastPkg, self).__init__()

        self.dropout = nn.Dropout(p=0.2)
        #Hidden layer
        self.hidden = nn.Linear(args.vocab_size * 2, args.emb_size)

        self.h2 = nn.Linear(args.emb_size, args.emb_size)
        self.h3 = nn.Linear(args.emb_size, args.emb_size)
        #Output layer
        self.output = nn.Linear(args.emb_size, args.vocab_size)


    def forward(self, total_out, total_in):

        x = torch.cat([total_out, total_in], 1)

        h = self.hidden(self.dropout(x))    #x: (batch_size, vocab_size), h:(batch_size, emb_size)

        h = self.h2(h)

        h = self.h3(h)

        o = F.softmax(self.output(h), dim=1)#o: (batch_size, vocab_size)
        return o, h



class outnodeDataset(Dataset):
    def __init__(self, features, args):
        self.data = features
        self.args = args
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):

        total_out = self.data[index][0]
        total_in = self.data[index][1]
        label = self.data[index][2]
        return  total_out, total_in, label

#训练过程
def train(dataloader, model, loss_fn, optimizer, config):
    size = len(dataloader.dataset)
    model.train()
    for batch, (total_out, total_in, label) in enumerate(dataloader):
        total_out, total_in, label = total_out.to(config.device), total_in.to(config.device), label.to(config.device)

        # 计算损失
        pred, h = model(total_out, total_in)
        loss = loss_fn(pred, label)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 1000 == 0:
            loss, current = loss.item(), batch * len(total_out)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")


#测试过程
def test(dataloader, model, loss_fn, config):
    size = len(dataloader.dataset)
    model.eval()
    test_loss, correct = 0, 0
    with torch.no_grad():
        for t, depend_list, label in dataloader:
            t, depend_list, label = t.to(config.device), depend_list.to(config.device), label.to(config.device)
            pred, h = model(t, depend_list)
            test_loss += loss_fn(pred, label).item()
            # print('pred : ', pred)
            # print('y : ', y)
            # print('pred.argmax(1) : ', pred.argmax(1))
            correct += (pred.argmax(1) == label).type(torch.float).sum().item()
    test_loss /= size
    correct /= size
    print(f"Test Error: \n Accuracy: {(100*correct):>0.6f}%, Avg loss: {test_loss:>8f}\n")
    return correct


def main():

    data, other_info = step1()
    args = Args()

    train_dataset = outnodeDataset(data, args)
    test_dataset = outnodeDataset(data, args)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.dataloader_batch_size,
        num_workers=0,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.dataloader_batch_size,
        num_workers=0,
    )

    args.train_dataloader = train_dataloader
    args.test_dataloader = test_dataloader

    model = fastPkg(args).to(args.device)
    #model.load_state_dict(torch.load(f'./input/fastPkg2.bin'))
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    max_acc = test(test_dataloader, model, loss_fn, args)
    for t in range(args.epochs):
        print(f"Epoch {t+1}\n-------------------------------")
        train(train_dataloader, model, loss_fn, optimizer, args)
        acc = test(test_dataloader, model, loss_fn, args)
        if acc > max_acc:
            max_acc = acc
            model_to_save = model.module if hasattr(model, 'module') else model
            torch.save(model_to_save.state_dict(), f'./input/fastPkg2.bin')
    
    print("Done!")





# 构建用于one-hot 的 词典、统计最大出度
def step1():
    print('从文件中读入依赖关系对list')
    dependency_relationship = load_data('./input/1228/rpm_all.dot')

    s = node_set()
    print('将依赖关系对list添加到点集中')
    s.add_pairs(dependency_relationship)

    pkg_to_index_dict = {}
    index_to_pkg_dict = {}
    max_outdegree = 0
    for node in s.nodes:
        pkg_to_index_dict[node.packages[0]] = len(pkg_to_index_dict)
        index_to_pkg_dict[pkg_to_index_dict[node.packages[0]]] = node.packages[0]

        if len(node.out_node) > max_outdegree:
            max_outdegree = len(node.out_node)
    print('最大的出度：', max_outdegree)

    print('包的数量: ', len(pkg_to_index_dict))

    

    pkg_to_out_onehot = {}
    pkg_to_onehot = {}
    data = []
    for node in s.nodes:
        pkg = node.packages[0]
        total_out = torch.zeros((6533))
        total_in = torch.zeros((6533))
        l = torch.zeros((6533))
        l[pkg_to_index_dict[pkg]] += 1
        for out_pkg_node in node.out_node:
            out_pkg = out_pkg_node.packages[0]
            total_out[pkg_to_index_dict[out_pkg]] += 1
            d = torch.zeros((6533))
            d[pkg_to_index_dict[out_pkg]] += 1
        
        for in_pkg_node in node.in_node:
            in_pkg = in_pkg_node.packages[0]
            total_in[pkg_to_index_dict[in_pkg]] += 1

        pkg_to_out_onehot[pkg] = total_out#.numpy().tolist()
        pkg_to_onehot[pkg] = l#.numpy().tolist()
        #data.append([t, pkg_to_index_dict[pkg]])

        data.append([total_out, total_in, pkg_to_index_dict[pkg]])
        

    return data, {
            'pkg_to_index_dict' : pkg_to_index_dict,
            'index_to_pkg_dict' : index_to_pkg_dict,
            'pkg_to_out_onehot' : pkg_to_out_onehot,
            'pkg_to_onehot' : pkg_to_onehot,
        }


def generate_embedding():
    data, other_info = step1()
    args = Args()

    train_dataset = outnodeDataset(data, args)
    test_dataset = outnodeDataset(data, args)

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=args.dataloader_batch_size,
        num_workers=0,
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.dataloader_batch_size,
        num_workers=0,
    )

    args.train_dataloader = train_dataloader
    args.test_dataloader = test_dataloader

    model = fastPkg(args)#.to(args.device)
    model.load_state_dict(torch.load(f'./input/fastPkg2.bin'))

    pkg_embedding = {}
    index_to_pkg_dict = other_info['index_to_pkg_dict']
    model.eval()
    i = 0
    for total_out, total_in, y in train_dataset:
        pred, h = model(torch.unsqueeze(total_out, 0), torch.unsqueeze(total_in, 0))
        pkg_embedding[index_to_pkg_dict[y]] = h[0].detach().numpy().tolist()
        i += 1
        if i % 100 == 0:
            print(i)
    
    with open('./input/pkg_embedding_bidirect.json', 'w') as f:
        json.dump(pkg_embedding, f)


def padding_package_emb():
    dic = {"内核": "" ,"内核服务": "" ,
        "核心工具": 1 ,"核心库": 1 ,"核心服务": 1 ,"基础环境": 1 ,
        "系统工具": 2 ,"系统服务": 2 ,"系统库": 2 ,"系统应用":2,
        "虚拟化": 3 ,"应用库": 3 ,"应用工具":3,"应用服务":3,"字体":3,
        "数据库":4,"云计算":4,"大数据":4,"桌面":4,"云原生":4,
        "编程语言":""}

    features = []

    # 读取文件
    with open(f'./input/1228/1228_bin.csv', 'r', encoding='utf-8') as f:
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

            features.append(feature)

    package_emb = {}
    with open('./input/pkg_embedding_bidirect.json', 'r') as f:
        package_emb = json.load(f)

    for feature in features:
        if feature['src_name'] not in package_emb:
            package_emb[feature['src_name']] = [0 for i in range(512)]
    
    with open('./input/pkg_embedding_bidirect_padding.json', 'w') as f:
        json.dump(package_emb, f)



if __name__ == "__main__":
    # 训练模型
    main()
    # 使用训练好的模型生成embedding
    generate_embedding()
    # 用0向量补充没有依赖关系的包
    padding_package_emb()
    print('Done')