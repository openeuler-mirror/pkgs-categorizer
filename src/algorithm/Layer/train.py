from datetime import datetime
from tqdm.auto import tqdm
import torch
from transformers import BertConfig
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import json
import model
from model import Args, BertLayerizer

import sys
import os
# 获取当前脚本所在的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
# 计算上两级目录的路径
parent_dir = os.path.dirname(os.path.dirname( os.path.abspath(__file__) ))
# 添加上两级目录到系统路径中
sys.path.append(parent_dir)
# 导入需要调用的包或模块
#import data_config
#from config import fcfl_config


class LayerDataset(Dataset):
    def __init__(self, features):
        self._data = features

    def __len__(self):
        return len(self._data)
    
    def __getitem__(self, index):
        content = None
        pkg_name = self._data[index]['src_name']
        description = self._data[index]['description_nltk']
        summary = self._data[index]['summary_nltk']

        indegree_description = self._data[index]['indegree_description_nltk']
        indegree_summary = self._data[index]['indegree_summary_nltk']
        indegree = '\n'.join(indegree_summary)

        outdegree_description = self._data[index]['outdegree_description_nltk']
        outdegree_summary = self._data[index]['outdegree_summary_nltk']
        outdegree = '\n'.join(outdegree_summary)

        #label = self._config.label_index('-'.join([f'{l}' for l in self._data[index]['label'] ]))
        label = self.labels_to_one_hot(self._data[index]['label'])

        content = "name: {}\ndescription: {}\nsummary: {}\nindegree: {}\n{}\noutdegree:{}\n{}"\
            .format(pkg_name, description, summary, indegree_description, indegree, 
                    outdegree_description, outdegree)

        return content, label
    
    # 功能case：print(labels_to_one_hot([2, 3]))
    # output：[0, 1, 1, 0]
    def labels_to_one_hot(self,a):
        b = [0, 0, 0]
        for i in a:
            b[i - 1] += 1
        return b
    
class Layer():

    def get_dataset(self,filename):
        f = open(filename, 'r')
        features = json.load(f)
        return LayerDataset(features)


    def collate_fn(self,data):
        contents, labels = [], []

        for content, label in data:
            contents.append(content)
            labels.append(label)

        return contents,  torch.tensor(labels)

    # 训练过程
    def train_model(self,dataloader, model, loss_fn, optimizer, args):
        model.train()

        progress_bar = tqdm(range(len(dataloader)))
        for content, label in dataloader:

            label = label.to(args._device)

            # 计算损失
            # pre_m = torch.cuda.memory_summary()
            pred = model(content).to(args._device)
            # after_m = torch.cuda.memory_summary()
            # print(pre_m, after_m)
            # print('pred:', pred, pred.size())
            # print('label:', label, label.size())
            loss = loss_fn(pred, label.float())

            # 反向传播
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            progress_bar.update(1)
            args._loss_record.append(loss.item())

            if progress_bar.n % 300 == 0:
                acc = self.test(args._test_dataloader, model, loss_fn, args)
                if acc > args._best_accuracy:
                    args._best_accuracy = acc
                    model_to_save = model.module if hasattr(model, 'module') else model
                    torch.save(model_to_save.state_dict(), args._best_model_file)
                # torch.cuda.empty_cache()

    # 多标签的测试过程
    def test(self,dataloader, model, loss_fn, config):
        size = len(dataloader.dataset)
        model.eval()
        test_loss, correct, tp, true_num, predict_true_num = 0, 0, 0, 0, 0
        with torch.no_grad():
            for content, label in dataloader:
                label = label.to(config._device)
                pred = model(content).to(config._device)
                test_loss += loss_fn(pred, label.float()).item()

                predicted = torch.sigmoid(pred) > 0.5

                correct += predicted.eq(label > 0.5).sum().item()
                tp += predicted.bitwise_and(label > 0.5).sum().item()
                true_num += (label > 0.5).sum().item()
                predict_true_num += predicted.sum().item()
        test_loss /= size
        # print('correct_num:', correct)
        correct /= (size * config._num_labels)
        recall = tp / true_num
        precision = tp / predict_true_num
        f1 = precision * recall * 2 / (recall + precision)
        print(f"Test Error: \n Accuracy: {(100 * correct):>0.7f}%, Precision: {(100 * precision):>0.7f}%, F1: {(100 * f1):>0.7f}%, Recall: {(100 * recall):>0.7f}%, Avg loss: {test_loss:>8f} \n")
    
        return correct

    # 校验模型的相关数据
    def test_model(self,model_path, test_file_path):
        test_dataset = self.get_dataset(test_file_path)
        dataloader_batch_size = 2

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=dataloader_batch_size,
            # shuffle=True,
            num_workers=0,
            collate_fn= self.collate_fn,
        )

        config = BertConfig.from_pretrained(model_path)
        args = Args(model_path)
        args._test_dataloader = test_dataloader
        model = BertLayerizer.from_pretrained(model_path, ignore_mismatched_sizes=True, 
                                            config=config, args=args)
        model.to(args._device)
        loss_fn = nn.BCEWithLogitsLoss()
        
        model.load_state_dict(torch.load(args._best_model_file))
        self.test(test_dataloader, model, loss_fn, args)
            
    def train(self,model_path, train_file_path, test_file_path, loss_file_path):
        train_dataset = self.get_dataset(train_file_path)
        test_dataset = self.get_dataset(test_file_path)
        dataloader_batch_size = 2

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=dataloader_batch_size,
            # shuffle=True,
            num_workers=0,
            collate_fn= self.collate_fn,
        )

        test_dataloader = DataLoader(
            test_dataset,
            batch_size=dataloader_batch_size,
            # shuffle=True,
            num_workers=0,
            collate_fn= self.collate_fn,
        )

        config = BertConfig.from_pretrained(model_path)
        args = Args(model_path)
        args._test_dataloader = test_dataloader
        model = BertLayerizer.from_pretrained(model_path, ignore_mismatched_sizes=True, 
                                            config=config, args=args)
        model.to(args._device)
        loss_fn = nn.BCEWithLogitsLoss()
        optimizer = optim.Adam(model.parameters(), lr=args._lr)

        for t in range(args._epochs):
            print(f"Epoch {t + 1}   ", datetime.now())
            print("------------------------------------------------------------------")
            self.train_model(train_dataloader, model, loss_fn, optimizer, args)
            self.test(test_dataloader, model, loss_fn, args)

        with open(loss_file_path, 'w') as fw:
            json.dump(args._loss_record, fw)

        print("Done!")


if __name__ == "__main__":
    #    config = fcfl_config()
    model_path = "/home/huan/Downloads/fcfl_data/fcfl/bert-base-uncased"
    #model_path = "/home/huan/workspce/data/bert-base-uncased"
    #model_path = '/var/fcfl/bert-base-uncased'
    train_dir = 'u_train_set.json'
    test_dir = 'u_test_set.json'
    loss_dir = './new_loss.json'
    # 训练模型
    layer = Layer()
    layer.train(model_path,train_dir, test_dir,loss_dir)
    layer.test_model(model_path, test_dir)
