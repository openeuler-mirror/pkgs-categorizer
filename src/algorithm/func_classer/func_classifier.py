# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
from transformers import AutoTokenizer
import numpy as np
from models import *

class FuncClassifier():
    def __init__(self, num_label_map, ):
        self.maxlen        = 192
        self.batch_size    = 8
        self.epochs        = 10
        self.learning_rate = 1e-6
        self.bert_path     = "/var/fcfl/bert-base-uncased"
        self.tokenizer     = AutoTokenizer.from_pretrained(self.bert_path)
        self.num_label_map = num_label_map
        self.num_classes   = len(num_label_map) 
        self.model         = BERT(self.bert_path, self.maxlen,
                                  self.num_classes, self.learning_rate)

    def calcClass():
        print("error virtual function")
        pass

    def tokenize(self, df):
        # 使用 huggingface 提供的 API 构造 BERT 需要的输入特征，
        # 包括 Token Embedding、Segment Embedding、Position Embedding、Attention Mask
        X_name = []
        X_input_ids = []
        X_token_type_ids = []
        X_attention_mask = []
        for content,name in zip(list(df['text']),list(df['rpm_name'])):
            bert_input = self.tokenizer.encode_plus(content,
                add_special_tokens = False, # add [CLS], [SEP]
                max_length = self.maxlen, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
                truncation = True)
            X_input_ids.append(bert_input['input_ids'])
            X_token_type_ids.append(bert_input['token_type_ids'])
            X_attention_mask.append(bert_input['attention_mask'])
            X_name.append(name)
        X_input_ids = np.array(X_input_ids)
        X_token_type_ids = np.array(X_token_type_ids)
        X_attention_mask = np.array(X_attention_mask)
        return X_input_ids, X_token_type_ids, X_attention_mask,  X_name

class subCalssifer(FuncClassifier):
    def __init__(self, typeName):
        self.typeName = typeName
        num_label_map = { 0: 'true', 1: 'false'}
        super().__init__(num_label_map)
        pass

    def calcClass(self, filePath, h5_file):
        print("读csv文件")
        df = pd.read_csv(filePath)
        df2 = df.fillna(" ")
        df.loc[:,'text'] = df2.apply(lambda x: str(x['summary']) + str(x['description'])  
                                                    + str(x['zero_summary']) + str(x['zero_description']), axis=1)
        
        print("读取包描述信息")
        x_input_ids, x_token_type, x_attention_mask, x_name = self.tokenize(df)
        model_input = [x_input_ids, x_token_type, x_attention_mask]

        print("调用模型")
        self.model = BERT(self.bert_path, self.maxlen,
                          self.num_classes, self.learning_rate)

        print("预测")
        name_pred_map = {}
        output = self.model.predict(model_input, h5_file)[:,:self.num_classes]
        df[self.typeName] = ''

        for i, name in enumerate(x_name):
            model_input = [np.array([x_input_ids[i]]), np.array([x_attention_mask[i]]), np.array([x_token_type[i]])]
            pred_probs = output[i]
            name_pred_map[name] = self.num_label_map[np.argmax(pred_probs)] #取最大值的索引
            df.loc[df[df['rpm_name']==name].index, self.typeName] = self.num_label_map[np.argmax(pred_probs)]

        return df



#tst = subCalssifer("media")
#tst.calcClass("./1228_bin.csv", "gene_best_model_bert.h5")
