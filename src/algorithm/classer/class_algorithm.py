# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import pandas as pd
import numpy as np
from transformers import BertTokenizer
# from global_variable import * 

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
from .models import BERT

class class_algorithm :

    def __init__(self):
        # parameters
        self.maxlen = 192   # BERT输入文本最大长度，超出截断，不足使用 [PAD] 补齐
        self.batch_size = 16 # batch 大小
        self.epochs = 10
        self.learning_rate = 1e-5 # 训练轮次与学习率经过实验，推荐使用这两个值

        # BERT模型，根据名称从 Huggingface 加载
        self.bert_path = data_config.model_path
        # 加载BERTTokenizer，与模型对应
        # print('Loading BERT tokenizer...')
        self.tokenizer = BertTokenizer.from_pretrained(self.bert_path)

    def resolve_dep(self,dot_dir):
        with open(dot_dir, 'r') as f:
            lines = f.readlines()
        deps = []
        for line in lines:
            _line = line.strip().split(' -> ')
            if (len(_line) == 2):
                deps.append(line.strip())
        return deps
    #列表，每项是一个带->的字串

    # getdataFrame
    def preprocess_edges(self,dot_dir):
        print("预处理依赖关系.......")
        dots_deps = self.resolve_dep(dot_dir)
        dots_deps = list(set(dots_deps))
        # dot 文件里的所有出现的包的集合
        pkg_names = list(set([dep.split(' -> ')[0].replace('"', '') for dep in dots_deps]))
        pkg_names += list(set([dep.split(' -> ')[1].replace('"', '')for dep in dots_deps]))
        filtered_deps = []
        for dep in dots_deps : #带->的字串列表
            dep = dep.split(' -> ')
            _dep = [dep[0].replace('"', ''), dep[1].replace('"', '')]
            filtered_deps.append(_dep)
        df_deps = pd.DataFrame(filtered_deps,columns=['out','in'])
        return  df_deps  #dataframe 'out' 'in'

    def build_edge_text_v2(self,name,df_edges):
        # { 0: '服务', 1: '工具', 2: '库', 3: '其它'}
        # 一阶邻居标签特征
        label_text_list_in = [["library", 0],["tool", 0], ["service", 0], ["other class", 0], ["unknown class", 0]]
        label_text_list_out = [["library", 0],["tool", 0], ["service", 0], ["other class", 0], ["unknown class", 0]]
        df_data =  pd.read_csv(data_config.class_data)
        
        edges = df_edges.values.tolist()
        for edge in edges:
            if edge[0] == name :
                if not df_data.loc[df_data['rpm_name'] == edge[1]].empty:
                    that_data = df_data.loc[df_data["rpm_name"] == edge[0]]
                    that_class_label = that_data["class_label"].values.tolist()[0]
                else :
                    that_class_label = 'unknown class'
                for tup in label_text_list_out:
                    if tup[0] == that_class_label:
                        tup[1] = tup[1] + 1

            elif edge[1] == name:
                if not df_data.loc[df_data['rpm_name'] == edge[0]].empty:
                    that_data = df_data.loc[df_data["rpm_name"] == edge[0]]
                    that_class_label = that_data["class_label"].values.tolist()[0]
                else :
                    that_class_label = 'unknown class'
                for tup in label_text_list_out:
                    if tup[0] == that_class_label:
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
        
        final_res = "depended by: {}, depending on: {}, similar: {}".format(in_res, out_res, "")
        return final_res

    def preprocess_descriptions_with_dep(self,pkgname,dot_df,csv_dir,feature):
        print("预处理描述信息.........") #只取描述信息
        csv_des = pd.read_csv(csv_dir)
        # 描述信息的整理
        # 用panda合并summary , description, zero_summary, zero_desc
        csv_des.loc[:,'text'] =  csv_des.fillna("").apply(lambda x: str(x['summary']) + str(x['description'])  
                                               + str(x['zero_summary']) + str(x['zero_description']), axis=1)
        csv_des['name'] = csv_des['rpm_name']
        csv_des['ori_text'] = csv_des['text'].apply(lambda x: x)
        #print(csv_des[['name','ori_text']])
        if feature == False : #False
            csv_des['text'] = csv_des.apply(lambda x: "[CLS]" + str(x["ori_text"]) + "[SEP]", axis=1)
        else :
            # 添加额外特征
            csv_des['text'] = csv_des.apply(lambda x: "[CLS]" + self.build_edge_text_v2(x['name'],dot_df)  + str(x["ori_text"]) + "[SEP]", axis=1)
        return csv_des[['name','text']]
        
    def tokenize(self,df):
        # 使用 huggingface 提供的 API 构造 BERT 需要的输入特征，包括 Token Embedding、Segment Embedding、Position Embedding、Attention Mask
        X_name = []
        X_input_ids = []
        X_token_type_ids = []
        X_attention_mask = []
        for content,name in zip(list(df['text']),list(df['name'])):
            bert_input = self.tokenizer.encode_plus(content,  #对content编码
                add_special_tokens = False, # add [CLS], [SEP]
                max_length = self.maxlen, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
                truncation=True)
            X_input_ids.append(bert_input['input_ids'])
            X_token_type_ids.append(bert_input['token_type_ids'])
            X_attention_mask.append(bert_input['attention_mask'])
            X_name.append(name)
        X_input_ids = np.array(X_input_ids)
        X_token_type_ids = np.array(X_token_type_ids)
        X_attention_mask = np.array(X_attention_mask)

        return X_input_ids, X_token_type_ids, X_attention_mask,  X_name #返回列表

    def pkg_classfication(self,df_data, model_file) :
        # 类别
        num_label_map = { 0: '服务', 1: '工具', 2: '库'}
        num_classes = len(num_label_map)
        # key: pkg_name, value: predicted label
        name_pred_map = {}
        print('转为 tokenize')
        X_input_ids_test, X_token_type_ids_test, X_attention_mask_test, name_test = self.tokenize(df_data)
        # 构建输入特征
        print('构建输入特征')
        model_input = [X_input_ids_test, X_attention_mask_test, X_token_type_ids_test]
        print('调用模型..........')
        model = BERT(self.bert_path, self.maxlen, num_classes, self.learning_rate)
        print("调用模型预测.........")
        output = model.predict(model_input, model_file)[:,:num_classes]
    
        # 处理预测结果
        for i, name in enumerate(name_test):
            # L_label = np.array(range(num_classes))
            model_input = [np.array([X_input_ids_test[i]]), np.array([X_attention_mask_test[i]]), np.array([X_token_type_ids_test[i]])]
            pred_probs = output[i]
            name_pred_map[name] = num_label_map[np.argmax(pred_probs)]
        # 返回 分类结果
        return name_pred_map

    def getALLPkgclassfication(self,dot_file,csv_files,feature,model_file):
        df_des = self.preprocess_descriptions_with_dep('all',dot_file,csv_files,feature)
        # 返回了df，只取得df两列name, text,其中test包含cls信息
        # df_edges = preprocess_edges(df_des,dot_files)
        class_dict = self.pkg_classfication(df_des, model_file)
        df_des.loc[:,'pred'] = df_des['name'].apply(lambda x: class_dict[x])
        # 输出 csv 文件
        df_print = df_des[['name', 'pred']]
        df_print.to_csv("{}allpkgsclass.csv".format('./'), index=False)
        return  class_dict

    def getSinglePkgclassfication(self,dot_file,csv_files,pkg_name,feature):

        df_des = self.preprocess_descriptions_with_dep(pkg_name,dot_file,csv_files,feature)
        # df_edges = preprocess_edges(df_des,dot_files)
        class_dict = self.pkg_classfication(df_des)
        df_des.loc[:,'pred'] = df_des['name'].apply(lambda x: class_dict[x])
        # 输出 csv 文件
        df_print = df_des[['name', 'pred']]
        df_pkg = df_print.loc[lambda df_print:df_des['name'] == pkg_name]
        df_pkg.to_csv("{}{}_class.csv".format('./').format(pkg_name), index=False)

        return  class_dict

# dot_file = '/var/fcfl/dot_files/rpm_all.dot'
# csv_file = '/var/fcfl/csv/rpm_all.csv'
# c_al = class_algorithm()
# c_al.getALLPkgclassfication(dot_file,csv_file,False)
