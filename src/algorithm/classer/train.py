from transformers import AutoTokenizer
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd
import time
from models import *
from utils import *
#from preprocess import *
import argparse

# 解析命令行参数
parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, default="all")
parser.add_argument('--use_graph_feat', type=str, default="false")
args = parser.parse_args()

# parameters
maxlen = 192   # BERT输入文本最大长度，超出截断，不足使用 [PAD] 补齐
batch_size = 8 # batch 大小
epochs = 10
learning_rate = 1e-5 # 训练轮次与学习率经过实验，推荐使用这两个值

# 结果输出目录
#save_path = "./output"
# BERT模型，根据名称从 Huggingface 加载
#bert_path = "bert-base-uncased"    
save_path = f'u0412.csv'
# BERT模型，根据名称从 Huggingface 加载
bert_path = "/home/huan/workspce/data/bert-base-uncased"
#bert_path = "/var/fcfl/bert-base-uncased"
# 加载BERTTokenizer，与模型对应
tokenizer = AutoTokenizer.from_pretrained(bert_path)

# 节点数据预处理，将 summary 和 description 拼合，并将标签分为分类标签、分层标签、分层分类标签
# 边数据预处理，将 dot 文件表示的依赖关系转化为 csv 表示
#preprocess_data(save_path)

# 数据读取
print("开始读取数据")
df_data = pd.read_csv(save_path)

mode = args.mode
label_str = "classes"

# # 将数据分为 10 组，采用 10 折交叉验证
groups = split_dataset_by_label_ratio(df_data, label_str)
df_data["text"] = df_data.apply(lambda x: x["summary"] + " " + x["description"], axis=1)
df_data["group_num"] = df_data["name"].apply(lambda x: get_group_num(x, groups))

# 在训练过程中用于将 index 与 name 和 label 进行对应的字典
idx_name_map, name_idx_map = get_idx_name_map(df_data["name"])
num_label_map, label_num_map = get_num_label_map(df_data[label_str])
num_classes = len(num_label_map)

#print(list(df_data[label_str]))
onehot_labels = encode_onehot(label_num_map, list(df_data[label_str]))
df_data["ori_text"] = df_data["text"].apply(lambda x: x)

# key: pkg_name, value: predicted label
name_pred_map = {}

for i in range(10):
    print("正在运行第 {} / 10 组训练集&测试集".format(i + 1))
    df_data["is_train"] = df_data["group_num"].apply(lambda x: 0 if x == i else (2 if x == (i+1)%10 else 1))
    print(len(df_data))
    if args.use_graph_feat == "true":
         print("正在构造额外特征，此步骤无GPU加速，预计耗时三分钟")
         df_data["text"] = df_data.apply(lambda x: "[CLS]" + build_edge_text_v2(x["name"], df_data, df_edges) + "[SEP]" + x["ori_text"] + "[SEP]", axis=1)
    else:
        df_data["text"] = df_data.apply(lambda x: "[CLS]" + x["ori_text"] + "[SEP]", axis=1)
   
    df_train = df_data.loc[df_data["is_train"] == 1]
    df_val = df_data.loc[df_data["is_train"] == 2]
    df_test = df_data.loc[df_data["is_train"] == 0]
    
    def tokenize(df):
        # 使用 huggingface 提供的 API 构造 BERT 需要的输入特征，包括 Token Embedding、Segment Embedding、Position Embedding、Attention Mask
        t = time.time()
        X_name = []
        X_input_ids = []
        X_token_type_ids = []
        X_attention_mask = []
        y = []
        for content,label,name in zip(list(df.text),list(df[label_str]),list(df.name)):
            bert_input = tokenizer.encode_plus(content, 
                add_special_tokens = False, # add [CLS], [SEP]
                max_length = maxlen, # max length of the text that can go to BERT
                pad_to_max_length = True, # add [PAD] tokens
                return_attention_mask = True, # add attention mask to not focus on pad tokens
                truncation=True)
            X_input_ids.append(bert_input['input_ids'])
            X_token_type_ids.append(bert_input['token_type_ids'])
            X_attention_mask.append(bert_input['attention_mask'])
            y.append(label_num_map[label])
            X_name.append(name)
        X_input_ids = np.array(X_input_ids)
        X_token_type_ids = np.array(X_token_type_ids)
        X_attention_mask = np.array(X_attention_mask)
        y = np.array(y)
        # print('tokenizing time cost:',time.time()-t,'s.')

        return X_input_ids, X_token_type_ids, X_attention_mask, y, X_name

    # ========== model traing: ==========
    # 构造训练、验证、测试集 BERT 需要的输入特征
    X_input_ids_train, X_token_type_ids_train, X_attention_mask_train, y_train, name_train = tokenize(df_train)
    X_input_ids_test, X_token_type_ids_test, X_attention_mask_test, y_test, name_test = tokenize(df_test)
    X_input_ids_val, X_token_type_ids_val, X_attention_mask_val, y_val, name_val = tokenize(df_val)


    data_package = [X_input_ids_train, X_token_type_ids_train, X_attention_mask_train, y_train, X_input_ids_val, X_token_type_ids_val, X_attention_mask_val, y_val]
    
    # 调用模型并训练
    model = BERT(bert_path, maxlen, num_classes, learning_rate)
    train_score_list, val_socre_list, best_val_score, test_score = model.train_val(data_package, batch_size, epochs, save_best=True)

    # print('test acc:', str(test_score))
    # print('best val acc:', str(best_val_score))
    # print('train acc list:\n', str(train_score_list))
    # print('val acc list:\n', str(val_socre_list), '\n')

print("train done!")
