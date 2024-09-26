# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import time
import tensorflow as tf
import time
from keras.losses import categorical_crossentropy
from keras.models import Sequential,Model
from tensorflow.keras.optimizers import Adam
from keras.layers import Input,Dense,LSTM,Embedding,Conv1D,MaxPooling1D,GlobalMaxPooling1D
from keras.layers import Flatten,Dropout,Concatenate,Lambda,Multiply,Reshape,Dot,Bidirectional
import keras.backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from transformers import TFAutoModel

    
class TextCNN(object):
    def __init__(self, maxlen,
                 num_features,
                 class_num=1,
                 last_activation='relu'):
        self.maxlen = maxlen
        self.num_features = num_features
        self.class_num = class_num
        self.last_activation = last_activation

    def get_model(self):
        input = Input((self.maxlen, self.num_features,))

        # Embedding part can try multichannel as same as origin paper
        # embedding = Embedding(self.max_features, self.embedding_dims, input_length=self.maxlen)(input)
        convs = []
        for kernel_size in [2, 3, 4]:
            c = Conv1D(128, kernel_size, activation='relu')(input)
            c = GlobalMaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)

        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)

        return model
    
class BERT:
    def __init__(self, bert_path, max_length, num_classes, learning_rate=1e-5, use_textcnn=False, use_lstm_att=False):
        self.num_classes = num_classes
        
        def custom_loss(y_true, y_pred):
            # Calculate categorical cross-entropy loss for the first 4 dimensions of y_pred
            return categorical_crossentropy(y_true, y_pred[:, :self.num_classes])

        # 模型定义
        # text_encoder:
        input_shape = (max_length, )
        input_ids = Input(input_shape, dtype=tf.int32)
        attention_mask = Input(input_shape, dtype=tf.int32)
        token_type_ids = Input(input_shape, dtype=tf.int32)
        
        text_inputs = [input_ids, attention_mask, token_type_ids] # 构造 bert 输入
        bert = TFAutoModel.from_pretrained(bert_path, from_pt=True, output_hidden_states=True)
        bert_out = bert(text_inputs)
        sequence_output, pooler_output, hidden_states = bert_out[0], bert_out[1], bert_out[2] 
        
        if use_textcnn and (not use_lstm_att):
            # 将每一层的第一个token(extra_feats，拼在一起当作textcnn的输入
            cls_embeddings = tf.expand_dims(hidden_states[1][:, 0, :], axis=1) # [bs, 1, hidden]
            for i in range(2, 9):
                cls_embeddings = Concatenate(axis=1)([cls_embeddings, tf.expand_dims(hidden_states[i][:, 0, :], axis=1)])
            textcnn = TextCNN(cls_embeddings.shape[1], num_features=cls_embeddings.shape[2], class_num=256).get_model()
            bert_output = textcnn(cls_embeddings)  
        elif use_lstm_att and (not use_textcnn):
            attention_weights = Dense(1, activation='tanh', use_bias=False, name='attention')(sequence_output)
            attention_weights = tf.nn.softmax(attention_weights, axis=1)
            bert_output = tf.reduce_sum(attention_weights * sequence_output, axis=1)
            # bert_output = lstm_output[:, 0, :]
        elif use_textcnn and use_lstm_att:
            cls_embeddings = tf.expand_dims(hidden_states[1][:, 0, :], axis=1) # [bs, 1, hidden]
            for i in range(2, 13):
                cls_embeddings = Concatenate(axis=1)([cls_embeddings, tf.expand_dims(hidden_states[i][:, 0, :], axis=1)])
            textcnn = TextCNN(cls_embeddings.shape[1], num_features=cls_embeddings.shape[2], class_num=256).get_model()
            bert_output1 = textcnn(cls_embeddings)
            lstm_output = Bidirectional(LSTM(512, return_sequences=True), name='bi_lstm')(sequence_output)
            lstm_output = Dropout(0.8)(lstm_output)
            bert_output2 = lstm_output[:, 0, :]
            bert_output = Concatenate()([bert_output1, bert_output2])
        else:
            bert_output = pooler_output
            
        
        pred_probs = Dense(num_classes, activation='softmax')(bert_output)  # n * num_classes
        
        # prob + embedding
        output = Concatenate()([pred_probs, bert_output])
        
        self.model = Model(text_inputs, output)
        self.model.compile(loss=custom_loss, optimizer=Adam(learning_rate))
    def train_val(self, data_package, batch_size, epochs, save_best=True,):
        dicta = {}
        X_input_ids_train, X_token_type_ids_train, X_attention_mask_train, y_train1, name_train, X_input_ids_test, X_token_type_ids_test, X_attention_mask_test, y_test, name_test = data_package
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，则保存当前模型
        """
        #print('in train_val: len(y_train)=%d,  len(y_test)=%d batch_size=%d, epochs=%d' % (len(y_train1), len(y_test), batch_size, epochs))

        for i in range(2):
            # print("====================i=%d, now fit X_input_idxs=%s" %(i, X_input_ids_train))
            self.model.fit([X_input_ids_train, X_attention_mask_train, X_token_type_ids_train], to_categorical(y_train1), batch_size=batch_size, epochs=epochs)  #训练
            # record train set result:
            pred_probs = self.model.predict([X_input_ids_train, X_attention_mask_train, X_token_type_ids_train], verbose=1)[:, :self.num_classes]
            # train 和 true_y 比较
            predictions = np.argmax(pred_probs, axis=1)
            for i,name in enumerate(name_train):
                dicta[name]={"pred":predictions[i],"y_train":y_train1[i]}
            train_score = round(accuracy_score(y_train1, predictions),5)
            train_score_list.append(train_score)
            # validation:
            # pred_probs = self.model.predict([X_input_ids_test, X_attention_mask_test, X_token_type_ids_test], verbose=1)[:, :self.num_classes]
            # print("====dddd====len val result =%d"%(len(pred_probs) ))
            predictions = np.argmax(pred_probs, axis=1)
            for i,name in enumerate(name_test):
                dicta[name_test[i]]={"pred":predictions[i],"y_val":y_test[i]}
            val_score = round(accuracy_score(y_test, predictions),5)
            val_socre_list.append(val_score)
            t2 = time.time()
            print('Epoch', i + 1, ' | train acc:', train_score, ' | val acc:',val_score)
            # save best model according to validation & test result:
            if val_score > best_val_score:
                best_val_score = val_score
                print('Current Best model!', 'current epoch:', i + 1, 'last best_val', best_val_score)
                if save_best:
                    self.model.save_weights('class_best_model_bert.h5')
                    print('best model saved!')
            # print("++++++++++++++++++++++")
            dict_print = []
            for i in dicta:
                x= {"name":i}
                x.update(dicta[i])
                dict_print.append(x)
            # print(dicta)
            print('----------------------')
            # print(dict_print)
            print('======================')
        return train_score_list, val_socre_list, best_val_score, test_score

    def predict(self, bert_inputs, h5_file):
        # self.model.load_weights('./class_best_model_bert.h5')
        self.model.load_weights(h5_file)
        return self.model.predict(bert_inputs, verbose=1)
