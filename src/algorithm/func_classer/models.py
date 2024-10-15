import numpy as np
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()
import time
#from keras.losses import sparse_categorical_crossentropy
from keras.losses import categorical_crossentropy
from keras.models import Sequential,Model
from keras.optimizers import adam_v2
from keras.layers import Input,Dense,LSTM,Embedding,Conv1D,MaxPooling1D,GlobalMaxPooling1D,Layer
from keras.layers import Flatten,Dropout,Concatenate,Lambda,Multiply,Reshape,Dot,Bidirectional
import keras.backend as K
from tensorflow.keras.utils import to_categorical
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score
from transformers import TFAutoModel
from sklearn.metrics import multilabel_confusion_matrix
from sklearn.metrics import classification_report

#A Capsule Implement with Pure Keras
class Capsule(Layer):
    def __init__(self, num_capsule, dim_capsule, routings=3, share_weights=True, activation='squash', **kwargs):
        super(Capsule, self).__init__(**kwargs)
        self.num_capsule = num_capsule
        self.dim_capsule = dim_capsule
        self.routings = routings
        self.share_weights = share_weights
        if activation == 'squash':
            self.activation = self.squash
        else:
            self.activation = activations.get(activation)

    def build(self, input_shape):
        super(Capsule, self).build(input_shape)
        input_dim_capsule = input_shape[-1]
        if self.share_weights:
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(1, input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)
        else:
            input_num_capsule = input_shape[-2]
            self.W = self.add_weight(name='capsule_kernel',
                                     shape=(input_num_capsule,
                                            input_dim_capsule,
                                            self.num_capsule * self.dim_capsule),
                                     initializer='glorot_uniform',
                                     trainable=True)

    def call(self, u_vecs):
        if self.share_weights:
            u_hat_vecs = K.conv1d(u_vecs, self.W)
        else:
            u_hat_vecs = K.local_conv1d(u_vecs, self.W, [1], [1])
        
        #print(u_hat_vecs.shape)
        batch_size = K.shape(u_vecs)[0]
        #print("batch_size ....2 : " , batch_size)
        input_num_capsule = K.shape(u_vecs)[1]
        u_hat_vecs = K.reshape(u_hat_vecs, (batch_size, input_num_capsule,
                                            self.num_capsule, self.dim_capsule))
        u_hat_vecs = K.permute_dimensions(u_hat_vecs, (0, 2, 1, 3))
        #final u_hat_vecs.shape = [None, num_capsule, input_num_capsule, dim_capsule]

        b = K.zeros_like(u_hat_vecs[:,:,:,0]) #shape = [None, num_capsule, input_num_capsule]
        for i in range(self.routings):
            c = self.softmax(b, 1)
            # o = K.batch_dot(c, u_hat_vecs, [2, 2])
            o = tf.einsum('bin,binj->bij', c, u_hat_vecs)
            if K.backend() == 'theano':
                o = K.sum(o, axis=1)
            if i < self.routings - 1:
                o = K.l2_normalize(o, -1)
                # b = K.batch_dot(o, u_hat_vecs, [2, 3])
                b = tf.einsum('bij,binj->bin', o, u_hat_vecs)
                if K.backend() == 'theano':
                    b = K.sum(b, axis=1)

        return self.activation(o)

    def compute_output_shape(self, input_shape):
        return (None, self.num_capsule, self.dim_capsule)
    
    def squash(x, axis=-1):
        s_squared_norm = K.sum(K.square(x), axis, keepdims=True) + K.epsilon()
        scale = K.sqrt(s_squared_norm)/ (0.5 + s_squared_norm)
        return scale * x


    #define our own softmax function instead of K.softmax
    def softmax(x, axis=-1):
        ex = K.exp(x - K.max(x, axis=axis, keepdims=True))
        return ex/K.sum(ex, axis=axis, keepdims=True)
    
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
        for kernel_size in [3, 4, 5]:
            c = Conv1D(128, kernel_size, activation='relu')(input)
            c = GlobalMaxPooling1D()(c)
            #c = MaxPooling1D()(c)
            convs.append(c)
        x = Concatenate()(convs)
        
        output = Dense(self.class_num, activation=self.last_activation)(x)
        model = Model(inputs=input, outputs=output)

        return model
    
class BERT:
    def __init__(self, bert_path, max_length, num_classes, learning_rate=1e-5, use_textcnn=False, use_lstm_att=False, use_cap=False):
        self.num_classes = num_classes
        
        def custom_loss(y_true, y_pred):
            # Calculate categorical cross-entropy loss for the first 4 dimensions of y_pred
            #return sparse_categorical_crossentropy(y_true, y_pred)
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
#             bert_output = lstm_output[:, 0, :]
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
        elif use_cap:
            capsule = Capsule(num_capsule=10, dim_capsule=16, routings=5)(sequence_output)
            bert_output = Flatten()(capsule)
        else:
            bert_output = pooler_output
            
        pred_probs = Dense(num_classes, activation='softmax')(bert_output)  # n * num_classes
        
        output = Concatenate()([pred_probs, bert_output])
        self.model = Model(text_inputs, output)
        self.model.compile(loss=custom_loss, optimizer=adam_v2.Adam(learning_rate))
    

    def train_val(self, data_package, batch_size, epochs, save_best=True):
        
        X_input_ids_train, X_token_type_ids_train, X_attention_mask_train, y_train, X_input_ids_test, X_token_type_ids_test, X_attention_mask_test, y_test = data_package
        
        best_val_score = 0
        test_score = 0
        train_score_list = []
        val_socre_list = []
        """实验说明：
        每一轮train完，在val上测试，记录其accuracy，
        每当val-acc达到新高，则保存当前模型
        """

        for i in range(epochs):
            # record train set result:i
            #self.model.fit([X_input_ids_train, X_attention_mask_train, X_token_type_ids_train], y_train, batch_size=batch_size, epochs=2)
            self.model.fit([X_input_ids_train, X_attention_mask_train,X_token_type_ids_train], y_train, batch_size=batch_size, epochs=1)
            
            #pred_probs =  tf.expand_dims(tf.argmax(logits[0], axis=10, output_type=tf.int32), axis=3)#shape=(?, ?, ?, 1)
            pred_probs = self.model.predict([X_input_ids_train, X_attention_mask_train, X_token_type_ids_train], verbose=1)[:, :self.num_classes]
            y_pred = pred_probs
            for i in range(len(y_pred)):
                max_value=max(y_pred[i])
                for j in range(len(y_pred[i])):
                    if max_value == y_pred[i][j]:
                        y_pred[i][j]=1
                    else:
                        y_pred[i][j]=0
            print(classification_report(y_train, y_pred))

            #val_score = np.mean(ex_equal)
            val_score = precision_score(y_train, y_pred, average='micro')

            if val_score > best_val_score:
                best_val_score = val_score
                print('Current Best model!', 'current epoch:', i + 1, 'last best_val', best_val_score)
                if save_best:
                    self.model.save_weights('FUNC_best_model_bert.h5')
                    print('best model saved!')

    def predict(self, bert_inputs):
        self.model.load_weights('./FUNC_best_model_bert.h5')
        return self.model.predict(bert_inputs, verbose=1)
