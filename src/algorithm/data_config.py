# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0
 
#self.model_dir = f'./data/best_model.bin'

tmp_json_file = "/var/fcfl/data/1228_bin_all_test_set1.json"
all_test_dir = f"/var/fcfl/data/1228_bin_all_test_set.json"
#Layerdata_path
model_dir = f'/var/fcfl/data/layer_best_model.bin'
train_dir = f'/var/fcfl/data/1228_bin_train_set.json'
test_dir = f'/var/fcfl/data/1228_bin_test_set.json'
loss_dir = f'/var/fcfl/data/1228_bin_loss_bert-no-cnn-all-new.json'

pkg_vector = f'/var/fcfl/data/data/package_vector.json'
pkg_emb = f'/var/fcfl/data/pkg_embedding_bidirect_padding.json'

#Model_path
model_path = "/var/fcfl/bert-base-uncased" 

#Classdata_path
train_csv_data = f'/var/fcfl/data/processed_datasource.csv'
train_dot_data = f'/var/fcfl/data/edges.csv'
class_data = f'/var/fcfl/data/20230330-lz.csv'

classer_save_path =  f'/var/fcfl/out/'
classer_model = classer_save_path + 'class_best_model_bert.h5'
tmp_class_file = "/var/fcfl/tmp_class_layer"
