# -*-coding:utf-8-*-
import os
import torch
from transformers import BertModel, BertTokenizer, BertPreTrainedModel
import torch.nn as nn
from config import fcfl_config

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:128"


class Args:
    def __init__(self,
                 model_dir, 
                 epochs=50, 
                 lr=0.000001, 
                 num_labels=4, 
                 dataloader_batch_size=1, 
                 dropout_rate=0.1, 
                 best_accuracy=0.0, 
                 hidden_size=768, 
                 kernel_sizes=[3, 4, 5], 
                 num_filters=128, 
                 loss_record = [], 
                 best_model_file=None 
                 ):
        self._model_dir = model_dir
        self._epochs = epochs
        self._lr = lr
        self._num_labels = num_labels
        self._dataloader_batch_size = dataloader_batch_size
        self._dropout_rate = dropout_rate
        self._device="cuda" if torch.cuda.is_available() else "cpu"  # 计算设备
        self._best_accuracy = best_accuracy
        self._hidden_size = hidden_size
        self._kernel_sizes = kernel_sizes
        self._num_filters = num_filters
        self._loss_record = loss_record
        self._test_dataloader = None
        if best_model_file is None:
            config = fcfl_config()
            layerizer_data_path = config.get_layerizer_data_path()
            best_model_file = '{}/best_layerizer_model.bin'.format(layerizer_data_path)
        self._best_model_file =  best_model_file
        

class BertLayerizer(BertPreTrainedModel):
    def __init__(self, config, args):
        # config 为各种参数
        super(BertLayerizer, self).__init__(config=config)

        self._args = args
        # 载入预训练 bert 模型
        # self._device = config.device
        self._bertTokenizer = BertTokenizer.from_pretrained(args._model_dir, local_files_only=True)
        # self._bert = BertModel.from_pretrained("bert-base-uncased").to(config.device)
        self._bert = BertModel(config=config)
        self._dropout = nn.Dropout(args._dropout_rate)

        # num_labels 为标签数量 
        self._cls = nn.Linear(self._args._hidden_size, args._num_labels)
        # self._cls = nn.Linear(768, args.num_labels)

        self.init_weights()

    def forward(self, content):
        tokens = self._bertTokenizer(content, return_tensors='pt', padding=True, truncation=True).to(self._args._device)
        outputs = self._bert(**tokens)
        logits = self._cls(self._dropout(outputs.pooler_output))

        return logits
