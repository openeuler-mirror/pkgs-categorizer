# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import sys
import pprint
import os
sys.path.append(os.getcwd())
#print(os.getcwd())
from algorithm.VectorLayerer.tools.exec_run import exec_run

class LAYER():
    """
    def __init__():
        self.seed = None
        self.config = "VectorLayerer/configs/random_percent_l1_norm.py"
        self.work_dirs = "VectorLayerer/tools/work_dirs"
    """
    #def exec_layer(self):
    def exec_layer(self,vec_config_path):
        layer_result = dict()
        exec_run(vec_config_path)
        """
        读取分层分类信息，形成字典
        """
        with open("/var/fcfl/layer_result.txt", 'r') as f:
            line = f.readline()
            line = f.readline()
            while line:
                line_list = line.split("'")
                pkgName = line_list[1]
                layer = line_list[2].split(" ")[1].rstrip()[:-1]

                layer_result[pkgName] = layer
                line = f.readline()
        #pprint.pprint(layer_result)
        return layer_result

class CLASSIFICATION():
    def __init__(self):
        a = 1

    def exec_calssification(self):
        classification_dict = {'a':1,
                                'b':2,
                                'c':3,
                                'd':4,
                                'f':5,
                                'e':6}
        return classification_dict



if __name__ == "__main__":
    layer = LAYER()
    layer_result = layer.exec_layer()
    pprint.pprint(layer_result)

