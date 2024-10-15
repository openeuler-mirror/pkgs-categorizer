# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import os
import sys
#def exec_run():
def exec_run(config_path):
    #output = os.popen("python3 algorithm/VectorLayerer/tools/run.py ./algorithm/VectorLayerer/configs/most_degree_l2_norm.py > /var/fcfl/layer_result.txt")
    output = os.popen("python3 algorithm/VectorLayerer/tools/run.py {config} > /var/fcfl/layer_result.txt".format(config = config_path))
    layer = output.read()
    return layer

if __name__ == "__main__":
    layer = exec_run()
    


