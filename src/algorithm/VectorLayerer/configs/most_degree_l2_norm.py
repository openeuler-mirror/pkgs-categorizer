# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

"""
runner_cfg = dict(type='AnchorBasedRunner',
                  input_file_path='./data/filtered_input_merged.txt',
                  expert_file_path='./data/filtered_human_1107.txt',
                  anchors_generator_cfg=dict(type='MostDegree', number=300),
                  distance_metric_cfg=dict(type='Norm', ord=2))
"""


runner_cfg = dict(type='AnchorBasedRunner',
                  input_file_path='./algorithm/VectorLayerer/data/2022-1228-src-input.txt',
                  expert_file_path='./algorithm/VectorLayerer/data/2022-1228-src-human.txt',
                  anchors_generator_cfg=dict(type='MostDegree', number=300),
                  distance_metric_cfg=dict(type='Norm', ord=2))
