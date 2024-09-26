# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

runner_cfg = dict(type='AnchorBasedRunner',
                  input_file_path='../data/filtered_input_merged.txt',
                  expert_file_path='../data/filtered_human_1107.txt',
                  anchors_generator_cfg=dict(type='MostDegree', number=300),
                  distance_metric_cfg=dict(type='Norm', ord=2))
