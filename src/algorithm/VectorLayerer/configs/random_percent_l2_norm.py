# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

runner_cfg = dict(type='AnchorBasedRunner',
                  input_file_path='./data/input_10-10(remove).txt',
                  expert_file_path='./data/人工标签1107_formatted.txt',
                  anchors_generator_cfg=dict(type='RandomPercent',
                                             percent=0.1),
                  distance_metric_cfg=dict(type='Norm', ord=2))
