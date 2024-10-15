# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Type

from hierarchy.core.common_types import DistanceMetricProtocol

from .norm import NormDistanceMetric

DISTANCE_METRIC_MAPPING: dict[str, Type[DistanceMetricProtocol]] = {
    'Norm': NormDistanceMetric
}


def build_distance_metric(distance_metric_cfg: dict) -> DistanceMetricProtocol:
    distance_metric_cfg = copy.deepcopy(distance_metric_cfg)

    if 'type' not in distance_metric_cfg:
        raise ValueError('Distance metric config must have `type` field')
    distance_metric_type = distance_metric_cfg.pop('type')
    if distance_metric_type not in DISTANCE_METRIC_MAPPING:
        raise ValueError(
            f'Distance metric `{distance_metric_type}` is not support, '
            f'available distance metrics: '
            f'{list(DISTANCE_METRIC_MAPPING.keys())}')
    distance_metric = DISTANCE_METRIC_MAPPING[distance_metric_type]

    return distance_metric(**distance_metric_cfg)
