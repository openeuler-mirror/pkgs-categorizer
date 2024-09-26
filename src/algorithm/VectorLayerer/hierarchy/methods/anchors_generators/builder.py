# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Type

from hierarchy.core.common_types import AnchorsGeneratorProtocol

from .degree import MostDegreeAnchorsGenerator
from .randoms import (
    RandomNumberAnchorsGenerator,
    RandomPercentAnchorsGenerator,
)

ANCHORS_GENERATOR_MAPPING: dict[str, Type[AnchorsGeneratorProtocol]] = {
    'RandomPercent': RandomPercentAnchorsGenerator,
    'RandomNumber': RandomNumberAnchorsGenerator,
    'MostDegree': MostDegreeAnchorsGenerator
}


def build_anchors_generator(
        anchors_generator_cfg: dict) -> AnchorsGeneratorProtocol:
    anchors_generator_cfg = copy.deepcopy(anchors_generator_cfg)

    if 'type' not in anchors_generator_cfg:
        raise ValueError('Anchors generator config must have `type` field')
    anchors_generator_type = anchors_generator_cfg.pop('type')
    if anchors_generator_type not in ANCHORS_GENERATOR_MAPPING:
        raise ValueError(
            f'Anchors generator `{anchors_generator_type}` is not support, '
            f'available Anchors generators: '
            f'{list(ANCHORS_GENERATOR_MAPPING.keys())}')
    anchors_generator = ANCHORS_GENERATOR_MAPPING[anchors_generator_type]

    return anchors_generator(**anchors_generator_cfg)
