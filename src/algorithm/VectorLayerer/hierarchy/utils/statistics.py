# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import collections
from typing import Iterable, Optional

import numpy as np


def get_layer_number(layers: Iterable[int]) -> dict[int, int]:
    layer2number: dict[int, int] = collections.defaultdict(lambda: 0)

    for layer in layers:
        layer2number[layer] += 1

    return layer2number


def get_layer_ratio(layers: Iterable[int]) -> dict[int, float]:
    layer2number = get_layer_number(layers)
    total_number = sum(layer2number.values())

    return {k: v / total_number for k, v in layer2number.items()}


def get_layer_vector(layers: Iterable[int],
                     min_layer: Optional[int] = None,
                     max_layer: Optional[int] = None) -> np.ndarray:
    layer2ratio = get_layer_ratio(layers)

    # TODO: add log
    if min_layer is None:
        min_layer = min(layer2ratio.keys())
    if max_layer is None:
        max_layer = max(layer2ratio.keys())

    vector = np.zeros(max_layer - min_layer + 1, dtype=np.float64)
    for layer, ratio in layer2ratio.items():
        vector[layer - min_layer] = ratio

    return vector
