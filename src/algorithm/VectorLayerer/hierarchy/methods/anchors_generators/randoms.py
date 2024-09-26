# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import random

from hierarchy.core import DirectedGraph
from hierarchy.core.common_types import NODE_TYPE


class RandomPercentAnchorsGenerator:

    def __init__(self, percent: float) -> None:
        if percent < 0 or percent > 1:
            raise ValueError(
                f'Percentage should between 0 and 1, but got: {percent}')
        self._percent = percent

    def __call__(self, graph: DirectedGraph) -> list[NODE_TYPE]:
        chosen_number = int(graph.node_number * self._percent)

        return random.sample(graph.node_list, chosen_number)


class RandomNumberAnchorsGenerator:

    def __init__(self, number: int) -> None:
        if number < 0:
            raise ValueError(
                f'Number should greater than 0, but got: {number}')
        self._number = number

    def __call__(self, graph: DirectedGraph) -> list[NODE_TYPE]:
        return random.sample(graph.node_list, self._number)
