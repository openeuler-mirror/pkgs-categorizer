# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

from hierarchy.core import DirectedGraph
from hierarchy.core.common_types import NODE_TYPE


class MostDegreeAnchorsGenerator:

    def __init__(self, number: int) -> None:
        if number < 0:
            raise ValueError(
                f'Number should greater than 0, but got: {number}')
        self._number = number

    def __call__(self, graph: DirectedGraph) -> list[NODE_TYPE]:
        node2degree = graph.node2degree
        sorted_node_degree = sorted(node2degree.items(),
                                    key=lambda x: x[1],
                                    reverse=True)

        return [
            node_degree[0] for node_degree in sorted_node_degree[:self._number]
        ]
