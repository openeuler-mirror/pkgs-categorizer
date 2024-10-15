# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import copy
from typing import Iterable

from hierarchy.core.common_types import EDGE_TYPE, GRAPH_TYPE, NODE_TYPE


class DirectedGraph:

    def __init__(self, node2targets: GRAPH_TYPE) -> None:
        node2targets = self.add_zero_out_degree_nodes(node2targets)
        self._node2targets = node2targets

        node2sources = self.reverse_graph(node2targets)
        node2sources = self.add_zero_out_degree_nodes(node2sources)
        self._node2sources = node2sources

    def get_node_target_list(self, node: NODE_TYPE) -> list[NODE_TYPE]:
        self._check_node_in_graph(node)

        return list(self._node2targets[node])

    def get_node_source_list(self, node: NODE_TYPE) -> list[NODE_TYPE]:
        self._check_node_in_graph(node)

        return list(self._node2sources[node])

    def get_node_out_degree(self, node: NODE_TYPE) -> int:
        self._check_node_in_graph(node)

        return len(self._node2targets[node])

    def get_node_in_degree(self, node: NODE_TYPE) -> int:
        self._check_node_in_graph(node)

        return len(self._node2sources[node])

    def _check_node_in_graph(self, node: NODE_TYPE) -> None:
        if node not in self._node2targets:
            raise ValueError(f'Node {node} is not in graph')

    @classmethod
    def from_file(cls, file_path) -> DirectedGraph:
        from hierarchy.utils import load_edges

        edges = load_edges(file_path)
        print(len(edges))
        return cls.from_edges(edges)

    @classmethod
    def from_edges(cls, edges: Iterable[EDGE_TYPE]) -> DirectedGraph:
        node2targets: GRAPH_TYPE = dict()

        for source, target in edges:
            if source in node2targets:
                if target in node2targets[source]:
                    # TODO: add warning
                    print(f'Trying to add existing edge: {source} -> {target}')
                node2targets[source].add(target)
            else:
                node2targets[source] = {target}

        return cls(node2targets)

    @property
    def node_list(self) -> list[NODE_TYPE]:
        return list(self._node2targets.keys())

    @property
    def node_set(self) -> set[NODE_TYPE]:
        return set(self._node2targets.keys())

    @property
    def node_number(self) -> int:
        return len(self._node2targets)

    @property
    def node2degree(self) -> dict[NODE_TYPE, int]:
        return {
            node: len(self._node2sources[node]) + len(self._node2targets[node])
            for node in self.node_list
        }

    @staticmethod
    def add_zero_out_degree_nodes(node2targets: GRAPH_TYPE) -> GRAPH_TYPE:
        new_node2targets = copy.deepcopy(node2targets)

        sources = set(node2targets.keys())
        for targets in node2targets.values():
            zero_out_degree_nodes = targets - sources
            for node in zero_out_degree_nodes:
                new_node2targets[node] = set()

        return new_node2targets

    @staticmethod
    def reverse_graph(graph: GRAPH_TYPE) -> GRAPH_TYPE:
        reversed_graph: GRAPH_TYPE = dict()

        for source, targets in graph.items():
            for target in targets:
                if target in reversed_graph:
                    reversed_graph[target].add(source)
                else:
                    reversed_graph[target] = {source}

        return reversed_graph


if __name__ == '__main__':
    dg = DirectedGraph.from_file(r'data\input_10-10(remove).txt')
    print(len(dg.node_list))

    print(dg.node2degree)
