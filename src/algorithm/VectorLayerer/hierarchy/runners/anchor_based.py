# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import collections

import numpy as np

from hierarchy.core import DirectedGraph
from hierarchy.core.common_types import NODE_TYPE, PATH_TYPE
from hierarchy.methods import build_anchors_generator, build_distance_metric
from hierarchy.utils import get_layer_vector, load_expert


class AnchorBasedRunner:

    def __init__(
        self,
        *,
        input_file_path: PATH_TYPE,
        expert_file_path: PATH_TYPE,
        anchors_generator_cfg: dict,
        distance_metric_cfg: dict = dict(type='l2')
    ) -> None:
        #print(input_file_path)
        self._graph = DirectedGraph.from_file(input_file_path)
        self._node2expert_layer = load_expert(expert_file_path)
        self._anchors_generator = build_anchors_generator(
            anchors_generator_cfg)
        self._distance_metric = build_distance_metric(distance_metric_cfg)

        self._max_layer = max(self._node2expert_layer.values())
        self._min_layer = min(self._node2expert_layer.values())
        self._unknown_layer = -1

        self._anchors_set: set[NODE_TYPE] = set()
        self._rest_node_set: set[NODE_TYPE] = set()
        self._node2layer: dict[NODE_TYPE, int] = dict()

    def run(self) -> None:
        self.before_run()

        rest_node_set = self._rest_node_set
        node2layer = self._node2layer
        while rest_node_set:
            picked_node = self._pick_node(rest_node_set, node2layer)
            rest_node_set.remove(picked_node)
            layer = self._layer_node(node2layer, picked_node)
            node2layer[picked_node] = layer

        self.after_run()

    def before_run(self) -> None:
        self._anchors_set = set(self._anchors_generator(self._graph))
        self._rest_node_set = self._graph.node_set - self._anchors_set
        # node layer calculated by algorithm
        self._node2layer = {
            node: self._node2expert_layer[node]
            for node in self._anchors_set
        }

    def after_run(self) -> None:
        anchors_set = self._anchors_set
        node2layer = self._node2layer

        #print('=== 不含锚定节点的准确率 ===')
        layer2number: dict[int, int] = collections.defaultdict(lambda: 0)
        layer2correct: dict[int, int] = collections.defaultdict(lambda: 0)
        for node, layer in node2layer.items():
            if node in anchors_set:
                continue
            correct_layer = self._node2expert_layer[node]
            layer2number[correct_layer] += 1

            if layer == correct_layer:
                layer2correct[layer] += 1

        for layer in range(self._min_layer, self._max_layer + 1):
            if layer not in layer2number:
                continue
            #print(f'第 {layer} 层的准确率: '
                  #f'{layer2correct[layer] / layer2number[layer]}')

        #print(
            #f'准确率: {sum(layer2correct.values()) / sum(layer2number.values())}')

        #print('=== 含锚定节点的准确率 ===')
        layer2number = collections.defaultdict(lambda: 0)
        layer2correct = collections.defaultdict(lambda: 0)
        for node, layer in node2layer.items():
            correct_layer = self._node2expert_layer[node]
            layer2number[correct_layer] += 1

            if layer == correct_layer:
                layer2correct[layer] += 1

        for layer in range(self._min_layer, self._max_layer + 1):
            if layer not in layer2number:
                continue
            #print(f'第 {layer} 层的准确率: '
                  #f'{layer2correct[layer] / layer2number[layer]}')

        #print(
            #f'准确率: {sum(layer2correct.values()) / sum(layer2number.values())}')

    def _pick_node(self, rest_node_set: set[NODE_TYPE],
                   node2layer: dict[NODE_TYPE, int]) -> NODE_TYPE:
        # 找到依赖的包中，已知节点占比最高的一个节点
        node2weight: dict[str, float] = dict()

        for node in rest_node_set:
            target_list = self._graph.get_node_target_list(node)
            if len(target_list) == 0:
                node2weight[node] = 0.
                continue

            layered_target_number = 0
            for target in target_list:
                if target in node2layer:
                    layered_target_number += 1
            node2weight[node] = layered_target_number / len(target_list)

        picked_node = max(node2weight.items(), key=lambda x: x[1])[0]

        return picked_node

    def _layer_node(self, node2layer: dict[NODE_TYPE, int],
                    picked_node: NODE_TYPE) -> int:
        if self._graph.get_node_out_degree(picked_node) == 0:
            return -1

        layer_vector = get_layer_vector(node2layer.values(),
                                        min_layer=self._min_layer,
                                        max_layer=self._max_layer)

        # TODO: optimize logic
        target_node2weighted_vector: dict[str, np.ndarray] = dict()
        for target_node in self._graph.get_node_target_list(picked_node):
            in_node2layer = {
                in_node: node2layer[in_node]
                for in_node in self._graph.get_node_source_list(target_node)
                if in_node in node2layer
            }
            target_node_vector = get_layer_vector(in_node2layer.values(),
                                                  min_layer=self._min_layer,
                                                  max_layer=self._max_layer)

            distance = self._distance_metric(layer_vector, target_node_vector)

            # TODO: check distance
            # l2 norm is different from euclidean distance,
            # but should not affect result?
            target_node2weighted_vector[
                target_node] = distance * target_node_vector

        probability_vector = sum(target_node2weighted_vector.values())

        return np.argmax(probability_vector) + self._min_layer


if __name__ == '__main__':
    runner = AnchorBasedRunner(input_file_path=r'data\input_10-10(remove).txt',
                               expert_file_path=r'data\人工标签1107_formatted.txt',
                               anchors_generator_cfg=dict(
                                   type='random_percent', percent=0.1))

    runner.run()
