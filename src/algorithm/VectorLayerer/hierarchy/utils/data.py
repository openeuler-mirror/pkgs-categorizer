# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

# TODO: put in `core` or `utils`
from pathlib import Path
from typing import Any, Iterable

from hierarchy.core.common_types import EDGE_TYPE, PATH_TYPE


def _to_int_list(items: Iterable[Any]) -> list[int]:
    res = []
    for item in items:
        try:
            int_item = int(item)
        except ValueError:
            continue
        else:
            res.append(int_item)

    return res


def _check_and_parse_path(path: PATH_TYPE) -> Path:
    if isinstance(path, str):
        path = Path(path)
    if not isinstance(path, Path):
        raise ValueError(f'Expect type to be Path, but got: {type(path)}')
    if not path.exists():
        raise ValueError(f'Path `{str(path)}` is not exists')

    return path


def load_edges(file_path: PATH_TYPE) -> list[EDGE_TYPE]:

    def _format(s: str) -> str:
        return s.strip(' \n"“”;；')

    file_path = _check_and_parse_path(file_path)

    edges: list[tuple[str, str]] = []
    with file_path.open('r', encoding='utf8') as f:
        for line in f:
            if '->' not in line:
                continue
            line = line.strip()

            edge = [_format(s) for s in line.split('->')]
            if len(edge) != 2:
                raise ValueError(f'Unable to process line `{line}`')

            edges.append(tuple(edge))  # type: ignore

    return edges


def load_expert(file_path: PATH_TYPE) -> dict[str, int]:
    file_path = _check_and_parse_path(file_path)

    package2expert_layer: dict[str, int] = dict()
    with file_path.open('r', encoding='utf8') as f:
        for line in f:
            line = line.strip()
            splitted_line = line.split('\t')

            package = splitted_line[0]
            int_list = _to_int_list(splitted_line[1:])
            if len(int_list) != 1:
                raise ValueError(f'Unable to process line `{line}`')

            layer = int_list[0]
            if package in package2expert_layer:
                raise ValueError(f'Package `{package}` already exists before')
            package2expert_layer[package] = layer

    return package2expert_layer


def load_expert_multi_label(file_path: PATH_TYPE) -> dict[str, set[int]]:
    file_path = _check_and_parse_path(file_path)

    package2expert_layers: dict[str, set[int]] = dict()
    with file_path.open('r', encoding='utf8') as f:
        for line in f:
            line = line.strip().replace(' ', '\t')
            splitted_line = line.split('\t')

            package = splitted_line[0]
            layers = _to_int_list(splitted_line[1:])
            if len(layers) < 1:
                print(f'Unable to process line `{line}`')
                continue

            if package in package2expert_layers:
                raise ValueError(f'Package `{package}` already exists before')
            package2expert_layers[package] = set(layers)

    return package2expert_layers


if __name__ == '__main__':
    edges = load_edges(r'data\input_10-10(remove).txt')
    print(len(edges))

    package2expert_layer = load_expert(r'data\人工标签1107_formatted.txt')
    print(len(package2expert_layer))

    layer2packages: dict[int, list[str]] = dict()
    for package, layer in package2expert_layer.items():
        try:
            layer2packages[layer].append(package)
        except KeyError:
            layer2packages[layer] = [package]
    total_packages = sum((len(x) for x in layer2packages.values()))
    for i in range(1, 5):
        print(f'第 i 层节点数量: {len(layer2packages[i])}, '
              f'占比: {len(layer2packages[i]) / total_packages}')
    old_packages = set([p for ps in layer2packages.values() for p in ps])

    package2expert_layers = load_expert_multi_label('data/人工分类(含多个评价).txt')
    print(len(package2expert_layers))
    layer2packages = dict()
    for package, layers in package2expert_layers.items():
        for layer in layers:
            try:
                layer2packages[layer].append(package)
            except KeyError:
                layer2packages[layer] = [package]

    total_packages = sum(len(x) for x in layer2packages.values())
    for i in range(1, 5):
        print(f'第 i 层节点数量: {len(layer2packages[i])}, '
              f'占比: {len(layer2packages[i]) / total_packages}')

    new_packages = set([p for ps in layer2packages.values() for p in ps])

    multi_label_packages = [
        p for p, ls in package2expert_layers.items() if len(ls) > 1
    ]
    print(len(multi_label_packages))
    print(multi_label_packages)

    assert old_packages == new_packages

    edges = load_edges('data/input_10-10(remove).txt')
    node2targets: dict[str, set[str]] = dict()
    for source, target in edges:
        if source in node2targets:
            if target in node2targets[source]:
                # TODO: add warning
                print(f'Trying to add existing edge: {source} -> {target}')
            node2targets[source].add(target)
        else:
            node2targets[source] = {target}
