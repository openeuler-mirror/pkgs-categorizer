# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

from pathlib import Path
from typing import Any, Protocol

import numpy as np

NODE_TYPE = str
EDGE_TYPE = tuple[NODE_TYPE, NODE_TYPE]
GRAPH_TYPE = dict[NODE_TYPE, set[NODE_TYPE]]

PATH_TYPE = str | Path


class RunnerProtocol(Protocol):

    def run(self) -> None:
        ...


class AnchorsGeneratorProtocol(Protocol):

    def __call__(self, graph: Any) -> list[NODE_TYPE]:
        ...


class DistanceMetricProtocol(Protocol):

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        ...
