# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import numpy as np


class NormDistanceMetric:

    def __init__(self, ord: int | str) -> None:
        self._ord = ord

    def __call__(self, x: np.ndarray, y: np.ndarray) -> float:
        return np.linalg.norm(x - y, ord=self._ord)
