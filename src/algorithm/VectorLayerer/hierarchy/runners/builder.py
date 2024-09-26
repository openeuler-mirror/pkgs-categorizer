# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import copy
from typing import Type

from hierarchy.core.common_types import RunnerProtocol

from .anchor_based import AnchorBasedRunner

RUNNER_MAPPING: dict[str, Type[RunnerProtocol]] = {
    'AnchorBasedRunner': AnchorBasedRunner
}


def build_runner(runner_cfg: dict) -> RunnerProtocol:
    runner_cfg = copy.deepcopy(runner_cfg)

    if 'type' not in runner_cfg:
        raise ValueError('Runner config must have `type` field')
    runner_type = runner_cfg.pop('type')
    if runner_type not in RUNNER_MAPPING:
        raise ValueError(f'Runner `{runner_type}` is not support, '
                         f'available runners: {list(RUNNER_MAPPING.keys())}')
    runner = RUNNER_MAPPING[runner_type]

    return runner(**runner_cfg)
