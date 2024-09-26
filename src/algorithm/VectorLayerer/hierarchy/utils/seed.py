# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
#                              Peking University
# SPDX-License-Identifier: Apache-2.0

import random
from typing import Any, Optional

import numpy as np

max_seed_value = np.iinfo(np.uint32).max
min_seed_value = np.iinfo(np.uint32).min


def _select_seed_randomly(min_seed_value: int = min_seed_value,
                          max_seed_value: int = max_seed_value) -> int:
    return random.randint(min_seed_value, max_seed_value)


def seed_everything(seed: Optional[int] = None) -> int:
    if seed is None:
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    if not (min_seed_value <= seed <= max_seed_value):
        seed = _select_seed_randomly(min_seed_value, max_seed_value)

    random.seed(seed)
    np.random.seed(seed)

    return seed


def collect_rng_states() -> dict[str, Any]:
    return {
        'numpy': np.random.get_state(),
        'python': random.getstate(),
    }


def set_rng_states(rng_state_dict: dict[str, Any]) -> None:
    np.random.set_state(rng_state_dict['numpy'])
    version, state, gauss = rng_state_dict['python']
    random.setstate((version, tuple(state), gauss))
