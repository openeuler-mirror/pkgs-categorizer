# SPDX-FileCopyrightText: 2024 UnionTech Software Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

import sys
import os

import shutil
from argparse import ArgumentParser
from pathlib import Path
sys.path.append(os.getcwd())
sys.path.append("/home/user/src_fcfl/algorithm/VectorLayerer/")
from hierarchy.config import Config
from hierarchy.runners import build_runner
from hierarchy.utils import seed_everything
import pprint

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('config', type=Path, help='Path of config file')
    parser.add_argument('--work_dirs',
                        '-w',
                        type=Path,
                        default='work_dirs',
                        help='Path of work directory')
    parser.add_argument('--seed', '-s', type=int, help='Random seed')

    args = parser.parse_args()
    
    seed_everything(args.seed)

    config_path: Path = args.config
    config = Config.fromfile(config_path)

    runner_cfg = config.runner_cfg
    work_dir: Path = args.work_dirs / config_path.stem
    if not work_dir.exists():
        work_dir.mkdir(parents=True)
    shutil.copy(config_path, work_dir)

    #print(config.pretty_text)

    runner = build_runner(config.runner_cfg)
    runner.run()
    pprint.pprint(runner._node2layer)
