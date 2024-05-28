#  Copyright (c) Meta Platforms, Inc. and affiliates.
#
#  This source code is licensed under the license found in the
#  LICENSE file in the root directory of this source tree.
#
# % pip install eztils
import argparse
from dataclasses import dataclass
import importlib
from itertools import product
import json
import math
import sys
import time
import torch
import numpy as np
import wandb
from benchmarl import hydra_config
from benchmarl.algorithms import MappoConfig, IppoConfig
from benchmarl.environments import VmasTask
from benchmarl.experiment import Experiment, ExperimentConfig
from benchmarl.models.mlp import MlpConfig
from omegaconf import DictConfig, OmegaConf
import hydra
import os
from hydra.core.hydra_config import HydraConfig
from eztils import abspath


def get_user_defined_attrs(cls) -> list:
    return [
        attr
        for attr in dir(cls)
        if not callable(getattr(cls, attr)) and not attr.startswith("__")
    ]


class BaseHyperParameters:
    @classmethod
    def get_product(cls):
        return list(
            product(*[getattr(cls, attr) for attr in get_user_defined_attrs(cls)])
        )


class HyperParameters(BaseHyperParameters):
    _0_gamma = [0.99]
    _1_lr = [0.002]
    _2_off_policy_collected_frames_per_batch = [4000, 6000, 8000]
    _3_off_policy_train_batch_size = [64, 128, 256]
    _4_off_policy_n_optimizer_steps = [500, 1000, 1500]
    _5_seed = [
        0, 1,
    ]


hparams = HyperParameters.get_product()


def calculate_split(total_splits, total_len, index):
    # Calculate the length of each split
    split_length = total_len // total_splits

    # Calculate the start and end indices of the split
    start_index = index * split_length
    end_index = start_index + split_length

    # Adjust the end index if the split is not evenly divided
    if index == total_splits - 1:
        end_index = total_len

    return start_index, end_index



if __name__ == "__main__":
    # parser=HfArgumentParser((JobParams,))
    # parser.add_argument("-c", "--config", type=str)
    # jobparams, d=parser.parse_args_into_dataclasses()

    parser = argparse.ArgumentParser(
        description="Run the experiment with dynamic configuration."
    )
    parser.add_argument("--job", type=int, required=True, help="Which job am I?")
    parser.add_argument(
        "--total",
        type=int,
        required=True,
        help="How many total jobs are there running?",
    )

    args = parser.parse_args()

    # Loads from "benchmarl/conf/experiment/base_experiment.yaml"
    experiment = ExperimentConfig.get_from_yaml()

    start_index, end_index = calculate_split(
        args.total, len(hparams), args.job
    )
    print('\n\n\n', 'TOTAL HPARAMS:', len(hparams), '\n', 'JOB HPARAMS:', end_index - start_index, '\n\n\n', file=sys.stderr)
    print(len(hparams))
    for hparam in hparams[start_index:end_index]:
        (
            gamma,
            lr,
            collected_frames,
            train_batch_size,
            n_optimizer_steps,
            seed,
        ) = hparam # format each item in tax_bracket to string
        # use hparams from assigned segment of the possible hparam configurations
        overrides = [
            f"experiment.gamma={gamma}",
            f"experiment.lr={lr}",
            f"experiment.off_policy_collected_frames_per_batch={collected_frames}",
            f"experiment.off_policy_train_batch_size={train_batch_size}",
            f"experiment.off_policy_n_optimizer_steps={n_optimizer_steps}",
            "experiment.off_policy_n_envs_per_worker=1",
            "experiment.loggers=[wandb]",
            "experiment.train_device=cuda",
            "experiment.sampling_device=cuda",
            "algorithm=isac",
            "model=layers/cnn",
            "model@critic_model=layers/cnn",
            "task=meltingpot/commons_harvest__open",  # reward = 0 after 100 timessteps?
            f"seed={seed}",
            "experiment.create_json=False",
        ]
        
        cmd = f"python {abspath()}/run.py {' '.join(overrides)}"
        print("Running", cmd, file=sys.stderr)

        os.system(cmd)