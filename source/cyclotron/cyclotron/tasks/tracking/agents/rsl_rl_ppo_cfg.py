# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""PPO configuration for motion tracking."""

from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg

from cyclotron.tasks.locomotion.agents.rsl_rl_ppo_cfg import Asimov1PPORunnerCfg


@configclass
class Asimov1TrackingPPORunnerCfg(Asimov1PPORunnerCfg):
    """The velocity task's PPO settings, with observation normalization as in whole_body_tracking.

    Checkpoints are saved more often than for the velocity task, so a stopped run loses little progress.
    """

    save_interval = 100
    experiment_name = "asimov1_tracking"
    run_name = "tracking"
    actor = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
        distribution_cfg=RslRlMLPModelCfg.GaussianDistributionCfg(init_std=1.0, std_type="scalar"),
    )
    critic = RslRlMLPModelCfg(
        hidden_dims=[512, 256, 128],
        activation="elu",
        obs_normalization=True,
    )
