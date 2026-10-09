# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""PPO configuration for the max-jump task."""

from isaaclab.utils import configclass

from cyclotron.tasks.tracking.agents.rsl_rl_ppo_cfg import Asimov1TrackingPPORunnerCfg


@configclass
class Asimov1MaxJumpPPORunnerCfg(Asimov1TrackingPPORunnerCfg):
    """The motion tracking PPO settings."""

    max_iterations = 1000
    experiment_name = "asimov1_max_jump"
    run_name = "max_jump"
