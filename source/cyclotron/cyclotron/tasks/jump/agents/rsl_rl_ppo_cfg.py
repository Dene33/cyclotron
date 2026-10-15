# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""PPO configuration for the jump."""

from isaaclab.utils import configclass

from cyclotron.tasks.tracking.agents.rsl_rl_ppo_cfg import Asimov1TrackingPPORunnerCfg


@configclass
class Asimov1JumpPPORunnerCfg(Asimov1TrackingPPORunnerCfg):
    """The motion tracking task's PPO settings, logged in their own experiment folder."""

    experiment_name = "asimov1_jump"
    run_name = "jump"
