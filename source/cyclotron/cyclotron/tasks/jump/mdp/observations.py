# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Observations for the jump's critic, which may know what the robot cannot measure."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from .commands import HOLD, JUMP, STAND, JumpCommand

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def jump_phase(env: ManagerBasedRLEnv, command_name: str) -> torch.Tensor:
    """Which phase the reference is in (standing, jumping, holding the landing; one-hot), how far through the jump
    (0 standing, 1 holding), and whether the robot is told to jump."""
    command: JumpCommand = env.command_manager.get_term(command_name)
    phases = torch.stack([command.phase == STAND, command.phase == JUMP, command.phase == HOLD], dim=-1).float()
    progress = ((command.time_steps - command.start_frame).float() / (command.jump_frames - 1)).clamp(0.0, 1.0)
    progress = torch.where(command.phase == STAND, 0.0, torch.where(command.phase == HOLD, 1.0, progress))
    told = (command.trigger_steps_left > 0).float()
    return torch.cat([phases, progress[:, None], told[:, None]], dim=-1)
