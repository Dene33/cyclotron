# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Actuator models."""

from __future__ import annotations

from dataclasses import MISSING

import torch

from isaaclab.actuators import DelayedPDActuator, DelayedPDActuatorCfg
from isaaclab.utils import configclass
from isaaclab.utils.types import ArticulationActions


class DelayedDCMotor(DelayedPDActuator):
    """Delayed PD actuator whose torque falls with speed, as in a DC motor.

    The torque limit falls linearly from ``saturation_effort`` at standstill to zero at ``no_load_speed``, and is
    capped at ``effort_limit``, as in Isaac Lab's ``DCMotor``. ``no_load_speed`` is a per-environment tensor, so it can
    be randomised when a motor's curve is not known exactly.
    """

    cfg: DelayedDCMotorCfg

    def __init__(self, cfg: DelayedDCMotorCfg, *args, **kwargs):
        super().__init__(cfg, *args, **kwargs)
        self.saturation_effort = self._parse_joint_parameter(cfg.saturation_effort, None)
        self.no_load_speed = self._parse_joint_parameter(cfg.no_load_speed, None)
        self.nominal_no_load_speed = self.no_load_speed.clone()
        self._joint_vel = torch.zeros_like(self.computed_effort)

    def compute(
        self, control_action: ArticulationActions, joint_pos: torch.Tensor, joint_vel: torch.Tensor
    ) -> ArticulationActions:
        self._joint_vel[:] = joint_vel
        return super().compute(control_action, joint_pos, joint_vel)

    def _clip_effort(self, effort: torch.Tensor) -> torch.Tensor:
        # Beyond the speeds where the curve meets the effort limit, hold the limit (as in Isaac Lab's DCMotor).
        corner = self.no_load_speed * (1.0 + self.effort_limit / self.saturation_effort)
        joint_vel = torch.clip(self._joint_vel, min=-corner, max=corner)
        max_effort = torch.clip(self.saturation_effort * (1.0 - joint_vel / self.no_load_speed), max=self.effort_limit)
        min_effort = torch.clip(
            self.saturation_effort * (-1.0 - joint_vel / self.no_load_speed), min=-self.effort_limit
        )
        return torch.clip(effort, min=min_effort, max=max_effort)


@configclass
class DelayedDCMotorCfg(DelayedPDActuatorCfg):
    """Configuration for :class:`DelayedDCMotor`."""

    class_type: type = DelayedDCMotor

    saturation_effort: dict[str, float] | float = MISSING
    """Torque at standstill (stall or peak torque), in N·m."""

    no_load_speed: dict[str, float] | float = MISSING
    """Speed at which the motor can no longer produce torque, in rad/s."""
