# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 jump configurations: a forward jump on command, with the walking policy's interface.

The policy observes and acts exactly as the walking policy does (``Asimov1-Velocity``): the same 78 observations in
the same order, with the same scales, delays and noise, and the same joint position actions, robot and actuators. So
it has the same deployment contract (``./cyclotron.sh --export``) and runs wherever the walking policy runs. Its
velocity command is the trigger: at zero it stands; with a forward speed held through the jump, it jumps, and still
held after landing, it jumps again.

The jump is learned from a reference motion with the motion tracking rewards (``Asimov1-Tracking``), but the policy
never sees the reference: only the rewards and the critic do (see ``mdp.JumpCommand``). Asimov 1's firmware clamps
the ankle targets it is sent (0.35 rad pitch, 0.1 rad roll), so the simulated ankles do too.
"""

import dataclasses

from isaaclab.actuators import ActuatorBaseCfg
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.utils import configclass

from cyclotron.assets.actuators import TargetLimitedDelayedPDActuatorCfg
from cyclotron.tasks.locomotion.velocity_env_cfg import ActionsCfg as VelocityActionsCfg
from cyclotron.tasks.locomotion.velocity_env_cfg import ObservationsCfg as VelocityObservationsCfg
from cyclotron.tasks.tracking.tracking_env_cfg import (
    ANCHOR_BODY,
    FEET_BODIES,
    FOOT_SOLE_POINTS,
    PUSH_VELOCITY_RANGE,
    TRACKED_BODIES,
    Asimov1TrackingEnvCfg,
)
from cyclotron.tasks.tracking.tracking_env_cfg import ObservationsCfg as TrackingObservationsCfg

from . import mdp

# Asimov 1's firmware clamps the ankle targets it is sent to these (rad), by actuator group: menlo-sdk's
# menlo.asimov.robots, from the firmware's policy_thread.c (apply_ankle_coupling).
FIRMWARE_TARGET_LIMITS = {"ankle_pitch": 0.35, "ankle_roll": 0.1}


def firmware_limited(actuators: dict[str, ActuatorBaseCfg]) -> dict[str, ActuatorBaseCfg]:
    """``actuators`` with the groups the firmware clamps the targets of, clamping them too."""
    limited = dict(actuators)
    for name, limit in FIRMWARE_TARGET_LIMITS.items():
        actuator = actuators[name]
        values = {field.name: getattr(actuator, field.name) for field in dataclasses.fields(actuator)}
        values.pop("class_type")
        limited[name] = TargetLimitedDelayedPDActuatorCfg(**values, target_limit=limit)
    return limited


@configclass
class CommandsCfg:
    motion = mdp.JumpCommandCfg(
        asset_name="robot",
        anchor_body_name=ANCHOR_BODY,
        body_names=TRACKED_BODIES,
        foot_body_names=FEET_BODIES,
        foot_sole_points=FOOT_SOLE_POINTS,
        resampling_time_range=(1.0e9, 1.0e9),
        debug_vis=False,
        pose_range={
            "x": (-0.05, 0.05),
            "y": (-0.05, 0.05),
            "z": (-0.01, 0.01),
            "roll": (-0.1, 0.1),
            "pitch": (-0.1, 0.1),
            "yaw": (-0.2, 0.2),
        },
        velocity_range=PUSH_VELOCITY_RANGE,
        joint_position_range=(-0.1, 0.1),
    )
    # Named as in the walking task, so the policy's command observation is the walking policy's.
    twist = mdp.JumpTriggerCommandCfg(jump_command_name="motion")


@configclass
class ObservationsCfg:
    @configclass
    class CriticCfg(TrackingObservationsCfg.CriticCfg):
        """The tracking critic's observations, the trigger, and where the reference is in the jump."""

        trigger = ObsTerm(func=mdp.generated_commands, params={"command_name": "twist"})
        jump_phase = ObsTerm(func=mdp.jump_phase, params={"command_name": "motion"})

    policy: VelocityObservationsCfg.PolicyCfg = VelocityObservationsCfg.PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class Asimov1JumpEnvCfg(Asimov1TrackingEnvCfg):

    observations: ObservationsCfg = ObservationsCfg()
    actions: VelocityActionsCfg = VelocityActionsCfg()
    commands: CommandsCfg = CommandsCfg()

    def __post_init__(self):
        super().__post_init__()
        self.scene.robot = self.scene.robot.replace(actuators=firmware_limited(self.scene.robot.actuators))


@configclass
class Asimov1JumpEnvCfg_PLAY(Asimov1JumpEnvCfg):
    """One robot story at a time: standing in the default pose for 1.5 s, one jump with the command released 0.4 s
    after landing, then holding the landing for 3 s."""

    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = int(1e9)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.qpos0_rand = None
        self.events.base_com = None
        self.events.pd_gains_rand = None
        motion = self.commands.motion
        motion.pose_range = {}
        motion.velocity_range = {}
        motion.joint_position_range = (0.0, 0.0)
        motion.stand_start_prob = 1.0
        motion.default_pose_prob = 1.0
        motion.stand_time_s = (1.5, 1.5)
        motion.hold_time_s = (3.0, 3.0)
        motion.release_after_landing_s = (0.4, 0.4)
        motion.jump_again_prob = 0.0
        motion.trigger_speed = (0.3, 0.3)
        motion.debug_vis = True
