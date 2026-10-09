# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 max-jump configurations: how high can the robot jump?

A diagnostic task with no reference motion. The robot starts standing and is rewarded for pelvis height above
standing while both feet are off the ground, so the policy finds the highest vertical jump Asimov-1's actuators and
joint limits allow. Jumps must be two-footed: the feet are kept at the same height, or the policy learns to hop on one
leg while swinging the other one up for momentum. There are no pushes or randomized gains or mass, so the result is
an upper bound for the nominal robot.
"""

from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from cyclotron.assets.robots.asimov_1 import ASIMOV_1_ACTION_SCALE, ASIMOV_1_JOINT_NAMES
from cyclotron.tasks.tracking.tracking_env_cfg import FEET_BODIES, Asimov1TrackingSceneCfg

from . import mdp

# Pelvis height of Asimov-1 standing in its default pose.
STANDING_HEIGHT = 0.62

_JOINTS_CFG = SceneEntityCfg("robot", joint_names=list(ASIMOV_1_JOINT_NAMES), preserve_order=True)
_FEET_CONTACT_CFG = SceneEntityCfg("contact_forces", body_names=list(FEET_BODIES))


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ASIMOV_1_JOINT_NAMES),
        preserve_order=True,
        scale=ASIMOV_1_ACTION_SCALE,
        use_default_offset=True,
    )


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel, noise=Unoise(n_min=-0.2, n_max=0.2))
        projected_gravity = ObsTerm(func=mdp.projected_gravity, noise=Unoise(n_min=-0.05, n_max=0.05))
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": _JOINTS_CFG}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": _JOINTS_CFG}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)
        progress = ObsTerm(func=mdp.episode_progress)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        projected_gravity = ObsTerm(func=mdp.projected_gravity)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _JOINTS_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _JOINTS_CFG})
        actions = ObsTerm(func=mdp.last_action)
        progress = ObsTerm(func=mdp.episode_progress)
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_height = ObsTerm(func=mdp.base_pos_z)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class EventCfg:
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FEET_BODIES)),
            "static_friction_range": (0.8, 1.2),
            "dynamic_friction_range": (0.8, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    reset_base = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={"pose_range": {"yaw": (-3.14, 3.14)}, "velocity_range": {}},
    )
    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_offset,
        mode="reset",
        params={"position_range": (-0.05, 0.05), "velocity_range": (0.0, 0.0)},
    )


@configclass
class RewardsCfg:
    airborne_height = RewTerm(
        func=mdp.airborne_height,
        weight=10.0,
        params={
            "sensor_cfg": _FEET_CONTACT_CFG,
            "standing_height": STANDING_HEIGHT,
            "asset_cfg": SceneEntityCfg("robot"),
        },
    )
    rising_velocity = RewTerm(
        func=mdp.rising_velocity,
        weight=2.0,
        params={"standing_height": STANDING_HEIGHT, "asset_cfg": SceneEntityCfg("robot")},
    )
    two_footed = RewTerm(
        func=mdp.feet_height_asymmetry,
        weight=-10.0,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FEET_BODIES), preserve_order=True),
            "tolerance": 0.03,
        },
    )
    termination = RewTerm(func=mdp.is_terminated, weight=-20.0)
    upright = RewTerm(func=mdp.flat_orientation_l2, weight=-2.0)
    horizontal_speed = RewTerm(func=mdp.horizontal_speed_l2, weight=-0.5, params={"asset_cfg": SceneEntityCfg("robot")})
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.005)
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-1.0, params={"asset_cfg": _JOINTS_CFG})


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    tilted = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 0.7})
    too_low = DoneTerm(func=mdp.root_height_below_minimum, params={"minimum_height": 0.35})
    body_contact = DoneTerm(
        func=mdp.illegal_contact,
        params={
            "threshold": 20.0,
            "sensor_cfg": SceneEntityCfg(
                "contact_forces",
                body_names=["pelvis_link", "waist_yaw_link", ".*_knee_link", ".*_elbow_link", ".*_wrist_yaw_link"],
            ),
        },
    )


@configclass
class Asimov1MaxJumpEnvCfg(ManagerBasedRLEnvCfg):

    scene: Asimov1TrackingSceneCfg = Asimov1TrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 3.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (2.5, 2.5, 1.0)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class Asimov1MaxJumpEnvCfg_PLAY(Asimov1MaxJumpEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 4
        self.observations.policy.enable_corruption = False
