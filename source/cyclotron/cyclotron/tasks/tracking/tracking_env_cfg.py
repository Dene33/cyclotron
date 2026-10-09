# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Asimov-1 motion tracking configurations.

The policy learns to follow one reference motion, e.g. a retargeted jump, frame by frame. The MDP follows
whole_body_tracking (BeyondMimic, https://github.com/HybridRobotics/whole_body_tracking), Copyright (c) 2024,
The Isaac Lab Project Developers, MIT License. The robot, actuators and domain randomization are the ones from the
Asimov-1 velocity task, and the policy only observes what Asimov-1 measures on board.
"""

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

from cyclotron.assets.robots.asimov_1 import ASIMOV_1_ACTION_SCALE, ASIMOV_1_DELAYED_CFG, ASIMOV_1_JOINT_NAMES

from . import mdp

# The pelvis carries the IMU, so the policy can sense its orientation on the real robot.
ANCHOR_BODY = "pelvis_link"
TRACKED_BODIES = [
    "pelvis_link",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "waist_yaw_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
]
END_EFFECTORS = ["left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"]
FEET_BODIES = ["left_ankle_roll_link", "right_ankle_roll_link"]
# Heel and toe: the bottoms of the collision spheres Asimov-1's feet touch the ground with, in the foot frame.
FOOT_SOLE_POINTS = [(-0.045, 0.0, -0.034), (0.122, 0.0, -0.034)]
TORSO_BODY = "waist_yaw_link"

PUSH_VELOCITY_RANGE = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.2, 0.2),
    "roll": (-0.52, 0.52),
    "pitch": (-0.52, 0.52),
    "yaw": (-0.78, 0.78),
}

_JOINTS_CFG = SceneEntityCfg("robot", joint_names=list(ASIMOV_1_JOINT_NAMES), preserve_order=True)


# Scene


@configclass
class Asimov1TrackingSceneCfg(InteractiveSceneCfg):

    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
        ),
        debug_vis=False,
    )

    robot: ArticulationCfg = ASIMOV_1_DELAYED_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

    contact_forces = ContactSensorCfg(
        prim_path="{ENV_REGEX_NS}/Robot/.*", history_length=3, track_air_time=True, force_threshold=10.0
    )

    sky_light = AssetBaseCfg(
        prim_path="/World/skyLight",
        spawn=sim_utils.DomeLightCfg(intensity=750.0, color=(0.9, 0.9, 0.9)),
    )


# Commands


@configclass
class CommandsCfg:
    motion = mdp.MotionCommandCfg(
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


# Actions


@configclass
class ActionsCfg:
    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=list(ASIMOV_1_JOINT_NAMES),
        preserve_order=True,
        scale=ASIMOV_1_ACTION_SCALE,
        use_default_offset=True,
    )


# Observations


@configclass
class ObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        """What the robot can measure: the reference, IMU orientation and angular velocity, and its joints."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(
            func=mdp.motion_anchor_ori_b, params={"command_name": "motion"}, noise=Unoise(n_min=-0.05, n_max=0.05)
        )
        base_ang_vel = ObsTerm(
            func=mdp.delayed_obs,
            params={"quantity": "base_ang_vel", "min_lag": 0, "max_lag": 1},
            noise=Unoise(n_min=-0.2, n_max=0.2),
        )
        joint_pos = ObsTerm(
            func=mdp.joint_pos_rel, params={"asset_cfg": _JOINTS_CFG}, noise=Unoise(n_min=-0.01, n_max=0.01)
        )
        joint_vel = ObsTerm(
            func=mdp.joint_vel_rel, params={"asset_cfg": _JOINTS_CFG}, noise=Unoise(n_min=-0.5, n_max=0.5)
        )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    @configclass
    class CriticCfg(ObsGroup):
        """Adds what only the simulator knows: position relative to the reference and the pose of every body."""

        command = ObsTerm(func=mdp.generated_commands, params={"command_name": "motion"})
        motion_anchor_pos_b = ObsTerm(func=mdp.motion_anchor_pos_b, params={"command_name": "motion"})
        motion_anchor_ori_b = ObsTerm(func=mdp.motion_anchor_ori_b, params={"command_name": "motion"})
        body_pos = ObsTerm(func=mdp.robot_body_pos_b, params={"command_name": "motion"})
        body_ori = ObsTerm(func=mdp.robot_body_ori_b, params={"command_name": "motion"})
        base_lin_vel = ObsTerm(func=mdp.base_lin_vel)
        base_ang_vel = ObsTerm(func=mdp.base_ang_vel)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, params={"asset_cfg": _JOINTS_CFG})
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, params={"asset_cfg": _JOINTS_CFG})
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = True

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


# Domain randomization


@configclass
class EventCfg:
    foot_friction = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=list(FEET_BODIES)),
            "static_friction_range": (0.6, 1.2),
            "dynamic_friction_range": (0.6, 1.2),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 64,
            "make_consistent": True,
        },
    )
    qpos0_rand = EventTerm(
        func=mdp.randomize_joint_default_pos,
        mode="startup",
        params={
            "ranges": (-0.02, 0.02),
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
        },
    )
    base_com = EventTerm(
        func=mdp.randomize_rigid_body_com,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=[TORSO_BODY]),
            "com_range": {"x": (0.0, 0.05), "y": (0.0, 0.0), "z": (0.03, 0.07)},
        },
    )
    pd_gains_rand = EventTerm(
        func=mdp.randomize_actuator_gains,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=[".*"]),
            "stiffness_distribution_params": (0.8, 1.2),
            "damping_distribution_params": (0.8, 1.2),
            "operation": "scale",
            "distribution": "uniform",
        },
    )
    push_robot = EventTerm(
        func=mdp.push_by_setting_velocity,
        mode="interval",
        interval_range_s=(1.0, 3.0),
        params={"velocity_range": PUSH_VELOCITY_RANGE},
    )


# Rewards


@configclass
class RewardsCfg:
    motion_global_anchor_pos = RewTerm(
        func=mdp.motion_global_anchor_position_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_global_anchor_ori = RewTerm(
        func=mdp.motion_global_anchor_orientation_error_exp,
        weight=0.5,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_pos = RewTerm(
        func=mdp.motion_relative_body_position_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.3},
    )
    motion_body_ori = RewTerm(
        func=mdp.motion_relative_body_orientation_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 0.4},
    )
    motion_body_lin_vel = RewTerm(
        func=mdp.motion_global_body_linear_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 1.0},
    )
    motion_body_ang_vel = RewTerm(
        func=mdp.motion_global_body_angular_velocity_error_exp,
        weight=1.0,
        params={"command_name": "motion", "std": 3.14},
    )
    # Only active while the reference's centre of mass rises fast (a jump's push-off), so they are weighted heavily:
    # the frames are few, and the takeoff impulse decides how high (vertical) and how far (horizontal) the robot gets.
    motion_takeoff_velocity = RewTerm(
        func=mdp.motion_com_takeoff_velocity_exp,
        weight=5.0,
        params={"command_name": "motion", "std": 0.5, "min_reference_rise": 0.3, "axes": "z"},
    )
    motion_takeoff_forward_velocity = RewTerm(
        func=mdp.motion_com_takeoff_velocity_exp,
        weight=5.0,
        params={"command_name": "motion", "std": 0.5, "min_reference_rise": 0.3, "axes": "xy"},
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.1)
    joint_limit = RewTerm(func=mdp.joint_pos_limits, weight=-10.0, params={"asset_cfg": _JOINTS_CFG})
    undesired_contacts = RewTerm(
        func=mdp.undesired_contacts,
        weight=-0.1,
        params={
            "sensor_cfg": SceneEntityCfg(
                "contact_forces", body_names=[r"^(?!" + "$)(?!".join(END_EFFECTORS) + r"$).+$"]
            ),
            "threshold": 1.0,
        },
    )


# Terminations


@configclass
class TerminationsCfg:
    time_out = DoneTerm(func=mdp.time_out, time_out=True)
    anchor_pos = DoneTerm(func=mdp.bad_anchor_pos_z_only, params={"command_name": "motion", "threshold": 0.25})
    anchor_ori = DoneTerm(
        func=mdp.bad_anchor_ori,
        params={"asset_cfg": SceneEntityCfg("robot"), "command_name": "motion", "threshold": 0.8},
    )
    ee_body_pos = DoneTerm(
        func=mdp.bad_motion_body_pos_z_only,
        params={"command_name": "motion", "threshold": 0.25, "body_names": END_EFFECTORS},
    )


# Environment


@configclass
class Asimov1TrackingEnvCfg(ManagerBasedRLEnvCfg):

    scene: Asimov1TrackingSceneCfg = Asimov1TrackingSceneCfg(num_envs=4096, env_spacing=2.5)
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 10.0
        self.sim.dt = 0.005
        self.sim.render_interval = self.decimation
        self.sim.physics_material = self.scene.terrain.physics_material
        self.sim.physx.gpu_max_rigid_patch_count = 10 * 2**15
        self.viewer.eye = (2.5, 2.5, 1.5)
        self.viewer.origin_type = "asset_root"
        self.viewer.asset_name = "robot"


@configclass
class Asimov1TrackingEnvCfg_PLAY(Asimov1TrackingEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = int(1e9)
        self.observations.policy.enable_corruption = False
        self.events.push_robot = None
        self.events.qpos0_rand = None
        self.events.base_com = None
        self.events.pd_gains_rand = None
        self.commands.motion.pose_range = {}
        self.commands.motion.velocity_range = {}
        self.commands.motion.joint_position_range = (0.0, 0.0)
        self.commands.motion.start_from_first_frame = True
        self.commands.motion.debug_vis = True
