"""Standing in an ideal simulator: Module 1 of the RL crash course.

The velocity task with everything a standing policy does not need taken out,
in the order Module 1 teaches the blocks: the robot, what it senses, what it
does, the reward, and the episode.
"""

from isaaclab.utils import configclass

from cyclotron.tasks.locomotion.velocity_env_cfg import Asimov1VelocityEnvCfg


@configclass
class Asimov1StandIdealEnvCfg(Asimov1VelocityEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 1. The robot: flat ground, motors without delay.
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        for actuator in self.scene.robot.actuators.values():
            actuator.min_delay = 0
            actuator.max_delay = 0

        # 2. What it senses: the same 78 values as the walking policy, so the
        # result runs in the browser, but without sensor noise or lag.
        self.observations.policy.enable_corruption = False
        self.observations.policy.base_ang_vel.params["max_lag"] = 0
        self.observations.policy.projected_gravity.params["max_lag"] = 0

        # 3. What it is asked to do: always stand still.
        self.commands.twist.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.twist.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.twist.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.twist.rel_standing_envs = 1.0
        self.commands.twist.heading_command = False

        # 4. The reward: drop the terms that only score walking.
        self.rewards.air_time = None
        self.rewards.foot_clearance = None
        self.rewards.foot_swing_height = None
        self.rewards.heading_drift = None

        # 5. No pushes and no randomised physics. The starting pose and joint
        # offsets stay random: doing nothing already stands, so the policy
        # must start from poses that need balancing.
        self.events.push_robot = None
        self.events.foot_friction = None
        self.events.qpos0_rand = None
        self.events.base_com = None
        self.events.pd_gains_rand = None


@configclass
class Asimov1StandIdealEnvCfg_PLAY(Asimov1StandIdealEnvCfg):
    def __post_init__(self):
        super().__post_init__()
        self.scene.num_envs = 16
        self.episode_length_s = int(1e9)
        self.events.reset_robot_joints.params["position_range"] = (0.0, 0.0)
        self.events.reset_robot_joints.params["velocity_range"] = (0.0, 0.0)
        self.events.reset_base.params["pose_range"] = {k: (0.0, 0.0) for k in self.events.reset_base.params["pose_range"]}
        self.events.reset_base.params["velocity_range"] = {k: (0.0, 0.0) for k in self.events.reset_base.params["velocity_range"]}


# Ablations for the RL crash course, Module 1, lesson 6: the same task with one
# piece left out, trained with the same seed and iterations as the reference.


@configclass
class Asimov1StandIdealNoPoseEnvCfg(Asimov1StandIdealEnvCfg):
    """Without the pose reward: nothing says what a good stance looks like."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.pose = None


@configclass
class Asimov1StandIdealNoActionRateEnvCfg(Asimov1StandIdealEnvCfg):
    """Without the action-rate penalty: nothing charges for jittery targets."""

    def __post_init__(self):
        super().__post_init__()
        self.rewards.action_rate_l2 = None


@configclass
class Asimov1StandIdealNoFallEndEnvCfg(Asimov1StandIdealEnvCfg):
    """Without episodes ending on a fall: a fallen robot lies there until the 20 s timeout."""

    def __post_init__(self):
        super().__post_init__()
        self.terminations.fell_over = None
