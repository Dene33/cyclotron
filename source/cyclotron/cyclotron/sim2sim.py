# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Play a policy in MuJoCo, on Asimov-1's published MJCF, without Isaac Sim (sim2sim).

The PD gains, effort limits, default pose, action scales and clips, timing and the policy's observations come from
the training run's ``env.yaml``. Two policy interfaces are supported:

- ``tracking`` (``Asimov1-Tracking``): the policy reads a reference motion, frame by frame.
- ``velocity`` (``Asimov1-Velocity``, ``Asimov1-Jump``): the walking policy's interface, a velocity command and what
  the robot measures, in the term order, joint order and scales the run's ``env.yaml`` records.

The robot clamps ankle targets as Asimov 1's firmware does (``ASIMOV_1_FIRMWARE_TARGET_LIMITS``), and its joints get
the run's armature (the motors' inertia as the joint sees it), which the training actuators set: the published MJCF
predates Menlo's ankle model, whose ankle pitch has 7.6 times the armature.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from cyclotron.assets.asimov_1_specs import ASIMOV_1_FIRMWARE_TARGET_LIMITS, ASIMOV_1_MAX_SPEED, ASIMOV_1_PEAK_TORQUE

TRACKING_OBSERVATIONS = ["command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel", "actions"]
# Training's pelvis terminations for tracking: height and tilt (projected gravity) away from the reference.
MAX_PELVIS_HEIGHT_ERROR = 0.25
MAX_PELVIS_TILT_ERROR = 0.8
# Without a reference: the firmware's fall trip (projected gravity z above this), or a pelvis this low.
FALL_GRAVITY_Z = -0.5
MIN_PELVIS_HEIGHT = 0.35
# Velocity commands (vx or vx/vy/yaw rate, then seconds) to play a velocity-interface policy with: stand, jump (the
# command held through the jump), stand; or stand, walk, stand.
DEFAULT_JUMP_COMMANDS = "0:1.5,0.3:2.0,0:2.5"
DEFAULT_WALK_COMMANDS = "0:1.0,0.3:4.0,0:1.0"


class _EnvYamlLoader(yaml.SafeLoader):
    """Reads Isaac Lab's env.yaml safely: tuples as tuples, other Python objects as placeholder strings."""


_EnvYamlLoader.add_constructor(
    "tag:yaml.org,2002:python/tuple", lambda loader, node: tuple(loader.construct_sequence(node))
)
_EnvYamlLoader.add_multi_constructor("tag:yaml.org,2002:python/", lambda loader, suffix, node: f"<{suffix}>")


def load_env_yaml(path: str | Path) -> dict:
    """A run's ``params/env.yaml``, read safely: Python tuples as tuples, other Python objects as placeholders."""
    with open(path) as f:
        return yaml.load(f, Loader=_EnvYamlLoader)


def per_joint(values: dict[str, float], joint_names: list[str]) -> np.ndarray:
    """Resolve an Isaac Lab ``{joint name regex: value}`` dict over ``joint_names``."""
    out = np.full(len(joint_names), np.nan)
    for pattern, value in values.items():
        for i, name in enumerate(joint_names):
            if re.fullmatch(pattern, name):
                out[i] = value
    missing = [name for name, value in zip(joint_names, out) if np.isnan(value)]
    if missing:
        raise ValueError(f"No value for joints {missing} in {values}.")
    return out


@dataclass
class ObservationTerm:
    """One term of a velocity-interface policy's observation."""

    name: str
    quantity: str
    """``base_ang_vel`` and ``projected_gravity`` (pelvis frame), ``command`` (forward and sideways speed, yaw rate),
    ``joint_pos`` (relative to the default pose), ``joint_vel`` or ``actions``."""
    scale: float = 1.0
    joint_ids: list[int] | None = None
    """For joint terms: which of the policy's joints, in order."""


# Observation functions a velocity-interface policy may use, by the quantity they read.
_QUANTITIES = {
    "base_ang_vel": "base_ang_vel",
    "projected_gravity": "projected_gravity",
    "generated_commands": "command",
    "joint_pos_rel": "joint_pos",
    "joint_vel_rel": "joint_vel",
    "last_action": "actions",
}


def velocity_observation_terms(policy_cfg: dict, joint_names: list[str]) -> list[ObservationTerm]:
    """The observation terms of a velocity-interface policy, from its env.yaml's ``observations.policy``. Raises
    ``ValueError`` for terms that can't be built from what the robot measures and a velocity command."""
    terms, unsupported = [], []
    for name, term in policy_cfg.items():
        if not isinstance(term, dict):
            continue
        func = str(term.get("func", "")).split(":")[-1]
        params = term.get("params") or {}
        if func == "delayed_obs":  # a sensor delay in training; on the robot, the delay is the robot's own
            func = str(params.get("quantity", ""))
        quantity = _QUANTITIES.get(func)
        if quantity is None or term.get("history_length") or term.get("clip") is not None:
            unsupported.append(name)
            continue
        joint_ids = None
        if quantity in ("joint_pos", "joint_vel"):
            asset = params.get("asset_cfg")
            names = asset.get("joint_names") if isinstance(asset, dict) else None
            if not isinstance(names, list) or any(joint not in joint_names for joint in names):
                unsupported.append(name)  # joints in the simulator's own order, which only Isaac Sim knows
                continue
            joint_ids = [joint_names.index(joint) for joint in names]
        scale = term.get("scale")
        terms.append(ObservationTerm(name, quantity, 1.0 if scale is None else float(scale), joint_ids))
    if unsupported:
        raise ValueError(f"Can't build the observation terms {unsupported} from what the robot measures.")
    return terms


def env_yaml_path(run_dir: str | Path) -> Path:
    """A run's ``params/env.yaml``, or the ``env.yaml`` of a shared policy's folder."""
    for path in (Path(run_dir) / "params" / "env.yaml", Path(run_dir) / "env.yaml"):
        if path.is_file():
            return path
    raise FileNotFoundError(f"No params/env.yaml or env.yaml in {run_dir}.")


@dataclass
class PolicyRun:
    """What a run was trained with, read from its ``env.yaml``: the robot, the actions and the policy's interface."""

    joint_names: list[str]
    default_joint_pos: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limit: np.ndarray
    armature: np.ndarray
    """Per joint, NaN where the run's actuators don't set it."""
    actuator_groups: list[str]
    action_scale: np.ndarray
    action_clip: np.ndarray
    """(joints, 2): the targets' lower and upper clip, infinite where not clipped."""
    physics_dt: float
    decimation: int
    root_height: float
    interface: str
    """``tracking`` or ``velocity`` (see the module docstring)."""
    observations: list[ObservationTerm]
    """The velocity interface's observation terms; empty for tracking."""
    motion_file: str | None

    @classmethod
    def load(cls, run_dir: str | Path) -> PolicyRun:
        cfg = load_env_yaml(env_yaml_path(run_dir))
        action = cfg["actions"]["joint_pos"]
        joints = list(action["joint_names"])
        if not action["use_default_offset"]:
            raise ValueError("Expected joint position actions around the default pose.")
        stiffness, damping, effort, armature = (np.full(len(joints), np.nan) for _ in range(4))
        groups = [""] * len(joints)
        for group, actuator in cfg["scene"]["robot"]["actuators"].items():
            for i, joint in enumerate(joints):
                if any(re.fullmatch(pattern, joint) for pattern in actuator["joint_names_expr"]):
                    stiffness[i], damping[i], effort[i] = (
                        actuator["stiffness"],
                        actuator["damping"],
                        actuator["effort_limit"],
                    )
                    armature[i] = actuator.get("armature") if actuator.get("armature") is not None else np.nan
                    groups[i] = group
        if not all(groups):
            raise ValueError(f"No actuator for joints {[j for j, g in zip(joints, groups) if not g]}.")
        scale = action["scale"]
        scale = per_joint(scale, joints) if isinstance(scale, dict) else np.full(len(joints), float(scale))
        clip = np.tile([-np.inf, np.inf], (len(joints), 1))
        for pattern, (low, high) in (action.get("clip") or {}).items():
            for i, joint in enumerate(joints):
                if re.fullmatch(pattern, joint):
                    clip[i] = (-np.inf if low is None else low, np.inf if high is None else high)

        policy = cfg["observations"]["policy"]
        names = [name for name, term in policy.items() if isinstance(term, dict)]
        if names == TRACKING_OBSERVATIONS:
            interface, observations = "tracking", []
        else:
            interface, observations = "velocity", velocity_observation_terms(policy, joints)
        motion = (cfg.get("commands") or {}).get("motion")
        return cls(
            joint_names=joints,
            default_joint_pos=per_joint(cfg["scene"]["robot"]["init_state"]["joint_pos"], joints),
            stiffness=stiffness,
            damping=damping,
            effort_limit=effort,
            armature=armature,
            actuator_groups=groups,
            action_scale=scale,
            action_clip=clip,
            physics_dt=float(cfg["sim"]["dt"]),
            decimation=int(cfg["decimation"]),
            root_height=float(cfg["scene"]["robot"]["init_state"]["pos"][2]),
            interface=interface,
            observations=observations,
            motion_file=motion.get("motion_file") if isinstance(motion, dict) else None,
        )

    @property
    def policy_dt(self) -> float:
        return self.physics_dt * self.decimation

    def joint_targets(self, action: np.ndarray) -> np.ndarray:
        """Joint position targets for a policy action: around the default pose, scaled, then clipped."""
        target = self.default_joint_pos + self.action_scale * action
        return np.clip(target, self.action_clip[:, 0], self.action_clip[:, 1])

    def firmware_target_limits(self) -> np.ndarray:
        """The largest target the firmware drives each joint to, either way (infinite where it doesn't clamp)."""
        return np.array([ASIMOV_1_FIRMWARE_TARGET_LIMITS.get(group, np.inf) for group in self.actuator_groups])


def motor_torque_limits(
    joint_vel: np.ndarray, effort_limit: np.ndarray, peak_torque: np.ndarray, no_load_speed: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Lowest and highest torque of DC motors at ``joint_vel``: falling linearly from ``peak_torque`` at standstill to
    zero at ``no_load_speed``, capped at ``effort_limit`` (as Isaac Lab's ``DCMotor``)."""
    corner = no_load_speed * (1.0 + effort_limit / peak_torque)
    joint_vel = np.clip(joint_vel, -corner, corner)
    high = np.minimum(peak_torque * (1.0 - joint_vel / no_load_speed), effort_limit)
    low = np.maximum(peak_torque * (-1.0 - joint_vel / no_load_speed), -effort_limit)
    return low, high


def anchor_orientation_6d(robot_quat: np.ndarray, reference_quat: np.ndarray) -> np.ndarray:
    """Reference pelvis orientation in the robot's pelvis frame, as the first two columns of its rotation matrix
    (``motion_anchor_ori_b``). Quaternions are (w, x, y, z)."""
    relative = Rotation.from_quat(robot_quat[[1, 2, 3, 0]]).inv() * Rotation.from_quat(reference_quat[[1, 2, 3, 0]])
    return relative.as_matrix()[:, :2].reshape(-1)


def projected_gravity(quat: np.ndarray) -> np.ndarray:
    """Gravity's direction in the frame of a body with orientation ``quat`` (w, x, y, z)."""
    return Rotation.from_quat(quat[[1, 2, 3, 0]]).inv().apply([0.0, 0.0, -1.0])


def projected_gravity_z(quat: np.ndarray) -> float:
    """z of gravity in the frame of a body with orientation ``quat`` (w, x, y, z)."""
    return float(projected_gravity(quat)[2])


def load_reference(run: PolicyRun) -> dict[str, np.ndarray]:
    """The run's reference motion at the policy rate: joint positions and velocities in the policy's joint order, and
    the pelvis position and orientation (w, x, y, z) in the world."""
    if not run.motion_file:
        raise ValueError("The run has no reference motion.")
    motion = np.load(run.motion_file)
    if [str(name) for name in motion["joint_names"]] != run.joint_names:
        raise ValueError(f"{run.motion_file} does not list the policy's joints in the policy's order.")
    if abs(float(np.asarray(motion["fps"]).item()) - 1.0 / run.policy_dt) > 1e-6:
        raise ValueError(f"{run.motion_file} is not at the policy rate of {1.0 / run.policy_dt:g} Hz.")
    return {
        "joint_pos": motion["joint_pos"],
        "joint_vel": motion["joint_vel"],
        "pelvis_pos": motion["body_pos_w"][:, 0],  # pelvis_link
        "pelvis_quat": motion["body_quat_w"][:, 0],
    }


def tracking_observation(
    reference_q: np.ndarray,
    reference_dq: np.ndarray,
    reference_quat: np.ndarray,
    robot_quat: np.ndarray,
    ang_vel: np.ndarray,
    joint_pos_rel: np.ndarray,
    joint_vel: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    """The ``Asimov1-Tracking`` policy observation (``TRACKING_OBSERVATIONS``) from one reference frame and what the
    robot measures: its pelvis orientation (w, x, y, z), the pelvis angular velocity in the pelvis frame, joint
    positions relative to the default pose, joint velocities and the last action."""
    return np.concatenate(
        [
            reference_q,
            reference_dq,
            anchor_orientation_6d(robot_quat, reference_quat),
            ang_vel,
            joint_pos_rel,
            joint_vel,
            action,
        ]
    ).astype(np.float32)


def velocity_observation(
    run: PolicyRun,
    ang_vel: np.ndarray,
    gravity: np.ndarray,
    command: np.ndarray,
    joint_pos: np.ndarray,
    joint_vel: np.ndarray,
    action: np.ndarray,
) -> np.ndarray:
    """A velocity-interface policy's observation, built as the run's env.yaml records, from the pelvis angular
    velocity and gravity's direction (pelvis frame), the velocity command, the joint positions and velocities (in the
    policy's joint order) and the last action."""
    values = {
        "base_ang_vel": np.asarray(ang_vel),
        "projected_gravity": np.asarray(gravity),
        "command": np.asarray(command),
        "joint_pos": np.asarray(joint_pos) - run.default_joint_pos,
        "joint_vel": np.asarray(joint_vel),
        "actions": np.asarray(action),
    }
    parts = []
    for term in run.observations:
        value = values[term.quantity]
        parts.append(term.scale * (value[term.joint_ids] if term.joint_ids is not None else value))
    return np.concatenate(parts).astype(np.float32)


def parse_command_schedule(text: str) -> list[tuple[float, tuple[float, float, float]]]:
    """``"0:1.5,0.3:2,0:2"``: a velocity command and how long to hold it (s), in turn. A command is a forward speed,
    or ``vx/vy/yaw_rate``."""
    schedule = []
    for part in text.split(","):
        command, seconds = part.split(":")
        values = [float(value) for value in command.split("/")]
        if len(values) not in (1, 3) or float(seconds) <= 0:
            raise ValueError(f"Expected <vx>:<seconds> or <vx>/<vy>/<yaw rate>:<seconds>, got {part!r}.")
        schedule.append((float(seconds), tuple(values + [0.0, 0.0]) if len(values) == 1 else tuple(values)))
    return schedule


def schedule_commands(schedule: Sequence[tuple[float, tuple[float, float, float]]], dt: float) -> np.ndarray:
    """The command at every policy step of ``schedule``, (steps, 3)."""
    return np.concatenate([np.tile(command, (int(round(seconds / dt)), 1)) for seconds, command in schedule])


class _MujocoRobot:
    """Asimov 1 in MuJoCo, driven by joint position targets through the run's PD gains and torque limits."""

    def __init__(
        self,
        run: PolicyRun,
        xml_path: str | Path,
        motor_speed_ratio: float | None,
        delay: int,
        foot_friction: float | None,
        firmware_clamp: bool,
        run_armature: bool = True,
    ):
        import mujoco

        self.mujoco, self.run = mujoco, run
        self.model = mujoco.MjModel.from_xml_path(str(xml_path))
        self.data = mujoco.MjData(self.model)
        self.model.opt.timestep = run.physics_dt
        self.qpos = np.array([self.model.jnt_qposadr[self.model.joint(name).id] for name in run.joint_names])
        self.qvel = np.array([self.model.jnt_dofadr[self.model.joint(name).id] for name in run.joint_names])
        if foot_friction is not None:
            for geom in range(self.model.ngeom):
                if "foot" in (self.model.geom(geom).name or ""):
                    self.model.geom_friction[geom, 0] = foot_friction
        self.motor_speed_ratio = motor_speed_ratio
        self.peak = np.array([ASIMOV_1_PEAK_TORQUE[group] for group in run.actuator_groups])
        self.no_load = np.array([ASIMOV_1_MAX_SPEED[group] for group in run.actuator_groups]) * (motor_speed_ratio or 1.0)
        self.target_limit = run.firmware_target_limits() if firmware_clamp else np.full(len(run.joint_names), np.inf)
        if run_armature:
            known = ~np.isnan(run.armature)
            self.model.dof_armature[self.qvel[known]] = run.armature[known]
        self.targets = [run.default_joint_pos.copy()] * (delay + 1)
        self.log = {"root_pos": [], "root_quat": [], "joint_pos": []}

    def place(self, root_pos: np.ndarray, root_quat: np.ndarray, joint_pos: np.ndarray):
        self.data.qpos[0:3], self.data.qpos[3:7], self.data.qpos[self.qpos] = root_pos, root_quat, joint_pos
        self.mujoco.mj_forward(self.model, self.data)
        self.record()

    def record(self):
        self.log["root_pos"].append(self.data.qpos[0:3].copy())
        self.log["root_quat"].append(self.data.qpos[3:7].copy())
        self.log["joint_pos"].append(self.data.qpos[self.qpos].copy())

    def step(self, target: np.ndarray):
        """One policy step: ``decimation`` physics steps towards ``target``."""
        run, data = self.run, self.data
        target = np.clip(target, -self.target_limit, self.target_limit)
        for _ in range(run.decimation):
            self.targets = self.targets[1:] + [target]
            torque = run.stiffness * (self.targets[0] - data.qpos[self.qpos]) - run.damping * data.qvel[self.qvel]
            if self.motor_speed_ratio is None:
                low, high = -run.effort_limit, run.effort_limit
            else:
                low, high = motor_torque_limits(data.qvel[self.qvel], run.effort_limit, self.peak, self.no_load)
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[self.qvel] = np.clip(torque, low, high)
            self.mujoco.mj_step(self.model, data)
        self.record()

    def rollout(self, fell_at: float, **extra) -> dict[str, np.ndarray]:
        rollout = {key: np.array(value) for key, value in self.log.items()}
        rollout["frame"] = np.arange(len(rollout["root_pos"]))
        rollout["fps"] = np.array(1.0 / self.run.policy_dt)
        rollout["joint_names"] = np.array(self.run.joint_names)
        rollout["fell_at"] = np.array(fell_at)
        rollout.update(extra)
        return rollout


def play(
    run: PolicyRun,
    policy: Callable[[np.ndarray], np.ndarray],
    xml_path: str | Path,
    motor_speed_ratio: float | None = None,
    delay: int = 0,
    foot_friction: float | None = None,
    firmware_clamp: bool = True,
    run_armature: bool = True,
) -> dict[str, np.ndarray]:
    """Play a tracking run's reference motion once in MuJoCo, starting on its first frame.

    Args:
        policy: Observation (124,) to action (23,).
        motor_speed_ratio: Torque falls with speed, to zero at this multiple of each actuator's maximum speed.
        delay: Physics steps between computing a joint target and the PD controller using it.
        foot_friction: Sliding friction of the feet. Their contact spheres take priority over the floor in the MJCF.
        firmware_clamp: Clamp the ankle targets as Asimov 1's firmware does.
        run_armature: Give the joints the run's armature rather than the MJCF's.

    Returns:
        The rollout as recorded by ``play --record``, plus ``fell_at`` (seconds, or -1).
    """
    reference = load_reference(run)
    reference_q, reference_dq = reference["joint_pos"], reference["joint_vel"]
    reference_pos, reference_quat = reference["pelvis_pos"], reference["pelvis_quat"]
    robot = _MujocoRobot(run, xml_path, motor_speed_ratio, delay, foot_friction, firmware_clamp, run_armature)
    data, qpos, qvel = robot.data, robot.qpos, robot.qvel
    robot.place(reference_pos[0], reference_quat[0], reference_q[0])

    action = np.zeros(len(run.joint_names), dtype=np.float32)
    fell_at = -1.0
    for frame in range(len(reference_q) - 1):
        obs = tracking_observation(
            reference_q[frame],
            reference_dq[frame],
            reference_quat[frame],
            data.qpos[3:7],
            data.qvel[3:6],  # free joint angular velocity, in the pelvis frame
            data.qpos[qpos] - run.default_joint_pos,
            data.qvel[qvel],
            action,
        )
        action = np.asarray(policy(obs), dtype=np.float32)
        robot.step(run.joint_targets(action))
        reference = frame + 1
        height_error = abs(data.qpos[2] - reference_pos[reference, 2])
        tilt_error = abs(projected_gravity_z(data.qpos[3:7]) - projected_gravity_z(reference_quat[reference]))
        if fell_at < 0 and (height_error > MAX_PELVIS_HEIGHT_ERROR or tilt_error > MAX_PELVIS_TILT_ERROR):
            fell_at = reference * run.policy_dt
    return robot.rollout(fell_at)


def play_velocity(
    run: PolicyRun,
    policy: Callable[[np.ndarray], np.ndarray],
    xml_path: str | Path,
    schedule: Sequence[tuple[float, tuple[float, float, float]]],
    motor_speed_ratio: float | None = None,
    delay: int = 0,
    foot_friction: float | None = None,
    firmware_clamp: bool = True,
    run_armature: bool = True,
) -> dict[str, np.ndarray]:
    """Play a velocity-interface policy in MuJoCo, from standing in the default pose, through ``schedule``: velocity
    commands (forward, sideways, yaw rate) and how long each is held (s).

    Returns:
        The rollout as recorded by ``play --record``, its ``command`` at every step, and ``fell_at`` (seconds, or -1):
        when the pelvis tilted past the firmware's fall trip or dropped below ``MIN_PELVIS_HEIGHT``.
    """
    commands = schedule_commands(schedule, run.policy_dt)
    robot = _MujocoRobot(run, xml_path, motor_speed_ratio, delay, foot_friction, firmware_clamp, run_armature)
    data, qpos, qvel = robot.data, robot.qpos, robot.qvel
    robot.place(np.array([0.0, 0.0, run.root_height]), np.array([1.0, 0.0, 0.0, 0.0]), run.default_joint_pos)

    action = np.zeros(len(run.joint_names), dtype=np.float32)
    fell_at = -1.0
    for step, command in enumerate(commands):
        obs = velocity_observation(
            run,
            data.qvel[3:6],  # free joint angular velocity, in the pelvis frame
            projected_gravity(data.qpos[3:7]),
            command,
            data.qpos[qpos],
            data.qvel[qvel],
            action,
        )
        action = np.asarray(policy(obs), dtype=np.float32)
        robot.step(run.joint_targets(action))
        if fell_at < 0 and (projected_gravity_z(data.qpos[3:7]) > FALL_GRAVITY_Z or data.qpos[2] < MIN_PELVIS_HEIGHT):
            fell_at = (step + 1) * run.policy_dt
    return robot.rollout(fell_at, command=np.concatenate([commands[:1], commands]))
