# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause
"""Play a motion tracking policy in MuJoCo, on Asimov-1's published MJCF, without Isaac Sim (sim2sim).

The PD gains, effort limits, default pose, action scale and timing come from the training run's ``env.yaml``. The
policy's observations are built as in ``Asimov1-Tracking``.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from cyclotron.assets.asimov_1_specs import ASIMOV_1_MAX_SPEED, ASIMOV_1_PEAK_TORQUE

POLICY_OBSERVATIONS = ["command", "motion_anchor_ori_b", "base_ang_vel", "joint_pos", "joint_vel", "actions"]
# Training's pelvis terminations: height and tilt (projected gravity) away from the reference.
MAX_PELVIS_HEIGHT_ERROR = 0.25
MAX_PELVIS_TILT_ERROR = 0.8


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
class TrackingRun:
    """What a motion tracking run was trained with, read from its ``params/env.yaml``."""

    joint_names: list[str]
    default_joint_pos: np.ndarray
    stiffness: np.ndarray
    damping: np.ndarray
    effort_limit: np.ndarray
    actuator_groups: list[str]
    action_scale: float
    physics_dt: float
    decimation: int
    motion_file: str

    @classmethod
    def load(cls, run_dir: str | Path) -> TrackingRun:
        cfg = load_env_yaml(Path(run_dir) / "params" / "env.yaml")
        policy = [name for name, term in cfg["observations"]["policy"].items() if isinstance(term, dict)]
        if policy != POLICY_OBSERVATIONS:
            raise ValueError(f"Expected the Asimov1-Tracking policy observations {POLICY_OBSERVATIONS}, got {policy}.")
        action = cfg["actions"]["joint_pos"]
        joints = list(action["joint_names"])
        if not action["use_default_offset"]:
            raise ValueError("Expected joint position actions around the default pose.")
        stiffness, damping, effort = (np.full(len(joints), np.nan) for _ in range(3))
        groups = [""] * len(joints)
        for group, actuator in cfg["scene"]["robot"]["actuators"].items():
            for i, joint in enumerate(joints):
                if any(re.fullmatch(pattern, joint) for pattern in actuator["joint_names_expr"]):
                    stiffness[i], damping[i], effort[i] = (
                        actuator["stiffness"],
                        actuator["damping"],
                        actuator["effort_limit"],
                    )
                    groups[i] = group
        if not all(groups):
            raise ValueError(f"No actuator for joints {[j for j, g in zip(joints, groups) if not g]}.")
        return cls(
            joint_names=joints,
            default_joint_pos=per_joint(cfg["scene"]["robot"]["init_state"]["joint_pos"], joints),
            stiffness=stiffness,
            damping=damping,
            effort_limit=effort,
            actuator_groups=groups,
            action_scale=float(action["scale"]),
            physics_dt=float(cfg["sim"]["dt"]),
            decimation=int(cfg["decimation"]),
            motion_file=cfg["commands"]["motion"]["motion_file"],
        )


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


def projected_gravity_z(quat: np.ndarray) -> float:
    """z of gravity in the frame of a body with orientation ``quat`` (w, x, y, z)."""
    return float(Rotation.from_quat(quat[[1, 2, 3, 0]]).inv().apply([0.0, 0.0, -1.0])[2])


def play(
    run: TrackingRun,
    policy: Callable[[np.ndarray], np.ndarray],
    xml_path: str | Path,
    motor_speed_ratio: float | None = None,
    delay: int = 0,
    foot_friction: float | None = None,
) -> dict[str, np.ndarray]:
    """Play the run's reference motion once in MuJoCo, starting on its first frame.

    Args:
        policy: Observation (124,) to action (23,).
        motor_speed_ratio: Torque falls with speed, to zero at this multiple of each actuator's maximum speed.
        delay: Physics steps between computing a joint target and the PD controller using it.
        foot_friction: Sliding friction of the feet. Their contact spheres take priority over the floor in the MJCF.

    Returns:
        The rollout as recorded by ``play --record``, plus ``fell_at`` (seconds, or -1).
    """
    import mujoco

    motion = np.load(run.motion_file)
    if [str(name) for name in motion["joint_names"]] != run.joint_names:
        raise ValueError(f"{run.motion_file} does not list the policy's joints in the policy's order.")
    fps = 1.0 / (run.physics_dt * run.decimation)
    if abs(float(np.asarray(motion["fps"]).item()) - fps) > 1e-6:
        raise ValueError(f"{run.motion_file} is not at the policy rate of {fps:g} Hz.")
    reference_q, reference_dq = motion["joint_pos"], motion["joint_vel"]
    reference_pos, reference_quat = motion["body_pos_w"][:, 0], motion["body_quat_w"][:, 0]  # pelvis_link

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.timestep = run.physics_dt
    qpos = np.array([model.jnt_qposadr[model.joint(name).id] for name in run.joint_names])
    qvel = np.array([model.jnt_dofadr[model.joint(name).id] for name in run.joint_names])
    if foot_friction is not None:
        for geom in range(model.ngeom):
            if "foot" in (model.geom(geom).name or ""):
                model.geom_friction[geom, 0] = foot_friction
    peak = np.array([ASIMOV_1_PEAK_TORQUE[group] for group in run.actuator_groups])
    no_load = np.array([ASIMOV_1_MAX_SPEED[group] for group in run.actuator_groups]) * (motor_speed_ratio or 1.0)

    data.qpos[0:3], data.qpos[3:7], data.qpos[qpos] = reference_pos[0], reference_quat[0], reference_q[0]
    mujoco.mj_forward(model, data)

    action = np.zeros(len(run.joint_names), dtype=np.float32)
    targets = [run.default_joint_pos.copy()] * (delay + 1)
    log = {
        "root_pos": [data.qpos[0:3].copy()],
        "root_quat": [data.qpos[3:7].copy()],
        "joint_pos": [data.qpos[qpos].copy()],
    }
    fell_at = -1.0
    for frame in range(len(reference_q) - 1):
        obs = np.concatenate(
            [
                reference_q[frame],
                reference_dq[frame],
                anchor_orientation_6d(data.qpos[3:7], reference_quat[frame]),
                data.qvel[3:6],  # free joint angular velocity, in the pelvis frame
                data.qpos[qpos] - run.default_joint_pos,
                data.qvel[qvel],
                action,
            ]
        ).astype(np.float32)
        action = np.asarray(policy(obs), dtype=np.float32)
        target = run.default_joint_pos + run.action_scale * action
        for _ in range(run.decimation):
            targets = targets[1:] + [target]
            torque = run.stiffness * (targets[0] - data.qpos[qpos]) - run.damping * data.qvel[qvel]
            if motor_speed_ratio is None:
                low, high = -run.effort_limit, run.effort_limit
            else:
                low, high = motor_torque_limits(data.qvel[qvel], run.effort_limit, peak, no_load)
            data.qfrc_applied[:] = 0.0
            data.qfrc_applied[qvel] = np.clip(torque, low, high)
            mujoco.mj_step(model, data)
        log["root_pos"].append(data.qpos[0:3].copy())
        log["root_quat"].append(data.qpos[3:7].copy())
        log["joint_pos"].append(data.qpos[qpos].copy())
        reference = frame + 1
        height_error = abs(data.qpos[2] - reference_pos[reference, 2])
        tilt_error = abs(projected_gravity_z(data.qpos[3:7]) - projected_gravity_z(reference_quat[reference]))
        if fell_at < 0 and (height_error > MAX_PELVIS_HEIGHT_ERROR or tilt_error > MAX_PELVIS_TILT_ERROR):
            fell_at = reference / fps

    rollout = {key: np.array(value) for key, value in log.items()}
    rollout["frame"] = np.arange(len(rollout["root_pos"]))
    rollout["fps"] = np.array(fps)
    rollout["joint_names"] = np.array(run.joint_names)
    rollout["fell_at"] = np.array(fell_at)
    return rollout
