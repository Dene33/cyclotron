from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cyclotron.sim2sim import (
    PolicyRun,
    anchor_orientation_6d,
    motor_torque_limits,
    parse_command_schedule,
    per_joint,
    schedule_commands,
    velocity_observation,
    velocity_observation_terms,
)

MODEL_DIR = Path(__file__).resolve().parents[1] / "third_party" / "asimov-1" / "sim-model"

ENV_YAML = """\
sim:
  dt: 0.005
decimation: 4
scene:
  robot:
    init_state:
      pos: !!python/tuple
      - 0.0
      - 0.0
      - 0.639
      joint_pos:
        left_knee_joint: 0.45
        .*_hip_pitch_joint: -0.15
    actuators:
      legs:
        joint_names_expr:
        - .*_knee_joint
        - .*_hip_pitch_joint
        stiffness: 150.0
        damping: 5.0
        effort_limit: 45.0
        armature: 0.43
actions:
  joint_pos:
    joint_names:
    - left_hip_pitch_joint
    - left_knee_joint
    scale: 0.25
    use_default_offset: true
    clip: null
observations:
  policy:
    concatenate_terms: true
    command: {func: generated_commands}
    motion_anchor_ori_b: {func: motion_anchor_ori_b}
    base_ang_vel: {func: delayed_obs}
    joint_pos: {func: joint_pos_rel, history_length: !!python/object/apply:builtins.slice [null, null, null]}
    joint_vel: {func: joint_vel_rel}
    actions: {func: last_action}
commands:
  motion:
    motion_file: /data/jump.npz
"""


def test_tracking_run_reads_env_yaml(tmp_path):
    (tmp_path / "params").mkdir()
    (tmp_path / "params" / "env.yaml").write_text(ENV_YAML)
    run = PolicyRun.load(tmp_path)

    assert run.joint_names == ["left_hip_pitch_joint", "left_knee_joint"]
    np.testing.assert_allclose(run.default_joint_pos, [-0.15, 0.45])
    np.testing.assert_allclose(run.stiffness, [150.0, 150.0])
    assert run.actuator_groups == ["legs", "legs"]
    np.testing.assert_allclose(run.action_scale, [0.25, 0.25])
    assert (run.physics_dt, run.decimation, run.root_height) == (0.005, 4, 0.639)
    assert run.motion_file == "/data/jump.npz"
    assert run.interface == "tracking" and run.observations == []
    np.testing.assert_allclose(run.armature, [0.43, 0.43])
    np.testing.assert_allclose(run.joint_targets(np.array([1.0, -1.0])), [0.1, 0.2])


# The walking policy's observations and actions as Menlo's env.yaml records them (their package is isaac_asimov).
VELOCITY_ENV_YAML = ENV_YAML.replace(
    """observations:
  policy:
    concatenate_terms: true
    command: {func: generated_commands}
    motion_anchor_ori_b: {func: motion_anchor_ori_b}
    base_ang_vel: {func: delayed_obs}
    joint_pos: {func: joint_pos_rel, history_length: !!python/object/apply:builtins.slice [null, null, null]}
    joint_vel: {func: joint_vel_rel}
    actions: {func: last_action}
commands:
  motion:
    motion_file: /data/jump.npz
""",
    """observations:
  policy:
    concatenate_terms: true
    base_ang_vel:
      func: isaac_asimov.tasks.locomotion.mdp.observations:delayed_obs
      params: {quantity: base_ang_vel, min_lag: 0, max_lag: 1}
      scale: 0.25
      history_length: 0
    projected_gravity:
      func: isaac_asimov.tasks.locomotion.mdp.observations:delayed_obs
      params: {quantity: projected_gravity, min_lag: 0, max_lag: 2}
      scale: null
    command:
      func: isaaclab.envs.mdp.observations:generated_commands
      params: {command_name: twist}
    joint_pos_slot23:
      func: isaaclab.envs.mdp.observations:joint_pos_rel
      params:
        asset_cfg: {name: robot, joint_names: [left_knee_joint]}
      scale: 1.0
    joint_vel_slot01:
      func: isaaclab.envs.mdp.observations:joint_vel_rel
      params:
        asset_cfg: {name: robot, joint_names: [left_hip_pitch_joint, left_knee_joint]}
      scale: 0.1
    actions: {func: isaaclab.envs.mdp.observations:last_action}
commands:
  twist: {}
""",
).replace(
    """    scale: 0.25
    use_default_offset: true
    clip: null""",
    """    scale:
      left_hip_pitch_joint: 0.25
      left_knee_joint: 0.08
    use_default_offset: true
    clip:
      .*_knee_joint: !!python/tuple [-0.09, 0.09]""",
)


def test_velocity_run_reads_the_observation_recipe_and_action_processing(tmp_path):
    (tmp_path / "env.yaml").write_text(VELOCITY_ENV_YAML)  # a shared policy's folder: env.yaml at its top
    run = PolicyRun.load(tmp_path)

    assert run.interface == "velocity" and run.motion_file is None
    assert [(t.name, t.quantity, t.scale, t.joint_ids) for t in run.observations] == [
        ("base_ang_vel", "base_ang_vel", 0.25, None),
        ("projected_gravity", "projected_gravity", 1.0, None),
        ("command", "command", 1.0, None),
        ("joint_pos_slot23", "joint_pos", 1.0, [1]),
        ("joint_vel_slot01", "joint_vel", 0.1, [0, 1]),
        ("actions", "actions", 1.0, None),
    ]
    np.testing.assert_allclose(run.action_scale, [0.25, 0.08])
    # 0.45 + 0.08 * 2 = 0.61 is clipped to 0.09; the hip pitch is not clipped.
    np.testing.assert_allclose(run.joint_targets(np.array([2.0, 2.0])), [0.35, 0.09])

    obs = velocity_observation(
        run,
        ang_vel=np.array([4.0, 0.0, -4.0]),
        gravity=np.array([0.0, 0.0, -1.0]),
        command=np.array([0.3, 0.0, 0.0]),
        joint_pos=np.array([-0.15, 0.5]),
        joint_vel=np.array([1.0, -2.0]),
        action=np.array([0.5, -0.5]),
    )
    np.testing.assert_allclose(
        obs, [1.0, 0.0, -1.0, 0.0, 0.0, -1.0, 0.3, 0.0, 0.0, 0.05, 0.1, -0.2, 0.5, -0.5], atol=1e-6
    )


def test_velocity_terms_refuse_what_the_robot_cannot_measure():
    with pytest.raises(ValueError, match="base_lin_vel"):
        velocity_observation_terms({"base_lin_vel": {"func": "mdp:base_lin_vel"}}, ["left_knee_joint"])
    with pytest.raises(ValueError, match="joint_pos"):  # joints in Isaac Sim's own order
        velocity_observation_terms({"joint_pos": {"func": "mdp:joint_pos_rel"}}, ["left_knee_joint"])


def test_command_schedules():
    schedule = parse_command_schedule("0:0.04,0.3:0.06,0.1/0.2/0.5:0.02")
    assert schedule == [(0.04, (0.0, 0.0, 0.0)), (0.06, (0.3, 0.0, 0.0)), (0.02, (0.1, 0.2, 0.5))]
    np.testing.assert_allclose(schedule_commands(schedule, 0.02)[:, 0], [0.0, 0.0, 0.3, 0.3, 0.3, 0.1])
    with pytest.raises(ValueError):
        parse_command_schedule("0.3/0.1:1")


def test_per_joint_matches_whole_names_and_reports_missing_joints():
    values = per_joint({".*_knee_joint": 1.0, "left_hip_pitch_joint": 2.0}, ["left_knee_joint", "left_hip_pitch_joint"])
    np.testing.assert_allclose(values, [1.0, 2.0])
    with pytest.raises(ValueError, match="left_hip_roll_joint"):
        per_joint({"left_hip": 1.0}, ["left_hip_roll_joint"])


def test_motor_torque_limits_follow_the_dc_motor_curve():
    effort, peak, no_load = np.array([45.0]), np.array([75.0]), np.array([12.0])

    low, high = motor_torque_limits(np.array([0.0]), effort, peak, no_load)
    np.testing.assert_allclose([low[0], high[0]], [-45.0, 45.0])  # capped at standstill
    low, high = motor_torque_limits(np.array([9.0]), effort, peak, no_load)
    np.testing.assert_allclose([low[0], high[0]], [-45.0, 75.0 * (1 - 9 / 12)])  # driving loses torque, braking not
    low, high = motor_torque_limits(np.array([-9.0]), effort, peak, no_load)
    np.testing.assert_allclose([low[0], high[0]], [-75.0 * (1 - 9 / 12), 45.0])


def test_anchor_orientation_is_the_reference_in_the_robot_frame():
    identity = np.array([1.0, 0.0, 0.0, 0.0])
    np.testing.assert_allclose(anchor_orientation_6d(identity, identity), [1, 0, 0, 1, 0, 0], atol=1e-12)

    # Robot turned 90 degrees left, reference straight: the reference points to the robot's right.
    robot = Rotation.from_euler("z", 90, degrees=True).as_quat()[[3, 0, 1, 2]]
    np.testing.assert_allclose(anchor_orientation_6d(robot, identity), [0, 1, -1, 0, 0, 0], atol=1e-12)


@pytest.mark.skipif(not (MODEL_DIR / "xmls" / "asimov_1.xml").is_file(), reason="Asimov-1 model not checked out")
def test_mujoco_model_has_the_urdf_kinematics():
    mujoco = pytest.importorskip("mujoco")
    from cyclotron.render import RobotModel

    model = mujoco.MjModel.from_xml_path(str(MODEL_DIR / "xmls" / "asimov_1.xml"))
    data = mujoco.MjData(model)
    urdf = RobotModel(str(MODEL_DIR / "urdf" / "asimov_1.urdf"), cell=0.05)
    rng = np.random.default_rng(0)
    joints = {}
    for j in range(1, model.njnt):  # 0 is the floating base
        low, high = model.jnt_range[j]
        joints[model.joint(j).name] = data.qpos[model.jnt_qposadr[j]] = rng.uniform(low, high)
    root_pos, root_rot = np.array([0.1, -0.2, 0.7]), Rotation.from_euler("xyz", [0.1, -0.2, 0.5])
    data.qpos[0:3], data.qpos[3:7] = root_pos, root_rot.as_quat()[[3, 0, 1, 2]]
    mujoco.mj_kinematics(model, data)

    poses = urdf.forward_kinematics(root_pos, root_rot, joints)
    for body in ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link"):
        np.testing.assert_allclose(data.body(body).xpos, poses[body][0], atol=1e-6)
