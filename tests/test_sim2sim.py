from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from cyclotron.sim2sim import TrackingRun, anchor_orientation_6d, motor_torque_limits, per_joint

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
    run = TrackingRun.load(tmp_path)

    assert run.joint_names == ["left_hip_pitch_joint", "left_knee_joint"]
    np.testing.assert_allclose(run.default_joint_pos, [-0.15, 0.45])
    np.testing.assert_allclose(run.stiffness, [150.0, 150.0])
    assert run.actuator_groups == ["legs", "legs"]
    assert (run.action_scale, run.physics_dt, run.decimation) == (0.25, 0.005, 4)
    assert run.motion_file == "/data/jump.npz"


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
