from __future__ import annotations

import json

import pytest

from cyclotron.deploy_contract import contract_differences, read_contract

WALKING = {
    "deploy_metadata_version": 1,
    "obs_dim": 78,
    "action_dim": 2,
    "observation_names": ["base_ang_vel", "command"],
    "joint_names": ["left_knee_joint", "left_ankle_roll_joint"],
    "action_scale": [0.25, 0.07999999821186066],
    "action_offset": [0.45, 0.0],
    "raw_action_clip": None,
    "action_clip": [[None, None], [-0.09000000357627869, 0.09000000357627869]],
    "joint_stiffness": [150.0, 35.20000076293945],
    "joint_damping": [5.0, 1.600000023841858],
    "sim_dt": 0.005,
    "decimation": 4,
    "policy_rate_hz": 50.0,
    "trained_commit": "abc",
}


def test_same_contract_ignores_traceability_keys_and_float32_rounding():
    jump = dict(WALKING, trained_commit="def", action_scale=[0.25, 0.08], joint_stiffness=[150.0, 35.2])
    assert contract_differences(WALKING, jump) == []


def test_contract_differences_name_each_key():
    jump = dict(WALKING, obs_dim=124, joint_stiffness=[150.0, 110.0], action_clip=None)
    jump.pop("decimation")
    differences = contract_differences(WALKING, jump)
    assert [line.split(":")[0] for line in differences] == ["obs_dim", "action_clip", "joint_stiffness", "decimation"]
    assert differences[-1] == "decimation: only in the first"


def test_read_contract_decodes_json_values(tmp_path):
    onnx = pytest.importorskip("onnx")
    model = onnx.helper.make_model(onnx.helper.make_graph([], "empty", [], []))
    for key, value in WALKING.items():
        model.metadata_props.add(key=key, value=json.dumps(value) if not isinstance(value, str) else value)
    onnx.save(model, tmp_path / "policy.onnx")

    assert read_contract(str(tmp_path / "policy.onnx")) == WALKING
