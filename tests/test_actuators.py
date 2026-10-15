"""Actuator checks; run pytest after starting Isaac Sim to enable these tests."""

import pytest
import torch

pytest.importorskip("isaaclab.actuators", reason="Requires a running Isaac Sim app", exc_type=ImportError)

from isaaclab.utils.types import ArticulationActions  # noqa: E402

from cyclotron.assets.actuators import DelayedDCMotor, DelayedDCMotorCfg  # noqa: E402
from cyclotron.assets.robots.asimov_1 import asimov_1_dc_motor_actuators  # noqa: E402
from cyclotron.tasks.jump.jump_env_cfg import Asimov1JumpEnvCfg  # noqa: E402


def test_motor_curve_conversion_preserves_firmware_limits():
    original = Asimov1JumpEnvCfg().scene.robot.actuators
    converted = asimov_1_dc_motor_actuators(1.0, original)

    for name, limit in (("ankle_pitch", 0.35), ("ankle_roll", 0.1)):
        assert converted[name].target_limit == original[name].target_limit == limit
        assert converted[name].stiffness == original[name].stiffness
        assert converted[name].min_delay == original[name].min_delay
        assert converted[name].max_delay == original[name].max_delay
    assert converted["knee"].target_limit is None


@pytest.mark.parametrize("target_limit, expected", [(None, 10.0), (0.35, 3.5)])
def test_dc_motor_targets_and_speed_curve(target_limit, expected):
    cfg = DelayedDCMotorCfg(
        joint_names_expr=["joint"],
        stiffness=10.0,
        damping=0.0,
        effort_limit=10.0,
        saturation_effort=10.0,
        no_load_speed=10.0,
        target_limit=target_limit,
        min_delay=0,
        max_delay=0,
    )
    motor = DelayedDCMotor(cfg, ["joint"], slice(None), 2, "cpu")
    motor.reset(None)
    positions = torch.zeros(2, 1)
    targets = torch.tensor([[1.0], [-1.0]])

    def action():
        return ArticulationActions(
            joint_positions=targets.clone(),
            joint_velocities=torch.zeros_like(targets),
            joint_efforts=torch.zeros_like(targets),
        )

    at_rest = motor.compute(action(), positions, positions)
    torch.testing.assert_close(at_rest.joint_efforts, torch.tensor([[expected], [-expected]]))

    # At 80% of no-load speed the curve caps driving torque at 2 N·m in either direction, with the target clamp
    # still applied before the PD controller computes its requested torque.
    moving = motor.compute(action(), positions, targets * 8.0)
    torch.testing.assert_close(motor.computed_effort, torch.tensor([[expected], [-expected]]))
    torch.testing.assert_close(moving.joint_efforts, torch.tensor([[2.0], [-2.0]]))
