"""Play a policy in MuJoCo, on Asimov 1's published MJCF (sim2sim).

The policy is the ONNX export of a training run, and the PD gains, effort limits, action processing, timing and
observations come from the run's env.yaml (``params/env.yaml``, or ``env.yaml`` in a shared policy's folder), and so
does the joints' armature, unless ``--mjcf_armature``. The ankle targets are clamped as Asimov 1's firmware does,
unless ``--no_firmware_clamp``.

- A motion tracking run plays the reference motion it was trained on, from its first frame, and reports whether the
  robot falls (training's pelvis height and tilt limits).
- A velocity-interface run (the walking policy, the jump) starts standing in the default pose and follows
  ``--command``, a list of velocity commands and how long each is held. It reports whether the robot falls (the
  firmware's fall trip), how high the pelvis rose and how far the robot travelled.

``--record`` saves the rollout for ./cyclotron.sh --render.
"""

import argparse
import os
import sys
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "cyclotron"))

from cyclotron.sim2sim import DEFAULT_JUMP_COMMANDS, DEFAULT_WALK_COMMANDS  # noqa: E402

DEFAULT_XML = (
    Path(os.environ.get("ASIMOV_1_MODEL_DIR", REPOSITORY_ROOT / "third_party" / "asimov-1" / "sim-model")).expanduser()
    / "xmls"
    / "asimov_1.xml"
)

parser = argparse.ArgumentParser(description="Play a policy in MuJoCo.")
parser.add_argument("run_dir", type=str, help="Training run, or a shared policy's folder with env.yaml and policy.onnx.")
parser.add_argument(
    "--onnx",
    type=str,
    default=None,
    help="Policy to play. Default: <run_dir>/exported/policy.onnx (./cyclotron.sh --export), or <run_dir>/policy.onnx.",
)
parser.add_argument(
    "--command",
    type=str,
    default=None,
    help="Velocity-interface runs: commands and seconds, e.g. '0:1.5,0.3:2,0:2.5' (vx, or vx/vy/yaw_rate).\n"
    f"Default: {DEFAULT_JUMP_COMMANDS} for a run with a reference motion (the jump), else {DEFAULT_WALK_COMMANDS}.",
)
parser.add_argument("--record", type=str, default=None, help="Save the rollout to this .npz, for --render.")
parser.add_argument("--xml", type=str, default=str(DEFAULT_XML), help="MuJoCo model of Asimov 1.")
parser.add_argument("--delay", type=int, default=0, help="Physics steps (5 ms) before a joint target is used.")
parser.add_argument(
    "--motor_speed_ratio",
    type=float,
    default=None,
    help="Model motor torque falling with speed: to zero at this multiple of each actuator's maximum speed (>= 1).",
)
parser.add_argument("--foot_friction", type=float, default=None, help="Sliding friction of the feet (MJCF: 0.6).")
parser.add_argument(
    "--no_firmware_clamp", action="store_true", help="Don't clamp the ankle targets as the firmware does."
)
parser.add_argument(
    "--mjcf_armature",
    action="store_true",
    help="Keep the MJCF's joint armature instead of the run's (the published MJCF predates Menlo's ankle model).",
)


def default_onnx(run_dir: Path) -> Path:
    for path in (run_dir / "exported" / "policy.onnx", run_dir / "policy.onnx"):
        if path.is_file():
            return path
    sys.exit(
        f"[ERROR] No policy in {run_dir}/exported or {run_dir}. Export one with: ./cyclotron.sh --export --checkpoint"
        f" {run_dir}/model_<N>.pt"
    )


def main() -> None:
    args = parser.parse_args()
    try:
        import onnxruntime
    except ImportError:
        sys.exit("[ERROR] sim2sim needs MuJoCo and ONNX Runtime: uv pip install -e 'source/cyclotron[sim2sim]'")
    from cyclotron.sim2sim import (
        FALL_GRAVITY_Z,
        PolicyRun,
        parse_command_schedule,
        play,
        play_velocity,
        projected_gravity_z,
    )

    run_dir = Path(args.run_dir).expanduser()
    onnx_path = Path(args.onnx).expanduser() if args.onnx else default_onnx(run_dir)
    run = PolicyRun.load(run_dir)
    session = onnxruntime.InferenceSession(str(onnx_path))
    name = session.get_inputs()[0].name

    def policy(obs):
        return session.run(None, {name: obs[None]})[0][0]

    options = dict(
        motor_speed_ratio=args.motor_speed_ratio,
        delay=args.delay,
        foot_friction=args.foot_friction,
        firmware_clamp=not args.no_firmware_clamp,
        run_armature=not args.mjcf_armature,
    )
    if run.interface == "tracking":
        rollout = play(run, policy, args.xml, **options)
        what = f"of {Path(run.motion_file).name}"
    else:
        commands = args.command or (DEFAULT_JUMP_COMMANDS if run.motion_file else DEFAULT_WALK_COMMANDS)
        schedule = parse_command_schedule(commands)
        rollout = play_velocity(run, policy, args.xml, schedule, **options)
        what = f"with commands {commands}"

    fell_at = float(rollout["fell_at"])
    duration = (len(rollout["frame"]) - 1) / float(rollout["fps"])
    print(f"[INFO] {onnx_path} in MuJoCo, {duration:.2f} s {what}")
    print(
        f"[INFO] {'Fell at %.2f s' % fell_at if fell_at >= 0 else 'Finished without falling'}; final pelvis height"
        f" {rollout['root_pos'][-1, 2]:.3f} m"
    )
    if run.interface == "velocity":
        # From the end of the first command (the robot standing), to the end.
        start = int(round(schedule[0][0] * float(rollout["fps"])))
        pos = rollout["root_pos"]
        travel = pos[-1, :2] - pos[start, :2]
        print(
            f"[INFO] After the first command: pelvis rose {100 * (pos[start:, 2].max() - pos[start, 2]):.1f} cm at"
            f" most, travelled {100 * travel[0]:+.1f} cm forward and {100 * travel[1]:+.1f} cm sideways"
        )
        tilt = max(projected_gravity_z(quat) for quat in rollout["root_quat"])
        print(f"[INFO] Highest pelvis tilt: projected gravity z {tilt:+.3f} (the firmware's fall trip is above"
              f" {FALL_GRAVITY_Z})")
    if args.record:
        record = Path(args.record).expanduser()
        record.parent.mkdir(parents=True, exist_ok=True)
        np.savez(record, **rollout)
        print(f"[INFO] Recorded {len(rollout['frame'])} frames to {record}")


if __name__ == "__main__":
    main()
