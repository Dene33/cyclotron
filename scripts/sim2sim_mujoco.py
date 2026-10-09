"""Play a motion tracking policy in MuJoCo, on Asimov 1's published MJCF (sim2sim).

The policy is the ONNX export of a training run, the reference motion is the one the run was trained on, and the PD
gains, effort limits and timing come from the run's params/env.yaml. Prints whether the robot falls (training's pelvis
height and tilt limits) and can record the rollout for ./cyclotron.sh --render.
"""

import argparse
import os
import sys
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "cyclotron"))

DEFAULT_XML = (
    Path(os.environ.get("ASIMOV_1_MODEL_DIR", REPOSITORY_ROOT / "third_party" / "asimov-1" / "sim-model")).expanduser()
    / "xmls"
    / "asimov_1.xml"
)

parser = argparse.ArgumentParser(description="Play a motion tracking policy in MuJoCo.")
parser.add_argument("run_dir", type=str, help="Training run, e.g. logs/rsl_rl/asimov1_tracking/<run>.")
parser.add_argument(
    "--onnx",
    type=str,
    default=None,
    help="Policy to play. Default: <run_dir>/exported/policy.onnx, which --play writes for the checkpoint it loads.",
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


def main() -> None:
    args = parser.parse_args()
    try:
        import onnxruntime
    except ImportError:
        sys.exit("[ERROR] sim2sim needs MuJoCo and ONNX Runtime: uv pip install -e 'source/cyclotron[sim2sim]'")
    from cyclotron.sim2sim import TrackingRun, play

    run_dir = Path(args.run_dir).expanduser()
    onnx_path = Path(args.onnx).expanduser() if args.onnx else run_dir / "exported" / "policy.onnx"
    if not onnx_path.is_file():
        sys.exit(
            f"[ERROR] No policy at {onnx_path}. Export one with: ./cyclotron.sh --play --task Asimov1-Tracking-Play-v0"
            f" --headless --load_run {run_dir.name} --checkpoint <model_N.pt> --export-only"
        )
    run = TrackingRun.load(run_dir)
    session = onnxruntime.InferenceSession(str(onnx_path))
    name = session.get_inputs()[0].name
    rollout = play(
        run,
        lambda obs: session.run(None, {name: obs[None]})[0][0],
        args.xml,
        motor_speed_ratio=args.motor_speed_ratio,
        delay=args.delay,
        foot_friction=args.foot_friction,
    )

    fell_at = float(rollout["fell_at"])
    duration = (len(rollout["frame"]) - 1) / float(rollout["fps"])
    print(f"[INFO] {onnx_path} in MuJoCo, {duration:.2f} s of {Path(run.motion_file).name}")
    print(
        f"[INFO] {'Fell at %.2f s' % fell_at if fell_at >= 0 else 'Finished without falling'}; final pelvis height"
        f" {rollout['root_pos'][-1, 2]:.3f} m"
    )
    if args.record:
        record = Path(args.record).expanduser()
        record.parent.mkdir(parents=True, exist_ok=True)
        import numpy as np

        np.savez(record, **rollout)
        print(f"[INFO] Recorded {len(rollout['frame'])} frames to {record}")


if __name__ == "__main__":
    main()
