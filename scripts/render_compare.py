# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Render a reference motion and policy rollouts side by side, with an orthographic camera, to a video.

Robots are drawn with Asimov-1's URDF meshes, posed from the recorded root pose and joint angles, so this needs no
Isaac Sim. Record rollouts with ``./cyclotron.sh --play ... --record rollout.npz``. The reference stands furthest
back-left, the rollouts follow in the order given, the last one front-right; a robot that travels further moves away
from the others, so list rollouts from the one that travels furthest.

.. code-block:: bash

    ./cyclotron.sh --render --motion motions/forward_jump.npz \\
        --rollout "run1.npz:1 · Tracking" --rollout "run2.npz:2 · + Lift" --out compare.mp4
"""

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "cyclotron"))

from cyclotron.render import Actor, RobotModel, View, fit_view, render_frame  # noqa: E402

DEFAULT_URDF = (
    Path(os.environ.get("ASIMOV_1_MODEL_DIR", REPOSITORY_ROOT / "third_party" / "asimov-1" / "sim-model")).expanduser()
    / "urdf"
    / "asimov_1.urdf"
)
REFERENCE_COLOR, REFERENCE_LABEL_COLOR = (0.93, 0.66, 0.30), "#8a5300"
ROLLOUT_COLORS = [
    (0.55, 0.68, 0.85),
    (0.50, 0.78, 0.55),
    (0.75, 0.60, 0.85),
    (0.88, 0.52, 0.52),
    (0.50, 0.75, 0.78),
    (0.72, 0.74, 0.42),
]
ROLLOUT_LABEL_COLORS = ["#2f4f7a", "#2e6b3a", "#5b3f7a", "#8a2f2f", "#2b6466", "#565a12"]
LABEL_HEIGHT = 1.72  # m, above every robot's head

parser = argparse.ArgumentParser(description="Render a reference motion and policy rollouts side by side.")
parser.add_argument("--motion", type=str, default=None, help="Reference motion file (.npz from --csv-to-npz).")
parser.add_argument(
    "--rollout",
    action="append",
    default=[],
    metavar="FILE.npz:LABEL",
    help="Rollout recorded with --play --record, and its label. Repeat for several.",
)
parser.add_argument("--out", type=str, required=True, help="Output video (.mp4), or image (.png) with --preview.")
parser.add_argument("--preview", type=float, default=None, help="Render one frame at this time (s) to a PNG instead.")
parser.add_argument("--title", type=str, default="", help="Text in the top right corner.")
parser.add_argument("--caption", type=str, default="", help="One line of text along the bottom.")
parser.add_argument("--offset", type=float, default=1.0, help="Sideways gap between the robots (m).")
parser.add_argument("--elev", type=float, default=25.0, help="Camera elevation (deg).")
parser.add_argument("--azim", type=float, default=45.0, help="Camera angle from the direction of travel (deg).")
parser.add_argument("--label_size", type=float, default=13.0)
parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF))
parser.add_argument("--workers", type=int, default=max(1, min(6, (os.cpu_count() or 2) // 2)))


def load_actors(args) -> tuple[list[Actor], np.ndarray, float, np.ndarray]:
    """Actors laid out side by side, the reference frame shown at each video frame, the frame rate, and the
    direction of travel."""
    rollouts = []
    for spec in args.rollout:
        path, _, label = spec.partition(":")
        rollouts.append((np.load(path), label or Path(path).stem))
    if not rollouts and not args.motion:
        parser.error("Give --motion, at least one --rollout, or both.")

    reference = np.load(args.motion) if args.motion else None
    if rollouts:
        num = min(len(r["frame"]) for r, _ in rollouts)
        frames = rollouts[0][0]["frame"][:num]
    else:
        num = len(reference["joint_pos"])
        frames = np.arange(num)
    if rollouts and "fps" in rollouts[0][0].files:
        fps = float(rollouts[0][0]["fps"])
    elif reference is not None:
        fps = float(np.asarray(reference["fps"]).item())
    else:
        parser.error("The rollouts do not record their frame rate; record them again with --play --record.")

    # Lay the robots out to the right of the reference's direction of travel (+x without a reference).
    forward = np.array([1.0, 0.0])
    if reference is not None:
        travel = reference["body_pos_w"][-1, 0, :2] - reference["body_pos_w"][0, 0, :2]
        if np.linalg.norm(travel) > 0.05:
            forward = travel / np.linalg.norm(travel)
    right = np.append([forward[1], -forward[0]], 0.0)

    actors = []
    if reference is not None:
        shift = right * args.offset * len(rollouts)
        actors.append(
            Actor(
                label="Reference",
                root_pos=reference["body_pos_w"][frames, 0] + shift,
                root_quat=reference["body_quat_w"][frames, 0],
                joint_pos=reference["joint_pos"][frames],
                joint_names=[str(n) for n in reference["joint_names"]],
                color=REFERENCE_COLOR,
                label_color=REFERENCE_LABEL_COLOR,
            )
        )
    for i, (rollout, label) in enumerate(rollouts):
        shift = right * args.offset * (len(rollouts) - 1 - i)
        actors.append(
            Actor(
                label=label,
                root_pos=rollout["root_pos"][:num] + shift,
                root_quat=rollout["root_quat"][:num],
                joint_pos=rollout["joint_pos"][:num],
                joint_names=[str(n) for n in rollout["joint_names"]],
                color=ROLLOUT_COLORS[i % len(ROLLOUT_COLORS)],
                label_color=ROLLOUT_LABEL_COLORS[i % len(ROLLOUT_LABEL_COLORS)],
            )
        )
    return actors, frames, fps, forward


_model: RobotModel | None = None


def _init_worker(urdf: str):
    global _model
    _model = RobotModel(urdf)


def _render(job):
    indices, actors, view, directory, times, text = job
    for k in indices:
        render_frame(_model, actors, k, view, os.path.join(directory, f"{k:05d}.png"), times[k], **text)
    return len(indices)


def main():
    args = parser.parse_args()
    actors, frames, fps, forward = load_actors(args)

    # Camera in front-left of the direction of travel, framing every robot over the whole motion, from the ground
    # up to the labels.
    xy = np.concatenate([actor.root_pos[:, :2] for actor in actors])
    top = LABEL_HEIGHT + 0.1
    view = View(
        center=np.append(xy.mean(axis=0), top / 2),
        half=max(np.ptp(xy, axis=0).max() / 2 + 0.3, top / 2),
        elev=args.elev,
        azim=float(np.degrees(np.arctan2(forward[1], forward[0])) + args.azim),
    )
    # Zoom in until the robots fill the frame: a box around each one, feet to head, plus its label, every 5th frame.
    roots = np.concatenate([actor.root_pos[::5] for actor in actors]) * [1.0, 1.0, 0.0]
    box = [[dx, dy, dz] for dx in (-0.3, 0.3) for dy in (-0.3, 0.3) for dz in (0.0, 1.3)] + [[0.0, 0.0, LABEL_HEIGHT]]
    view = fit_view(view, np.concatenate([roots + corner for corner in box]))
    times = frames / fps
    text = {"title": args.title, "caption": args.caption, "label_size": args.label_size, "label_height": LABEL_HEIGHT}

    if args.preview is not None:
        _init_worker(args.urdf)
        k = int(np.argmin(np.abs(times - args.preview)))
        render_frame(_model, actors, k, view, args.out, times[k], **text)
        print(f"[INFO] Saved {args.out} (t = {times[k]:.2f} s)")
        return

    if shutil.which("ffmpeg") is None:
        sys.exit("[ERROR] ffmpeg is needed to write the video: install it, e.g. sudo apt install ffmpeg.")
    with tempfile.TemporaryDirectory() as directory:
        jobs = [
            (range(i, len(frames), args.workers), actors, view, directory, times, text) for i in range(args.workers)
        ]
        with ProcessPoolExecutor(args.workers, initializer=_init_worker, initargs=(args.urdf,)) as pool:
            done = sum(pool.map(_render, jobs))
        print(f"[INFO] Rendered {done} frames")
        os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
        frames_pattern = os.path.join(directory, "%05d.png")
        encode = ["-c:v", "libx264", "-pix_fmt", "yuv420p", "-crf", "18"]
        subprocess.run(
            ["ffmpeg", "-v", "error", "-y", "-framerate", f"{fps:g}", "-i", frames_pattern, *encode, args.out],
            check=True,
        )
    print(f"[INFO] Saved {args.out}")


if __name__ == "__main__":
    main()
