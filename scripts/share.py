"""Upload a completed training run to the Hugging Face Hub.

Uploads ``params/agent.yaml``, ``params/env.yaml`` and ``exported/policy.onnx`` from a run directory,
together with a generated model card, and for motion tracking runs the reference motion the policy follows. If the
run has no ONNX export yet, the latest checkpoint is exported first via ``play.py --export-only``. Requires a prior
``hf auth login`` (or ``HF_TOKEN``).

File paths from the training machine are reduced to file names in the uploaded yaml files; the run directory
itself is not changed.
"""

import argparse
import os
import re
import subprocess
import sys

import numpy as np

from cyclotron.hub import link_text, strip_local_paths, tracking_model_card
from cyclotron.sim2sim import TrackingRun, load_env_yaml

README_TEMPLATE = """---
library_name: asimov
pipeline_tag: robotics
license: bsd-3-clause
---

# {title}

{summary}

- `policy.onnx` is the exported policy for inference, producing joint-position actions.
- `agent.yaml` records the actor-critic and AMP training settings, including the motion data configuration.
- `env.yaml` records the simulation and locomotion task settings, including observations, actions, commands, and rewards.
- Training and evaluation code: [{code_name}]({code_url}).
"""

DEFAULT_CODE_URL = "https://github.com/menloresearch/cyclotron"
DEFAULT_TITLE = "Asimov 1 locomotion policy checkpoint"
DEFAULT_SUMMARY = (
    "Basic velocity-commanded locomotion policy for Asimov 1, trained in Isaac Lab with PPO and adversarial motion"
    " priors (AMP)."
)
DEFAULT_TRACKING_SUMMARY = (
    "Motion tracking policy for Asimov 1, trained in Isaac Lab with PPO to follow a retargeted human motion."
)

# Training task for each experiment name, used when exporting a run that has no policy.onnx yet.
EXPERIMENT_TASKS = {
    "asimov1_velocity": "Asimov1-Velocity-v0",
    "asimov_velocity_amp": "Asimov1-Velocity-AMP-v0",
    "asimov1_tracking": "Asimov1-Tracking-Play-v0",
}

PLAY_SCRIPT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rsl_rl", "play.py")

parser = argparse.ArgumentParser(description="Upload a trained Cyclotron run to the Hugging Face Hub.")
parser.add_argument("run_dir", type=str, help="Run directory, e.g. logs/rsl_rl/<experiment_name>/<run>.")
parser.add_argument("--repo-id", "--repo_id", dest="repo_id", required=True, help="Target repo, e.g. user/name.")
parser.add_argument("--title", type=str, default=None, help="Title of the model card.")
parser.add_argument(
    "--summary",
    type=str,
    default=None,
    help="Paragraph shown under the model card title, e.g. describing the training method or results.",
)
parser.add_argument(
    "--code-url", "--code_url", dest="code_url", default=DEFAULT_CODE_URL, help="Training code linked from the card."
)
parser.add_argument(
    "--motion-source",
    "--motion_source",
    dest="motion_source",
    default=None,
    help="Motion tracking runs: where the reference motion comes from, with any credit its licence asks for.",
)
parser.add_argument(
    "--onnx",
    type=str,
    default=None,
    help="Path to the ONNX policy. Defaults to <run_dir>/exported/policy.onnx, exported automatically if missing.",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help=(
        "Checkpoint to export and upload: a full path to a .pt file, or a filename inside run_dir."
        " Defaults to the latest model_*.pt in the run if policy.onnx is missing."
    ),
)
parser.add_argument(
    "--task",
    type=str,
    default=None,
    help="Task used to export the policy. Inferred from the experiment name in agent.yaml if omitted.",
)
parser.add_argument("--private", action="store_true", help="Create the repo as private if it does not exist.")
parser.add_argument("--dry-run", "--dry_run", dest="dry_run", action="store_true", help="Print the card and exit.")


def latest_checkpoint(run_dir: str) -> str:
    checkpoints = [f for f in os.listdir(run_dir) if re.fullmatch(r"model_\d+\.pt", f)]
    if not checkpoints:
        sys.exit(f"[ERROR] No model_*.pt checkpoints found in: {run_dir}")
    return os.path.join(run_dir, max(checkpoints, key=lambda f: int(f[len("model_") : -len(".pt")])))


def resolve_checkpoint(run_dir: str, checkpoint: str) -> str:
    """Resolve --checkpoint the same way as train.py and play.py, with run_dir standing in for --load_run."""
    path = os.path.expanduser(checkpoint) if os.sep in checkpoint else os.path.join(run_dir, checkpoint)
    if not os.path.isfile(path):
        sys.exit(f"[ERROR] Checkpoint not found: {path}")
    return os.path.abspath(path)


def infer_task(agent_yaml_path: str) -> str:
    with open(agent_yaml_path) as f:
        match = re.search(r"^experiment_name: *(\S+)", f.read(), re.MULTILINE)
    experiment_name = match.group(1).strip("'\"") if match else None
    if experiment_name not in EXPERIMENT_TASKS:
        sys.exit(f"[ERROR] Cannot infer the task for experiment '{experiment_name}'. Pass it with --task.")
    return EXPERIMENT_TASKS[experiment_name]


def export_onnx(checkpoint: str, task: str) -> None:
    print(f"[INFO] Exporting {checkpoint} to ONNX using task {task}")
    command = [sys.executable, PLAY_SCRIPT, "--task", task, "--checkpoint", checkpoint]
    command += ["--num_envs", "1", "--headless", "--export-only"]
    if subprocess.run(command).returncode != 0:
        sys.exit("[ERROR] ONNX export failed.")


def main() -> None:
    args_cli = parser.parse_args()
    run_dir = os.path.abspath(os.path.expanduser(args_cli.run_dir))
    onnx_path = os.path.abspath(os.path.expanduser(args_cli.onnx)) if args_cli.onnx else None

    files = {
        "agent.yaml": os.path.join(run_dir, "params", "agent.yaml"),
        "env.yaml": os.path.join(run_dir, "params", "env.yaml"),
        "policy.onnx": onnx_path or os.path.join(run_dir, "exported", "policy.onnx"),
    }
    missing = [path for name, path in files.items() if name != "policy.onnx" and not os.path.isfile(path)]
    if onnx_path and not os.path.isfile(onnx_path):
        missing.append(onnx_path)
    if missing:
        sys.exit("[ERROR] Missing files:\n  " + "\n  ".join(missing))

    needs_export = not onnx_path and (args_cli.checkpoint or not os.path.isfile(files["policy.onnx"]))
    if needs_export:
        if args_cli.checkpoint:
            checkpoint = resolve_checkpoint(run_dir, args_cli.checkpoint)
        else:
            checkpoint = latest_checkpoint(run_dir)
        task = args_cli.task or infer_task(files["agent.yaml"])
        # play.py exports next to the checkpoint, so upload from there.
        files["policy.onnx"] = os.path.join(os.path.dirname(checkpoint), "exported", "policy.onnx")

    env_cfg = load_env_yaml(files["env.yaml"])
    motion_cfg = (env_cfg.get("commands") or {}).get("motion")
    if isinstance(motion_cfg, dict) and motion_cfg.get("motion_file"):
        # A tracking policy reads its reference motion at run time, so the motion goes with it, under the file name
        # the shared env.yaml refers to.
        run = TrackingRun.load(run_dir)
        if not os.path.isfile(run.motion_file):
            sys.exit(f"[ERROR] The run's reference motion is missing: {run.motion_file}")
        motion_name = os.path.basename(run.motion_file)
        files[motion_name] = run.motion_file
        fps = 1.0 / (run.physics_dt * run.decimation)
        readme = tracking_model_card(
            title=args_cli.title or f"Asimov 1 motion tracking policy: {os.path.splitext(motion_name)[0]}",
            summary=args_cli.summary or DEFAULT_TRACKING_SUMMARY,
            motion=motion_name,
            duration=len(np.load(run.motion_file)["joint_pos"]) / fps,
            fps=fps,
            action_scale=run.action_scale,
            code_url=args_cli.code_url,
            checkpoint=os.path.basename(checkpoint) if needs_export else None,
            motion_source=args_cli.motion_source,
        )
    else:
        readme = README_TEMPLATE.format(
            title=args_cli.title or DEFAULT_TITLE,
            summary=args_cli.summary or DEFAULT_SUMMARY,
            code_url=args_cli.code_url,
            code_name=link_text(args_cli.code_url),
        )

    # Upload copies of the yaml files without the training machine's file paths; the run directory is left as is.
    yamls = {}
    for name in ("agent.yaml", "env.yaml"):
        with open(files[name]) as f:
            yamls[name], removed = strip_local_paths(f.read())
        for path in removed:
            print(f"[INFO] Removing local path from {name}: {path}")

    if args_cli.dry_run:
        if needs_export:
            print(f"[INFO] Would export {checkpoint} to ONNX using task {task}")
        for repo_path, local_path in files.items():
            print(f"[INFO] {local_path} -> {repo_path}")
        print(readme)
        return

    try:
        from huggingface_hub import CommitOperationAdd, HfApi
    except ImportError:
        sys.exit("[ERROR] huggingface_hub is not installed. Install it with: ./cyclotron.sh --install")

    api = HfApi()
    try:
        user = api.whoami()["name"]
    except Exception:
        sys.exit("[ERROR] Not logged in to Hugging Face. Run: hf auth login")
    print(f"[INFO] Logged in as: {user}")

    if needs_export:
        export_onnx(checkpoint, task)

    repo_url = api.create_repo(args_cli.repo_id, repo_type="model", private=args_cli.private, exist_ok=True)
    operations = [
        CommitOperationAdd(path_in_repo=name, path_or_fileobj=yamls[name].encode() if name in yamls else path)
        for name, path in files.items()
    ]
    operations.append(CommitOperationAdd(path_in_repo="README.md", path_or_fileobj=readme.encode()))
    api.create_commit(
        repo_id=args_cli.repo_id,
        repo_type="model",
        operations=operations,
        commit_message=f"Upload Asimov policy from {os.path.basename(run_dir)}",
    )
    print(f"[INFO] Uploaded to: {repo_url}")


if __name__ == "__main__":
    main()
