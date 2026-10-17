from __future__ import annotations

import hashlib

import yaml

from cyclotron.hub import link_text, rehash_uploaded_run_configs, strip_local_paths, tracking_model_card


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def test_rehash_uploaded_run_configs_points_code_state_at_the_uploaded_copies():
    env, agent = "asset_path: /home/me/robot.urdf\n", "seed: 1\n"
    recorded = {"env.yaml": sha256(env), "agent.yaml": sha256(agent)}
    uploaded = {"env.yaml": strip_local_paths(env)[0], "agent.yaml": agent}
    code_state = yaml.safe_dump({"run_configs": recorded}, sort_keys=False)
    rehashed = yaml.safe_load(rehash_uploaded_run_configs(code_state, recorded, uploaded))["run_configs"]
    assert rehashed == {"env.yaml": sha256("asset_path: robot.urdf\n"), "agent.yaml": sha256(agent)}


# Trimmed copy of a real shared env.yaml: Isaac Lab tags, training-machine paths and internal module names.
ENV_YAML = """\
sim:
  dt: 0.005
  render_interval: 4
  gravity: !!python/tuple
  - 0.0
  - 0.0
  - -9.81
seed: 42
decimation: 4
scene:
  num_envs: 8192
  robot:
    class_type: isaaclab.assets.articulation.articulation:Articulation
    prim_path: /World/envs/env_.*/Robot
    spawn:
      asset_path: /home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/asimov_1.urdf
    init_state:
      pos: !!python/tuple
      - 0.0
      - 0.0
      - 0.639
      joint_pos:
        left_knee_joint: 0.45
        .*_hip_roll_joint: 0.0
    actuators:
      hip_pitch:
        class_type: isaaclab.actuators.actuator_pd:DelayedPDActuator
        joint_names_expr:
        - .*_hip_pitch_joint
        effort_limit: 45.0
        stiffness: 150.0
        damping: 5.0
      neck:
        stiffness: 10.0
observations:
  policy:
    enable_corruption: true
    base_ang_vel:
      func: isaaclab_tasks.manager_based.locomotion.velocity.config.asimov_1.mdp.observations:delayed_obs
      params:
        quantity: base_ang_vel
        max_lag: 1
      noise:
        func: isaaclab.utils.noise.noise_model:uniform_noise
        n_min: -0.01
      scale: 0.25
    joint_pos:
      params:
        asset_cfg:
          name: robot
          joint_names:
          - left_hip_pitch_joint
          joint_ids: !!python/object/apply:builtins.slice
          - null
          - null
          - null
    extra_term:
      scale: 1.0
  amp:
    concatenate_terms: true
actions:
  joint_pos:
    class_type: isaaclab.envs.mdp.actions.joint_actions:JointPositionAction
    joint_names:
    - left_hip_pitch_joint
    - left_knee_joint
    scale: 0.25
"""


def test_strip_local_paths_keeps_file_names_and_prim_paths():
    text = """\
viewer:
  cam_prim_path: /OmniverseKit_Persp
scene:
  robot:
    prim_path: /World/envs/env_.*/Robot
    spawn:
      asset_path: /home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/asimov_1.urdf
      usd_dir: null
      visual_material_path: material
  contact_forces:
    filter_prim_paths_expr:
    - /World/envs/env_.*/Robot/torso_link
  terrain:
    terrain_generator:
      cache_dir: '/tmp/isaaclab/terrains'
    usd_path: https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/arrow_x.usd
recorders:
  dataset_export_dir_path: C:\\runs\\logs
amp_data:
  motion_files:
  - /home/menlo/motions/walk_slow.npz
  - ~/motions/run.npz
  joint_names:
  - left_hip_pitch_joint
"""
    stripped, removed = strip_local_paths(text)
    assert removed == [
        "/home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/asimov_1.urdf",
        "/tmp/isaaclab/terrains",
        "C:\\runs\\logs",
        "/home/menlo/motions/walk_slow.npz",
        "~/motions/run.npz",
    ]
    assert stripped == (
        text.replace("/home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/", "")
        .replace("/tmp/isaaclab/", "")
        .replace("C:\\runs\\", "")
        .replace("/home/menlo/motions/", "")
        .replace("~/motions/", "")
    )


def test_strip_local_paths_keeps_policy_settings():
    stripped, removed = strip_local_paths(ENV_YAML)
    assert removed == ["/home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/asimov_1.urdf"]
    changed = [(a, b) for a, b in zip(ENV_YAML.splitlines(), stripped.splitlines()) if a != b]
    assert changed == [
        (
            "      asset_path: /home/menlo/IsaacLab/source/isaaclab_assets/data/Robots/Asimov1/asimov_1.urdf",
            "      asset_path: asimov_1.urdf",
        )
    ]


def test_tracking_model_card_says_how_to_run_the_policy():
    card = tracking_model_card(
        title="Forward jump",
        summary="A jump.",
        motion="jump.npz",
        duration=2.74,
        fps=50.0,
        action_scale=0.25,
        code_url="https://github.com/someone/cyclotron/tree/motion-tracking",
        checkpoint="model_3300.pt",
        motion_source="CMU 13_11.",
    )

    assert card.startswith("---\nlibrary_name: asimov\n")
    assert "# Forward jump\n\nA jump.\n" in card
    assert "`jump.npz`: the reference motion at 50 Hz, 2.74 s long" in card
    assert "every 20 ms" in card and "default pose + 0.25 x action" in card
    assert "Checkpoint `model_3300.pt`." in card
    assert "## Reference motion\n\nCMU 13_11.\n" in card
    assert "[someone/cyclotron/tree/motion-tracking](https://github.com/someone/cyclotron/tree/motion-tracking)" in card
    assert "{" not in card


def test_tracking_model_card_leaves_out_what_is_not_known():
    card = tracking_model_card("T", "S", "m.npz", 1.0, 50.0, 0.25, "https://github.com/menloresearch/cyclotron")
    assert "Checkpoint" not in card
    assert "## Reference motion" not in card


def test_link_text_drops_the_scheme_and_github_host():
    assert link_text("https://github.com/menloresearch/cyclotron") == "menloresearch/cyclotron"
    assert link_text("https://example.org/models/") == "example.org/models"


def test_jump_model_card_says_how_to_trigger_the_jump():
    from cyclotron.hub import jump_model_card

    card = jump_model_card(
        title="Jump",
        summary="A jump.",
        walking="[Menlo/walk](https://huggingface.co/Menlo/walk)",
        motion="jump.npz",
        speed=0.3,
        speed_range=(0.25, 0.4),
        hold_s=2.24,
        landing_s=1.4,
        code_url="https://github.com/me/cyclotron",
        checkpoint="model_3999.pt",
        motion_source="CMU 13_11.",
    )
    assert card.startswith("---\nlibrary_name: asimov\n")
    assert "A forward speed of 0.3 m/s (it was trained with 0.25 to 0.4), held for the whole jump" in card
    assert "about 2.2 s" in card and "lands 1.4 s after the\n  command starts; release the command after landing" in card
    assert "[Menlo/walk](https://huggingface.co/Menlo/walk)" in card and "Checkpoint `model_3999.pt`." in card
    assert "Reference motion: CMU 13_11." in card and "[me/cyclotron](https://github.com/me/cyclotron)" in card
