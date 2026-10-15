# Forward jump validation — 15 October 2026

`Asimov1-Jump-v0`, checkpoint `model_3999.pt`, completed 4,000 training iterations. The policy uses the walking
interface and needs no reference motion at inference. These results come from simulation; no physical robot was tested.

The run is `logs/rsl_rl/asimov1_jump/2026-10-15_16-16-05_jump_cmu_13_11_forward_jump_froude_nolift`.
Its `exported/` folder holds the validated policy and run settings. Measurements, rollouts, logs, the model-card
preview and `summary.json` are in `logs/rsl_rl/asimov1_jump/validation_3999/` (ignored training artifacts).

## Export and interface

`--export --strict` passed, including the saved settings, resolved joints and gains, and model/network checks.
Maximum action differences against the checkpoint were 1.4e-6 for ONNX and 1.2e-6 for TorchScript.

All 14 deployment contract fields match `Menlo/asimov1-locomotion-26-09-24`, snapshot
`fd55ba701bfb91f649ea1ffe127a4f26f6d282fe`: dimensions, observations, joint order, action processing, gains and timing.
That verifies interface compatibility. Physical deployment remains untested.

- Checkpoint SHA-256: `64a114815640d0f4ea5312304cf751b9e3038084ba057162b4a675ab2aa6d09d`
- ONNX SHA-256: `dc4cb4852b979abf2203f2efa25b235c10a4b7f469e10da71366caca534d7572`

## Isaac Sim

Each condition uses 1,024 robots, seed 1, and one 6.74 s cycle: 1.5 s standing in the default pose, one jump, then
3 s holding the landing. The trigger is 0.3 m/s, released 0.4 s after the reference lands. A failure means any
training termination, including excessive tracking error. Rise, travel and foot force below are medians among finishers.

| Condition | Failures | Pelvis rise | Forward travel | Peak foot force / body weight |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 0 / 1,024 | 12.4 cm | 48.0 cm | 13.4 |
| Training randomization, no pushes | 1 / 1,024 | 11.0 cm | 64.3 cm | 13.7 |
| Training randomization and pushes | 7 / 1,024 | 10.9 cm | 67.5 cm | 13.4 |
| Randomization, no pushes, motor speed ratio 1 | 0 / 1,024 | 10.2 cm | 63.2 cm | 13.6 |

The randomized failure was an end-effector tracking error at 3.68 s. With pushes, six failures were end-effector
tracking errors and one was an anchor position error. The motor curve uses the weakest assumed no-load speed;
the real motor curve is unknown. Firmware ankle target limits remain active when this curve is enabled.

Several motors reach their simulated torque or speed limits, and peak contact loads are substantial. These tests
do not establish hardware load margins. `isaac_*.npz` contains per-motor results and physics-rate traces.
`isaac_play.npz` records a full cycle of 338 frames at 50 Hz.

Reproduce a condition, replacing `RUN` with the run name above:

```bash
./cyclotron.sh --evaluate --task Asimov1-Jump-v0 --load_run RUN --checkpoint model_3999.pt \
    --condition randomized --num_envs 1024 --seed 1 --save results.npz
```

Use `--condition nominal` or `--condition pushes` for the other conditions, and add `--motor_speed_ratio 1`
for the motor curve.

## MuJoCo and SDK runner

The exported ONNX policy runs against the published Asimov 1 MJCF with firmware ankle limits enabled. The command
schedule is `0:1.5,0.3:2,0:2.5`. Rise and displacement are measured from the end of the first standing interval.

| Condition | Fell | Pelvis rise | Forward travel | Sideways travel |
| --- | --- | ---: | ---: | ---: |
| Nominal | No | 14.0 cm | 61.2 cm | +0.7 cm |
| 25 ms target delay | No | 13.9 cm | 61.2 cm | -7.3 cm |
| Motor speed ratio 1 | No | 12.3 cm | 45.0 cm | -7.8 cm |
| Foot friction 0.4 | No | 14.1 cm | 61.9 cm | +3.0 cm |
| All three changes together | No | 12.0 cm | 57.1 cm | -2.5 cm |

Contact reconstruction from the nominal 50 Hz rollout shows both feet airborne from approximately 2.54 to 2.80 s.
The highest projected gravity z is -0.674, below the firmware fall trip of -0.5. These are individual deterministic
MuJoCo rollouts, not randomized success-rate estimates.

The SDK runner also completed 300 control steps at 50 Hz against `--mujoco-edge` over loopback UDP, with the
simulated robot facing 0.7 rad. Both nominal communication and 10 ms command plus 10 ms state delays completed
without faults or alerts. All traffic stayed on `127.0.0.1`. Logs and robot rollouts are `runner_*.log`,
`runner_*.npz` and `edge_*.npz` in the artifact directory.

## Comparison video

`comparison.mp4` shows Reference, the earlier tracking jump, and the new commanded jump in progression order,
with an orthographic 45-degree camera at 25-degree elevation. Both policies run in MuJoCo with firmware ankle
limits. The earlier tracking policy rises 8.3 cm; the new policy rises 14.0 cm.

The new rollout's clock is shifted by 1 s so its trigger at 1.5 s aligns with the reference's crouch at 0.5 s.
The comparison contains 137 recorded frames at 50 fps, without slowing or repeating frames. The full new rollout
remains available as `mujoco_nominal.npz`; `comparison_preview.png` is a still from the video.

## Code checks

The standard suite passes 117 tests, with two skips: SDK integration in the Isaac Python environment, and the
actuator module requiring a running Isaac Sim app. All three actuator regression cases pass when run after
starting Isaac Sim. They verify conversion preserves the firmware target limits and that target clamping and
speed-dependent torque work together, while ordinary DC motors retain unclamped targets.

The evaluator now runs the jump's complete stand–jump–hold cycle. The DC motor model accepts and applies optional
position target limits, fixing `--evaluate --motor_speed_ratio` for this task.

Code: [Dene33/cyclotron, motion-tracking](https://github.com/Dene33/cyclotron/tree/motion-tracking).
Model: [dene33/asimov1-forward-jump](https://huggingface.co/dene33/asimov1-forward-jump).
The local `orbit-hip` branch stays local.
