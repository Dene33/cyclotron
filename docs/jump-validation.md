# Forward jump validation — 18 October 2026

The published policy, [dene33/asimov1-forward-jump](https://huggingface.co/dene33/asimov1-forward-jump), is
`Asimov1-Jump-v0` checkpoint `model_5498.pt` of run
`logs/rsl_rl/asimov1_jump/2026-10-17_19-26-17_jump_finetune_sync`. It is the first run's `model_3999.pt` (run
`2026-10-15_16-16-05_jump_cmu_13_11_forward_jump_froude_nolift`, 4000 iterations) fine-tuned for 1500 iterations
with the legs' left/right symmetry, feet lifting and landing together, the arms' and waist's joint angles, the heading,
smoothness penalties and randomised joint armature. The policy uses the walking policy's interface and needs no
reference motion to run. All results come from simulation: no physical robot was tested.

Measurements, rollouts, logs and the comparison video are in `logs/rsl_rl/asimov1_jump/validation_5498/` (training
artifacts, not in git).

## Export and interface

`--export --strict` passed with no warnings. The largest action differences against the checkpoint were 2.8e-6 for
`policy.onnx` and 1.2e-6 for `policy.pt`.

All 14 deployment contract fields match `Menlo/asimov1-locomotion-26-09-24` (snapshot
`fd55ba701bfb91f649ea1ffe127a4f26f6d282fe`): dimensions, observations, joint order, action processing, gains and
timing. That is what a runtime checks before running a policy; it does not show the policy runs on hardware.

- Checkpoint SHA-256: `5f94cad74ae54fc72fc70cd8112064129a31d8113de603de0a57769acde72e00`
- ONNX SHA-256: `6756e537bac2280babfdf99abf9c93566c65c14064951c7e619f13c2d97d4901`

## Isaac Sim

Each condition uses 1,024 robots, seed 1, and one 6.74 s cycle: 1.5 s standing in the default pose, the 0.3 m/s
command held through the jump and released 0.4 s after the reference lands, then 3 s holding the landing. A failure
is any training termination, including a large tracking error. Rise, travel and foot force are medians of the robots
that finished.

| Condition | Failures | Pelvis rise | Forward travel | Peak foot force / body weight |
| --- | ---: | ---: | ---: | ---: |
| Nominal | 0 / 1,024 | 10.2 cm | 39.5 cm | 13.8 |
| Training randomisation, no pushes | 1 / 1,024 | 9.0 cm | 53.3 cm | 12.9 |
| Training randomisation and pushes | 7 / 1,024 | 9.1 cm | 49.6 cm | 12.3 |
| Randomisation, no pushes, motor speed ratio 1 | 1 / 1,024 | 8.3 cm | 52.1 cm | 12.6 |

Both single randomised failures are the same robot (number 193 of the seed) at 0.14 s, before the jump command, so
they say nothing about the jump. With pushes, five failures were end-effector tracking errors and two
anchor position errors, between 0.90 and 3.76 s. The motor curve uses the weakest assumed no-load speed; the real
curve is unknown. Several motors reach their simulated torque or speed limits and the peak contact loads are large:
these tests don't establish hardware load margins.

## Legs, arms and smoothness

Before (`model_3999`) and after the fine-tune, from one Isaac Sim rollout each (`--play --record`) and the MuJoCo
rollouts below. Takeoff and touchdown are when each foot's lowest sole point leaves or reaches 1.5 cm.

| | Before | After | Reference |
| --- | ---: | ---: | ---: |
| Largest foot height difference, Isaac / MuJoCo | 14.5 / 11.5 cm | 3.8 / 6.3 cm | 2.1 cm |
| Takeoff gap (left foot first), Isaac / MuJoCo | 80 / 80 ms | 60 / 20 ms | 0 ms |
| Touchdown gap, Isaac / MuJoCo | 40 / 80 ms | 0 / 60 ms | 20 ms |
| Knee left/right asymmetry (RMS), Isaac | 12.9° | 6.8° | 7.3° |
| Right wrist yaw at the jump command | +30° | within ±5° | 0° |
| Waist yaw swing | ±14° | ±8° | ±3° |
| Pelvis heading change over the jump, Isaac | -20° | -14° | -2° |

Without a clock among its inputs, the first run's policy turned a wrist and the waist, which no tracked body
notices, apparently to keep time; the arms' and waist's joint angle reward removed that. `finetune_joints.png` plots
the knees, hips, wrist and waist against the reference.

## MuJoCo

The exported `policy.onnx` on the published Asimov 1 MJCF, with the run's PD gains, effort limits and joint armature
and the firmware's ankle target limits, following `0:1.5,0.3:2,0:2.5` (stand, the jump command, stand). Rise and
displacement are measured from the end of the first standing interval. These are single deterministic rollouts, not
success rates.

| Condition | Fell | Pelvis rise | Forward | Sideways | Steps after landing |
| --- | --- | ---: | ---: | ---: | ---: |
| Nominal | No | 12.9 cm | 40.2 cm | -7.8 cm | 0 |
| 25 ms target delay | No | 11.7 cm | 47.4 cm | -6.4 cm | 1 |
| Motor speed ratio 1 | No | 11.4 cm | 36.8 cm | -7.2 cm | 1 |
| Foot friction 0.4 | No | 13.0 cm | 40.8 cm | -4.5 cm | 1 |
| All three together | No | 10.0 cm | 44.9 cm | -1.6 cm | 0 |

The highest projected gravity z is -0.71, with the firmware's fall trip above -0.5. Both feet are off the ground from
2.58 to 2.80 s in the nominal rollout, the command having started at 1.5 s. At zero command the robot stands
still for 6 s. With the command held for 5 s it jumps twice, 2.2 s apart; released in the crouch of a third jump, it
doesn't jump and takes two steps to settle. So the command should be released after landing, not in the next crouch.

## SDK runner against a simulated robot

`./cyclotron.sh --sim2real run` streamed the policy's targets for 300 steps at 50 Hz to `--mujoco-edge` over loopback
UDP through menlo-sdk, with the simulated robot facing 0.7 rad: once nominal and once with 10 ms command and 10 ms state
delays. Both completed with no faults or alerts; the simulated robot's pelvis rose 13.8 and 12.7 cm.

## Known limitations

- The robot turns about 14° to its right over the jump; the reference turns 2°. The heading reward reduced it from
  20°, but the policy has no absolute heading among its inputs.
- The jump is lower than the reference's: the pelvis rises 9 to 13 cm against its 15.6 cm, and less than before the
  fine-tune (Isaac nominal 10.2 cm against 12.4 cm, MuJoCo 12.9 cm against 14.3 cm).
- After landing it holds a slightly deeper crouch than the reference's last frame (knees about 30° instead of 25°).
- The landing depends on the ankle's inertia, which is known through Menlo's ankle model only. Training randomises the
  joints' armature from 0.6 to 1.4 times the model's.

## Corrections to the 15 October report

- Its MuJoCo results used the published MJCF's ankle armature, 7.6 times less than Menlo's ankle model the policy was
  trained with, which made the robot take two extra steps after landing. With the trained armature (`--sim2sim` now
  uses the run's), `model_3999` rose 14.3 cm and travelled 51.5 cm forward and -9.4 cm sideways, with no extra steps.
- Its comparison video showed the commanded jump turned by 90°: the rollout was shifted in time but not turned to the
  reference's heading. `finetune_comparison.mp4` turns and places each rollout where the reference stands at the jump
  command.

## Comparison video

`finetune_comparison.mp4` shows the reference, the policy before the fine-tune and the published one in MuJoCo, with an
orthographic 45° camera, at 50 fps in real time, each rollout starting at the reference's crouch (0.5 s into the
motion) when its jump command starts.

## Reproduce

```bash
RUN=2026-10-17_19-26-17_jump_finetune_sync
./cyclotron.sh --export --strict --checkpoint logs/rsl_rl/asimov1_jump/$RUN/model_5498.pt
./cyclotron.sh --compare-contract logs/rsl_rl/asimov1_jump/$RUN/exported/policy.onnx Menlo/asimov1-locomotion-26-09-24
./cyclotron.sh --evaluate --task Asimov1-Jump-v0 --load_run $RUN --checkpoint model_5498.pt \
    --condition randomized --num_envs 1024 --seed 1 --save results.npz
./cyclotron.sh --sim2sim logs/rsl_rl/asimov1_jump/$RUN --command 0:1.5,0.3:2,0:2.5
```

Use `--condition nominal` or `--condition pushes` for the other Isaac conditions and add `--motor_speed_ratio 1` for
the motor curve; `--delay 5`, `--motor_speed_ratio 1` and `--foot_friction 0.4` give the MuJoCo variants.

Code: [Dene33/cyclotron, motion-tracking](https://github.com/Dene33/cyclotron/tree/motion-tracking).
Model: [dene33/asimov1-forward-jump](https://huggingface.co/dene33/asimov1-forward-jump).
