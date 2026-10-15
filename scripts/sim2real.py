"""Play a motion tracking policy on the real Asimov 1 (sim2real), through Menlo's Python SDK (menlo-sdk).

Three commands, from the safest:

- ``check``: read the robot's state for a few seconds; report the state rate, the IMU and every joint in the policy's
  frame, and flag joints outside the URDF's limits (a sign or ankle conversion error). Sends nothing.
- ``dry-run``: run the policy at the policy rate on the live state, as ``run`` does, but send nothing. Measures the
  loop's timing and shows the targets the policy would send.
- ``run``: stream the policy's joint targets: first tracking the reference's first frame, standing still, for
  ``--settle`` seconds (the policy balances the robot from STAND into the motion's starting pose), then the motion,
  then its last frame for ``--hold`` seconds; then DAMP (or hand back to the walking policy with ``--end balance``).
  The robot must hang from its gantry hook with both feet on the floor, in STAND and armed.

``dry-run`` and ``run`` record every step to ``<run>/sim2real/<command>_<time>.npz``.

menlo-sdk needs Python 3.12 or newer, and Asimov Edge must send its state to this machine (``udp-state-host``):
run this with ``./cyclotron.sh --sim2real``. ``scripts/mujoco_edge.py`` stands in for the robot, for testing.
"""

import argparse
import contextlib
import os
import sys
import threading
import time
from datetime import datetime
from pathlib import Path

import numpy as np

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "cyclotron"))

DEFAULT_URDF = (
    Path(os.environ.get("ASIMOV_1_MODEL_DIR", REPOSITORY_ROOT / "third_party" / "asimov-1" / "sim-model")).expanduser()
    / "urdf"
    / "asimov_1.urdf"
)
# A state sample older than this is not used: the policy would act on where the robot was.
MAX_STATE_AGE_S = 0.1
# The firmware warns at 60 C on an actuator (and latches DAMP at 80 C) and below 20 % battery.
MAX_START_TEMPERATURE_C = 60.0
MIN_START_BATTERY_PERCENT = 20.0
# The firmware's fall trip: projected gravity z above this.
FALL_GRAVITY_Z = -0.5

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawTextHelpFormatter)
parser.add_argument(
    "--host",
    type=str,
    default=None,
    help="The robot's address. Default: menlo-sdk's saved robot (menlo setup) or MENLO_UDP_HOST.",
)
parser.add_argument("--command_port", type=int, default=8850, help="Asimov Edge's UDP command port.")
parser.add_argument("--state_port", type=int, default=8851, help="Local UDP port Asimov Edge sends state to.")
parser.add_argument("--version_port", type=int, default=3000, help="Asimov Edge's HTTP version endpoint port.")
commands = parser.add_subparsers(dest="command", required=True)

check_parser = commands.add_parser("check", help="Report the robot's state in the policy's frame. Sends nothing.")
check_parser.add_argument(
    "run_dir", type=str, nargs="?", default=None, help="Optional training run, to compare with its default pose."
)
check_parser.add_argument("--seconds", type=float, default=3.0, help="How long to read the state.")
check_parser.add_argument("--urdf", type=str, default=str(DEFAULT_URDF), help="URDF with the joint limits.")

for name, help_text in (
    ("dry-run", "Run the policy on the live state without sending anything."),
    ("run", "Play the policy on the robot. It must hang from its gantry hook, feet on the floor, in STAND."),
):
    command = commands.add_parser(name, help=help_text)
    command.add_argument("run_dir", type=str, help="Training run, e.g. logs/rsl_rl/asimov1_tracking/<run>.")
    command.add_argument(
        "--onnx", type=str, default=None, help="Policy to play. Default: <run_dir>/exported/policy.onnx."
    )
    command.add_argument("--urdf", type=str, default=str(DEFAULT_URDF), help="URDF with the joint limits.")
    command.add_argument(
        "--settle", type=float, default=1.0, help="Seconds tracking the first frame, standing still, before the motion."
    )
    command.add_argument("--hold", type=float, default=1.0, help="Seconds to keep tracking the reference's last frame.")
    command.add_argument("--record", type=str, default=None, help="Where to save the log (.npz).")
    if name == "run":
        command.add_argument(
            "--blend",
            type=float,
            default=0.0,
            help="Seconds to blend the joints to the first frame's pose before the policy starts (0: off). The\n"
            "blend only holds joint positions, without balance: the robot tips over unless the gantry holds it.",
        )
        command.add_argument(
            "--end",
            choices=("damp", "balance"),
            default="damp",
            help="After the motion: DAMP (robot supported), or balance() to hand back to the walking policy\n"
            "(not documented after joint control: try it on the gantry first).",
        )
        command.add_argument(
            "--from_move",
            action="store_true",
            help="Allow starting in MOVE: the first joint target switches the walking policy off.",
        )
        command.add_argument("--yes", action="store_true", help="Do not ask for confirmation before moving.")


def connect(args):
    from menlo.asimov import ConnectionConfig, Robot, UdpConfig

    if args.host is None:
        return Robot().connect()
    config = ConnectionConfig(
        udp=UdpConfig(
            host=args.host,
            command_port=args.command_port,
            state_bind=("0.0.0.0", args.state_port),
            version_port=args.version_port,
        )
    )
    return Robot(config).connect("udp")


def wait_for_state(robot, timeout: float = 3.0):
    from menlo.asimov import NotConnectedError

    deadline = time.monotonic() + timeout
    while True:
        try:
            state = robot.get_state()
            if state.age_s < MAX_STATE_AGE_S:
                return state
        except NotConnectedError:
            pass
        if time.monotonic() > deadline:
            sys.exit(
                "[ERROR] No state from the robot. Asimov Edge must run with udp-control on and send state to this"
                " machine (udp-state-host), on the state port."
            )
        time.sleep(0.01)


def check_joint_table(robot) -> None:
    from cyclotron.sim2real import FIRMWARE_JOINTS

    names = robot.info.joint_names
    if names is None or tuple(names) != FIRMWARE_JOINTS:
        sys.exit(f"[ERROR] Expected Asimov 1's 25 motors {FIRMWARE_JOINTS}, the robot has {names}.")


def describe(state, robot) -> str:
    hottest = max((j for j in state.joints if j.temp is not None), key=lambda j: j.temp, default=None)
    battery = f"{state.battery.soc_percent:.0f} %" if state.battery is not None else "not reported"
    alerts = ", ".join(a.name for a in state.alerts) or "none"
    return (
        f"mode {state.mode.name}, armed {robot.armed}, faulted {state.faulted}, alerts {alerts}, hottest joint "
        + (f"{hottest.temp:.0f} C ({hottest.name})" if hottest else "not reported")
        + f", battery {battery}"
    )


def stats_ms(values) -> str:
    values = np.asarray(values, dtype=np.float64) * 1e3
    if len(values) == 0:
        return "no samples"
    return f"mean {values.mean():.1f}, p99 {np.percentile(values, 99):.1f}, max {values.max():.1f} ms"


def gravity_from_quat(quat: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    return Rotation.from_quat(np.asarray(quat)[..., [1, 2, 3, 0]]).inv().apply([0.0, 0.0, -1.0])


def body_rates_from_quats(quats: np.ndarray, dt: np.ndarray) -> np.ndarray:
    """Angular velocity in the body frame between consecutive orientations (w, x, y, z)."""
    from scipy.spatial.transform import Rotation

    rotations = Rotation.from_quat(quats[:, [1, 2, 3, 0]])
    return (rotations[:-1].inv() * rotations[1:]).as_rotvec() / dt[:, None]


# check


def check(args) -> None:
    from cyclotron.sim2real import FIRMWARE_JOINTS, MODEL_JOINTS, NECK_JOINTS, JointMap, load_joint_limits
    from cyclotron.sim2sim import load_env_yaml, per_joint

    samples = []
    with connect(args) as robot:
        robot.on_state = samples.append
        check_joint_table(robot)
        wait_for_state(robot)
        samples.clear()
        time.sleep(args.seconds)
        robot.on_state = None
        state = robot.get_state()
        print(f"[INFO] {describe(state, robot)}")
    if len(samples) < 2:
        sys.exit(f"[ERROR] Only {len(samples)} state samples in {args.seconds:g} s.")

    fw_time = np.array([s.fw_timestamp_us for s in samples], dtype=np.float64) * 1e-6
    arrival = np.array([s.received_at for s in samples])
    sequence = np.array([s.sequence for s in samples], dtype=np.int64)
    print(f"[INFO] {len(samples)} samples in {args.seconds:g} s ({len(samples) / args.seconds:.0f} Hz)")
    print(f"[INFO]   firmware sample interval: {stats_ms(np.diff(fw_time))}")
    print(f"[INFO]   arrival interval:          {stats_ms(np.diff(arrival))}")
    print(f"[INFO]   sequence gaps (lost samples): {int(np.sum(np.diff(sequence) - 1))}")

    # IMU: projected gravity against the quaternion, and the gyro against the quaternion's rate of change.
    quats = np.array([s.quat for s in samples if s.quat is not None])
    gravity = np.array([s.gravity for s in samples if s.gravity is not None])
    if len(quats) != len(samples) or len(gravity) != len(samples):
        print("[WARN] The robot does not report its orientation (quat) and projected gravity in every sample.")
    else:
        cos = np.sum(gravity_from_quat(quats) * gravity, axis=1) / np.linalg.norm(gravity, axis=1)
        angle = np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))
        verdict = "consistent" if angle.max() < 5.0 else "INCONSISTENT: check the quaternion's order (w, x, y, z)"
        print(f"[INFO] IMU: gravity z {gravity[:, 2].mean():+.3f}; quat vs gravity: max {angle.max():.1f} deg"
              f" ({verdict})")
        gyro = np.array([s.gyro for s in samples])
        rates = body_rates_from_quats(quats, np.maximum(np.diff(fw_time), 1e-4))
        measured = 0.5 * (gyro[1:] + gyro[:-1])
        moving = np.linalg.norm(rates, axis=1) > 0.3
        if moving.sum() < 10:
            print("[INFO]   gyro: not enough rotation to compare with the quaternion; tilt or turn the robot during"
                  " check to test it")
        else:
            cos = np.sum(rates[moving] * measured[moving], axis=1) / (
                np.linalg.norm(rates[moving], axis=1) * np.linalg.norm(measured[moving], axis=1) + 1e-9
            )
            ratio = np.linalg.norm(measured[moving], axis=1) / np.linalg.norm(rates[moving], axis=1)
            print(f"[INFO]   gyro vs quaternion rate ({moving.sum()} moving samples): direction cos"
                  f" {np.median(cos):+.2f}, magnitude ratio {np.median(ratio):.2f} (expect +1.00 and 1.00)")

    # Joints, in the policy's frame.
    joint_map = JointMap(list(MODEL_JOINTS.values()))
    lower, upper = load_joint_limits(args.urdf, joint_map.joint_names)
    pos = np.array([joint_map.to_joints(s.joint_pos) for s in samples])
    vel = np.array([joint_map.to_joints([j.vel if j.vel is not None else np.nan for j in s.joints]) for s in samples])
    default = None
    if args.run_dir:
        cfg = load_env_yaml(Path(args.run_dir).expanduser() / "params" / "env.yaml")
        default = per_joint(cfg["scene"]["robot"]["init_state"]["joint_pos"], joint_map.joint_names)
    print("[INFO] Joints in the robot model's frame (rad). 'default' is the run's default pose, 'motion' the range"
          " each joint moved during the check.")
    print(f"       {'joint':28s} {'position':>9s} {'velocity':>9s} {'limits':>15s} {'default':>8s} {'motion':>7s}")
    outside = []
    for i, name in enumerate(joint_map.joint_names):
        p = pos[:, i]
        flag = ""
        if p.mean() < lower[i] - 0.05 or p.mean() > upper[i] + 0.05:
            flag = "  OUTSIDE LIMITS"
            outside.append(name)
        fd = np.diff(p) / np.maximum(np.diff(fw_time), 1e-4)
        if np.ptp(p) > 0.1 and not np.isnan(vel[:, i]).any():
            scale = np.dot(vel[1:, i], fd) / max(np.dot(fd, fd), 1e-9)
            flag += f"  velocity/d(position)/dt {scale:+.2f}"
        print(f"       {name:28s} {p.mean():+9.3f} {np.nanmean(vel[:, i]):+9.3f}"
              f"  [{lower[i]:+.2f}, {upper[i]:+.2f}] "
              + (f"{default[i]:+8.3f}" if default is not None else f"{'':8s}")
              + f" {np.ptp(p):7.3f}{flag}")
    neck = [FIRMWARE_JOINTS.index(name) for name in NECK_JOINTS]
    print(f"       neck (yaw, pitch): {np.mean([[s.joint_pos[k] for k in neck] for s in samples], axis=0).round(3)}")
    if outside:
        print(f"[WARN] {len(outside)} joints read outside their URDF limits: {outside}. The robot's joint frame may"
              " differ from the model's for them (sign, zero or ankle conversion). Do not run a policy.")
    else:
        print("[INFO] Every joint reads within its URDF limits.")
    if np.isnan(vel).any():
        print("[WARN] The robot does not report every joint velocity; the policy needs them.")


# dry-run and run


def load_policy(onnx_path: Path):
    import onnxruntime

    session = onnxruntime.InferenceSession(str(onnx_path))
    name = session.get_inputs()[0].name
    return lambda obs: session.run(None, {name: obs[None]})[0][0]


class Log:
    """Every policy step, saved as arrays."""

    def __init__(self):
        self.steps: dict[str, list] = {}

    def add(self, **values) -> None:
        for key, value in values.items():
            self.steps.setdefault(key, []).append(value)

    def save(self, path: Path, **metadata) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        arrays = {key: np.array(value) for key, value in self.steps.items()}
        np.savez(path, **arrays, **{key: np.array(value) for key, value in metadata.items()})
        print(f"[INFO] Saved {len(next(iter(arrays.values()), []))} steps to {path}")


def stop_reason(state, sending: bool, step: int) -> str | None:
    from menlo.asimov import Mode

    if state.faulted:
        return "the robot faulted: " + (", ".join(a.name for a in state.alerts) or f"error flags {state.error_flags:#x}")
    if state.age_s > MAX_STATE_AGE_S:
        return f"the state is {state.age_s * 1e3:.0f} ms old"
    if state.quat is None or state.gyro is None:
        return "the robot did not report its orientation and angular velocity"
    if any(j.vel is None for j in state.joints):
        return "the robot did not report every joint velocity"
    # The first setpoint takes an armed STAND to MOVE; give the report a few steps to arrive.
    if sending and step > 10 and state.mode is not Mode.MOVE:
        return f"the robot left joint control: it reports {state.mode.name}"
    return None


def play(
    robot, controller, joint_map, kp, kd, *, send: bool, settle_s: float, hold_s: float, log: Log, stop: threading.Event
) -> str:
    """Step the policy at its rate on the freshest state, sending its targets if ``send``. Returns why it ended."""
    state = robot.get_state()
    neck = [state.joint_pos[i] for i in joint_map.neck_index]
    settle_steps = round(settle_s / controller.dt)
    controller.start(np.asarray(state.quat), settle_steps)
    steps = settle_steps + controller.frames - 1 + round(hold_s / controller.dt)
    start = time.perf_counter()
    tick = start
    for step in range(steps):
        while (wait := tick - time.perf_counter()) > 0:
            time.sleep(wait)
        woke = time.perf_counter()
        if stop.is_set():
            return "stopped by the user"
        state = robot.get_state()
        reason = stop_reason(state, send, step)
        if reason:
            return reason
        joint_pos = joint_map.to_joints(state.joint_pos)
        joint_vel = joint_map.to_joints([j.vel for j in state.joints])
        result = controller.step(np.asarray(state.quat), np.asarray(state.gyro), joint_pos, joint_vel)
        motor_target = joint_map.to_motors(result.target, neck)
        if send:
            robot.trajectory(motor_target, kp=kp, kd=kd)
        sent = time.perf_counter()
        log.add(
            time=woke - start,
            late=woke - tick,
            compute=sent - woke,
            state_age=state.age_s,
            state_sequence=state.sequence,
            fw_timestamp_us=state.fw_timestamp_us,
            mode=int(state.mode),
            quat=state.quat,
            gyro=state.gyro,
            gravity=state.gravity if state.gravity is not None else (np.nan,) * 3,
            motor_pos=state.joint_pos,
            motor_vel=[j.vel for j in state.joints],
            motor_current=[j.current if j.current is not None else np.nan for j in state.joints],
            motor_temp=[j.temp if j.temp is not None else np.nan for j in state.joints],
            alerts=",".join(a.name for a in state.alerts),
            joint_pos=joint_pos,
            joint_vel=joint_vel,
            observation=result.observation,
            action=result.action,
            target=result.target,
            clamped=result.clamped,
            motor_target=motor_target,
        )
        tick += controller.dt
        if time.perf_counter() > tick + controller.dt:
            tick = time.perf_counter()  # more than a step behind: start again from now rather than rush
    return "finished"


def summarize(log: Log, joint_names: list[str], reason: str) -> None:
    steps = log.steps
    if not steps:
        print(f"[INFO] Ended before the first step: {reason}")
        return
    print(f"[INFO] Ended after {len(steps['time'])} steps ({steps['time'][-1]:.2f} s): {reason}")
    print(f"[INFO]   step start lateness:  {stats_ms(steps['late'])}")
    print(f"[INFO]   state to sent:        {stats_ms(steps['compute'])}")
    print(f"[INFO]   state age when used:  {stats_ms(steps['state_age'])}")
    fw = np.array(steps["fw_timestamp_us"], dtype=np.float64) * 1e-6
    print(f"[INFO]   firmware time between the samples used: {stats_ms(np.diff(fw))}")
    clamped = np.array(steps["clamped"]).sum(axis=0)
    if clamped.any():
        print("[INFO]   targets clamped to the joint limits: "
              + ", ".join(f"{joint_names[i]} {int(n)}x" for i, n in enumerate(clamped) if n))
    gravity_z = np.array(steps["gravity"])[:, 2]
    print(f"[INFO]   highest projected gravity z (tilt): {np.nanmax(gravity_z):+.3f} (fall trip above {FALL_GRAVITY_Z})")
    alerts = sorted({a for entry in steps["alerts"] for a in entry.split(",") if a})
    print(f"[INFO]   alerts seen: {', '.join(alerts) or 'none'}")


def play_command(args) -> None:
    from menlo.asimov import MenloError, Mode

    from cyclotron.sim2real import FIRMWARE_JOINTS, JointMap, TrackingController, load_joint_limits
    from cyclotron.sim2sim import TrackingRun

    sending = args.command == "run"
    run_dir = Path(args.run_dir).expanduser()
    onnx_path = Path(args.onnx).expanduser() if args.onnx else run_dir / "exported" / "policy.onnx"
    if not onnx_path.is_file():
        sys.exit(f"[ERROR] No policy at {onnx_path}. Export one with ./cyclotron.sh --export --checkpoint"
                 f" {run_dir}/model_<N>.pt")
    run = TrackingRun.load(run_dir)
    joint_map = JointMap(run.joint_names)
    controller = TrackingController(run, load_policy(onnx_path), load_joint_limits(args.urdf, run.joint_names))
    kp, kd = joint_map.gains(run.stiffness, run.damping)
    record = Path(args.record).expanduser() if args.record else (
        run_dir / "sim2real" / f"{args.command}_{datetime.now():%Y-%m-%d_%H-%M-%S}.npz"
    )
    log = Log()
    stop = threading.Event()
    reason = "not started"

    with connect(args) as robot:
        check_joint_table(robot)
        state = wait_for_state(robot)
        # menlo-sdk counts arming (STAND upright for 0.5 s) from its own samples, so give it a moment after connecting.
        deadline = time.monotonic() + 1.5
        while state.mode is Mode.STAND and robot.armed is not True and time.monotonic() < deadline:
            time.sleep(0.05)
            state = robot.get_state()
        print(f"[INFO] Robot: {describe(state, robot)}")
        print(f"[INFO] Policy: {onnx_path}; reference {Path(run.motion_file).name}, {controller.frames} frames at"
              f" {1 / controller.dt:g} Hz ({(controller.frames - 1) * controller.dt:.2f} s)")
        if sending:
            hottest = max((j.temp for j in state.joints if j.temp is not None), default=None)
            problems = []
            if state.faulted:
                problems.append("the robot is faulted (latched DAMP until the firmware restarts)")
            if hottest is not None and hottest >= MAX_START_TEMPERATURE_C:
                problems.append(f"an actuator is at {hottest:.0f} C")
            if state.battery is not None and state.battery.soc_percent < MIN_START_BATTERY_PERCENT:
                problems.append(f"the battery is at {state.battery.soc_percent:.0f} %")
            if state.mode is Mode.MOVE and not args.from_move:
                problems.append("the robot is in MOVE; pass --from_move to switch the walking policy off and start")
            elif state.mode is Mode.STAND and robot.armed is not True:
                problems.append("the robot is in STAND but not armed (upright for 0.5 s) yet")
            elif state.mode not in (Mode.STAND, Mode.MOVE):
                problems.append(f"the robot is in {state.mode.name}: put it in STAND first (menlo stand), hanging"
                                " from its gantry hook with both feet on the floor")
            if problems:
                sys.exit("[ERROR] Not starting: " + "; ".join(problems) + ".")
            plan = f"blend the joints to the first frame's pose over {args.blend:g} s, " if args.blend > 0 else ""
            print(f"[INFO] Plan: {plan}track the first frame for {args.settle:g} s, play the motion, hold its last"
                  f" frame for {args.hold:g} s, then"
                  f" {'DAMP' if args.end == 'damp' else 'hand back to the walking policy (balance)'}.")
            if not args.yes:
                print("[WARN] The robot must hang from its gantry hook with both feet on the floor. Keep the E-Stop"
                      " in Asimov Manager at hand. Ctrl+C DAMPs the robot.")
                if input("Type 'go' to start: ").strip() != "go":
                    sys.exit("[INFO] Not started.")

            if args.blend > 0:
                neck = [state.joint_pos[i] for i in joint_map.neck_index]
                first = joint_map.to_motors(controller.first_pose(), neck)
                try:
                    robot.set_joints(first, duration=args.blend, kp=kp, kd=kd, wait=True, timeout=args.blend + 3.0)
                except MenloError as exc:  # not there in time, a fault or a stale stream: the checks below decide
                    print(f"[WARN] The blend did not complete: {exc}")
                state = robot.get_state()
                error = np.abs(joint_map.to_joints(state.joint_pos) - controller.first_pose())
                print(f"[INFO] After the blend, the largest joint error is {error.max():.3f} rad"
                      f" ({run.joint_names[int(np.argmax(error))]})")
            state = robot.get_state()
            if state.faulted or state.age_s > MAX_STATE_AGE_S or state.gravity is None or state.gravity[2] > -0.9:
                with contextlib.suppress(MenloError):
                    robot.damp()
                sys.exit(f"[ERROR] Not starting the policy: {describe(state, robot)}, gravity z"
                         f" {state.gravity[2] if state.gravity else float('nan'):+.2f}, state"
                         f" {state.age_s * 1e3:.0f} ms old. Sent DAMP.")

        try:
            reason = play(
                robot,
                controller,
                joint_map,
                kp,
                kd,
                send=sending,
                settle_s=args.settle,
                hold_s=args.hold,
                log=log,
                stop=stop,
            )
        except KeyboardInterrupt:
            reason = "stopped by the user (Ctrl+C)"
        finally:
            if sending:
                try:
                    if reason == "finished" and args.end == "balance":
                        robot.balance()
                        print("[INFO] Sent balance(): the walking policy should be balancing the robot.")
                    else:
                        robot.damp()
                        print("[INFO] The robot is in DAMP.")
                except Exception as exc:  # the link may be gone; Asimov Edge DAMPs 2 s after the last setpoint
                    print(f"[WARN] Could not end joint control ({exc}); Asimov Edge DAMPs the robot 2 s after the"
                          " last setpoint.")
            summarize(log, run.joint_names, reason)
            log.save(
                record,
                joint_names=run.joint_names,
                firmware_joints=list(FIRMWARE_JOINTS),
                kp=kp,
                kd=kd,
                run_dir=str(run_dir),
                onnx=str(onnx_path),
                motion_file=run.motion_file,
                command=args.command,
                reason=reason,
                dt=controller.dt,
            )


def main() -> None:
    args = parser.parse_args()
    try:
        from menlo.asimov import ConnectError
    except ImportError:
        sys.exit("[ERROR] sim2real needs menlo-sdk (Python 3.12+): run it with ./cyclotron.sh --sim2real.")
    try:
        if args.command == "check":
            check(args)
        else:
            play_command(args)
    except ConnectError as exc:
        sys.exit(f"[ERROR] Could not connect to the robot: {exc}")


if __name__ == "__main__":
    main()
