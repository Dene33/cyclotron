"""Stand in for the real Asimov 1 when testing scripts/sim2real.py: Asimov 1 in MuJoCo behind Asimov Edge's UDP protocol.

It speaks what menlo-sdk's ``udp`` connection mode speaks: RobotCommand datagrams in on the command port, RobotState
datagrams out to one state host at the firmware's 200 Hz, and the HTTP version endpoint. The robot follows the
firmware's mode rules as menlo-sdk documents them (and its test suite's FakeEdge models):

- DAMP: no joint holds a position; a velocity or trajectory is dropped.
- STAND: every joint blends to the standing pose over 2 s and holds it, with no balance loop. The robot arms once
  upright (projected gravity z below -0.87) for 0.5 s.
- A trajectory or velocity in an armed STAND enters MOVE. Trajectory targets are PD targets with the command's gains
  (or the gain table); an ankle's two entries are its pitch and roll, limited to 0.35 and 0.1 rad. The robot DAMPs 2 s
  after the last setpoint.
- A velocity in MOVE: there is no walking policy here, so the joints hold the standing pose, without balance.
- A fall (projected gravity z above -0.5) or a joint past its limit latches DAMP, with error flags set, until restart.

State reports the motors as the robot does, the ankles as their motors A and B, and a trajectory's ankle gains are
motor gains, seen at the ankle joints through the linkage (``cyclotron.sim2real.ANKLE_GAIN_FACTORS``). The robot can
face any heading (``--yaw``) to test the reference's alignment, and commands and state can be delayed. The physics,
gain table, standing pose and effort limits come from a training run's env.yaml, as in sim2sim. ``--record`` saves the robot's
motion at 50 Hz in the format ``./cyclotron.sh --render`` reads.
"""

import argparse
import json
import socket
import sys
import threading
import time
from collections import deque
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

import numpy as np
from scipy.spatial.transform import Rotation

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT / "source" / "cyclotron"))

from cyclotron.sim2real import FIRMWARE_JOINTS, JointMap, heading, load_joint_limits  # noqa: E402
from cyclotron.sim2sim import PolicyRun, load_reference  # noqa: E402

DEFAULT_MODEL_DIR = REPOSITORY_ROOT / "third_party" / "asimov-1" / "sim-model"
DAMP, STAND, MOVE = 0, 1, 2
FALL_DETECTED, JOINT_OUT_OF_RANGE = 7, 11
ANKLE_LIMITS = {"pitch": 0.35, "roll": 0.1}
STAND_BLEND_S = 2.0
ARM_GRAVITY_Z, ARM_HOLD_S = -0.87, 0.5
FALL_GRAVITY_Z = -0.5
TRAJECTORY_TIMEOUT_S = 2.0
JOINT_LIMIT_MARGIN = 0.05
DAMP_DAMPING = 1.0  # N·m·s/rad: in DAMP no joint holds a position

parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
parser.add_argument("run_dir", type=str, help="Training run whose env.yaml gives the gains, standing pose and limits.")
parser.add_argument("--xml", type=str, default=str(DEFAULT_MODEL_DIR / "xmls" / "asimov_1.xml"), help="MuJoCo model.")
parser.add_argument("--urdf", type=str, default=str(DEFAULT_MODEL_DIR / "urdf" / "asimov_1.urdf"), help="Joint limits.")
parser.add_argument("--host", type=str, default="127.0.0.1", help="Address for the command port and version endpoint.")
parser.add_argument("--command_port", type=int, default=8850)
parser.add_argument("--state_host", type=str, default="127.0.0.1", help="Where to send state (udp-state-host).")
parser.add_argument("--state_port", type=int, default=8851)
parser.add_argument("--version_port", type=int, default=3000)
parser.add_argument("--yaw", type=float, default=0.7, help="The robot's heading (rad); the IMU's yaw zero is 0.")
parser.add_argument("--command_delay", type=float, default=0.0, help="Milliseconds before a command takes effect.")
parser.add_argument("--state_delay", type=float, default=0.0, help="Milliseconds before a state sample is sent.")
parser.add_argument(
    "--start",
    choices=("stand", "damp", "reference"),
    default="stand",
    help="At start: STAND, DAMP, or (to compare with sim2sim) on the reference's first frame, armed, with the physics"
    " paused until the first trajectory.",
)
parser.add_argument("--duration", type=float, default=None, help="Seconds to run. Default: until Ctrl+C.")
parser.add_argument("--record", type=str, default=None, help="Save the robot's motion (.npz) here on exit.")
parser.add_argument("--viewer", action="store_true", help="Show the robot in MuJoCo's viewer.")


def min_jerk(t: float) -> float:
    t = min(max(t, 0.0), 1.0)
    return 10 * t**3 - 15 * t**4 + 6 * t**5


class Robot:
    """Asimov 1 in MuJoCo with the firmware's mode machine."""

    def __init__(self, args, run: PolicyRun):
        import mujoco

        self.mujoco = mujoco
        self.run = run
        self.model = mujoco.MjModel.from_xml_path(args.xml)
        self.model.opt.timestep = run.physics_dt
        self.data = mujoco.MjData(self.model)
        self.qpos = np.array([self.model.jnt_qposadr[self.model.joint(n).id] for n in run.joint_names])
        self.qvel = np.array([self.model.jnt_dofadr[self.model.joint(n).id] for n in run.joint_names])
        known = ~np.isnan(run.armature)  # the run's motor inertia, as in sim2sim
        self.model.dof_armature[self.qvel[known]] = run.armature[known]
        self.joint_map = JointMap(run.joint_names)
        self.lower, self.upper = load_joint_limits(args.urdf, run.joint_names)
        self.stand_pose = run.default_joint_pos.copy()

        self.data.qpos[0:3] = (0.0, 0.0, run.root_height)
        self.data.qpos[3:7] = Rotation.from_euler("z", args.yaw).as_quat()[[3, 0, 1, 2]]
        self.data.qpos[self.qpos] = self.stand_pose
        mujoco.mj_forward(self.model, self.data)

        self.time = 0.0
        self.mode = DAMP
        self.armed = False
        self.upright_since: float | None = None
        self.error_flags = 0
        self.alerts: list[int] = []
        self.blend_from = self.stand_pose.copy()
        self.blend_start = 0.0
        self.trajectory: np.ndarray | None = None
        self.kp, self.kd = run.stiffness.copy(), run.damping.copy()
        self.last_setpoint = 0.0
        self.last_cmd_sequence = 0
        self.warned_no_walking_policy = False
        self.paused = False
        if args.start in ("stand", "reference"):
            self.enter(STAND)
        if args.start == "reference":
            reference = load_reference(run)
            turn = Rotation.from_euler("z", args.yaw - heading(reference["pelvis_quat"][0]))
            self.data.qpos[0:3] = (0.0, 0.0, reference["pelvis_pos"][0, 2])
            self.data.qpos[3:7] = (turn * Rotation.from_quat(reference["pelvis_quat"][0][[1, 2, 3, 0]])).as_quat()[
                [3, 0, 1, 2]
            ]
            self.data.qpos[self.qpos] = reference["joint_pos"][0]
            mujoco.mj_forward(self.model, self.data)
            self.armed, self.paused = True, True

    # The firmware's mode machine.

    def enter(self, mode: int) -> None:
        if mode == self.mode:
            return
        if not (self.mode == MOVE and mode == STAND):
            self.armed, self.upright_since = False, None
        if mode == STAND:
            self.blend_from, self.blend_start = self.data.qpos[self.qpos].copy(), self.time
        if mode != MOVE:
            self.trajectory = None
        self.mode = mode

    def fault(self, alert: int, why: str) -> None:
        if not self.error_flags:
            print(f"[mujoco_edge] t={self.time:.2f} s: {why}; DAMP latched until restart")
        self.error_flags |= 1 | (1 << (1 + alert))
        if alert not in self.alerts:
            self.alerts.append(alert)
        self.enter(DAMP)

    def command(self, c) -> None:
        self.last_cmd_sequence = c.sequence
        if self.error_flags:
            return  # latched: DAMP until the firmware restarts
        drive = c.HasField("policy") or c.HasField("all_trajectory")
        if not drive:
            if c.mode == STAND:
                self.enter(STAND)
            elif c.mode == DAMP:
                self.enter(DAMP)
            return
        if self.mode == DAMP:
            return  # Asimov Edge drops a velocity or trajectory in DAMP
        if self.mode == STAND:
            if not self.armed:
                return  # stays in STAND until armed
            self.enter(MOVE)
        if c.HasField("all_trajectory"):
            wire = np.asarray(c.all_trajectory.positions, dtype=np.float64)
            target = wire[self.joint_map.motor_index].copy()  # an ankle's entries are already its pitch and roll
            for _, _, pitch, roll in self.joint_map.ankles:
                target[pitch] = np.clip(target[pitch], -ANKLE_LIMITS["pitch"], ANKLE_LIMITS["pitch"])
                target[roll] = np.clip(target[roll], -ANKLE_LIMITS["roll"], ANKLE_LIMITS["roll"])
            self.trajectory = target
            self.paused = False
            if len(c.all_trajectory.kp) == len(FIRMWARE_JOINTS):
                # Motor gains; the ankle motors' are seen at the ankle joints through the linkage.
                self.kp, self.kd = self.joint_map.joint_gains(c.all_trajectory.kp, c.all_trajectory.kd)
            else:
                self.kp, self.kd = self.run.stiffness.copy(), self.run.damping.copy()
            self.last_setpoint = self.time
        else:
            self.trajectory = None
            if not self.warned_no_walking_policy:
                print("[mujoco_edge] A velocity in MOVE: there is no walking policy here; holding the standing pose")
                self.warned_no_walking_policy = True

    def gravity(self) -> np.ndarray:
        return Rotation.from_quat(self.data.qpos[[4, 5, 6, 3]]).inv().apply([0.0, 0.0, -1.0])

    def step(self) -> None:
        if self.paused:
            self.time += self.run.physics_dt
            return
        q, dq = self.data.qpos[self.qpos], self.data.qvel[self.qvel]
        gravity_z = self.gravity()[2]
        if gravity_z < ARM_GRAVITY_Z:
            self.upright_since = self.time if self.upright_since is None else self.upright_since
            if self.mode == STAND and not self.armed and self.time - self.upright_since >= ARM_HOLD_S:
                self.armed = True
        else:
            self.upright_since = None
        if self.mode in (STAND, MOVE):
            if gravity_z > FALL_GRAVITY_Z:
                self.fault(FALL_DETECTED, f"fall detected (projected gravity z {gravity_z:+.2f})")
            outside = (q < self.lower - JOINT_LIMIT_MARGIN) | (q > self.upper + JOINT_LIMIT_MARGIN)
            if outside.any():
                self.fault(JOINT_OUT_OF_RANGE, f"{np.array(self.run.joint_names)[outside].tolist()} out of range")
        if self.mode == MOVE and self.trajectory is not None and self.time - self.last_setpoint > TRAJECTORY_TIMEOUT_S:
            print(f"[mujoco_edge] t={self.time:.2f} s: no setpoint for {TRAJECTORY_TIMEOUT_S:g} s; DAMP")
            self.enter(DAMP)

        if self.mode == DAMP:
            torque = -DAMP_DAMPING * dq
        elif self.mode == MOVE and self.trajectory is not None:
            torque = self.kp * (self.trajectory - q) - self.kd * dq
        else:
            blend = min_jerk((self.time - self.blend_start) / STAND_BLEND_S) if self.mode == STAND else 1.0
            target = self.blend_from + blend * (self.stand_pose - self.blend_from)
            torque = self.run.stiffness * (target - q) - self.run.damping * dq
        self.data.qfrc_applied[:] = 0.0
        self.data.qfrc_applied[self.qvel] = np.clip(torque, -self.run.effort_limit, self.run.effort_limit)
        self.mujoco.mj_step(self.model, self.data)
        self.time += self.run.physics_dt

    def state(self, pb, sequence: int) -> bytes:
        s = pb.state.RobotState(
            timestamp_us=int(round(self.time * 1e6)),
            sequence=sequence,
            current_mode=self.mode,
            error_flags=self.error_flags,
            last_cmd_sequence=self.last_cmd_sequence,
            protocol_version=1,
        )
        s.joint_pos.extend(self.joint_map.to_motors(self.data.qpos[self.qpos]))
        s.joint_vel.extend(self.joint_map.to_motors(self.data.qvel[self.qvel]))
        s.joint_current.extend([0.0] * len(FIRMWARE_JOINTS))
        s.joint_temp.extend([35.0] * len(FIRMWARE_JOINTS))
        s.base_ang_vel.extend(self.data.qvel[3:6])  # free joint angular velocity, in the pelvis frame
        s.projected_gravity.extend(self.gravity())
        s.base_quat.extend(self.data.qpos[3:7])  # w, x, y, z
        for alert in self.alerts:
            s.active_alerts.add(id=alert, severity=0)
        return s.SerializeToString()


def serve_version(host: str, port: int) -> HTTPServer:
    body = json.dumps({"robot_model": "asimov_1", "robot_os_version": "0.2.0"}).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200 if self.path == "/api/version" else 404)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *_):
            pass

    server = HTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server


def main() -> None:
    from menlo.asimov._proto import load

    args = parser.parse_args()
    sys.stdout.reconfigure(line_buffering=True)  # status lines show up at once, also when redirected to a file
    pb = load()
    run = PolicyRun.load(args.run_dir)
    robot = Robot(args, run)
    version = serve_version(args.host, args.version_port)
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind((args.host, args.command_port))
    sock.setblocking(False)
    state_to = (args.state_host, args.state_port)
    commands: deque = deque()
    states: deque = deque()
    log = {"root_pos": [], "root_quat": [], "joint_pos": []}
    record_every = run.decimation  # 50 Hz
    viewer, viewer_threads = None, set()
    if args.viewer:
        import mujoco.viewer

        before = set(threading.enumerate())
        viewer = mujoco.viewer.launch_passive(robot.model, robot.data)
        # Its render thread, to wait for on exit: GLFW shutting down under it crashes the process.
        viewer_threads = set(threading.enumerate()) - before
    print(f"[mujoco_edge] Asimov Edge on {args.host}:{args.command_port} (version endpoint :{args.version_port}),"
          f" state to {state_to[0]}:{state_to[1]} at {1 / run.physics_dt:g} Hz; robot in"
          f" {'STAND' if robot.mode == STAND else 'DAMP'}, heading {args.yaw:+.2f} rad. Ctrl+C to stop.")

    start = time.perf_counter()
    tick, step = start, 0
    try:
        while args.duration is None or robot.time < args.duration:
            now = time.perf_counter()
            while True:
                try:
                    payload = sock.recv(65535)
                except BlockingIOError:
                    break
                commands.append((now + args.command_delay * 1e-3, payload))
            while commands and commands[0][0] <= now:
                c = pb.command.RobotCommand()
                c.ParseFromString(commands.popleft()[1])
                robot.command(c)
            robot.step()
            step += 1
            states.append((now + args.state_delay * 1e-3, robot.state(pb, step)))
            while states and states[0][0] <= time.perf_counter():
                sock.sendto(states.popleft()[1], state_to)
            if step % record_every == 0:
                log["root_pos"].append(robot.data.qpos[0:3].copy())
                log["root_quat"].append(robot.data.qpos[3:7].copy())
                log["joint_pos"].append(robot.data.qpos[robot.qpos].copy())
                if viewer is not None:
                    viewer.sync()
            tick += run.physics_dt
            if (wait := tick - time.perf_counter()) > 0:
                time.sleep(wait)
            elif wait < -0.05:
                tick = time.perf_counter()  # far behind real time: do not try to catch up
    except KeyboardInterrupt:
        pass
    finally:
        version.shutdown()
        sock.close()
        if viewer is not None:
            viewer.close()
            for thread in viewer_threads:
                thread.join(timeout=5.0)
        if args.record:
            rollout = {key: np.array(value) for key, value in log.items()}
            rollout["frame"] = np.arange(len(rollout["root_pos"]))
            rollout["fps"] = np.array(1.0 / (run.physics_dt * run.decimation))
            rollout["joint_names"] = np.array(run.joint_names)
            Path(args.record).expanduser().parent.mkdir(parents=True, exist_ok=True)
            np.savez(Path(args.record).expanduser(), **rollout)
            print(f"[mujoco_edge] Recorded {len(rollout['frame'])} frames to {args.record}")


if __name__ == "__main__":
    main()
