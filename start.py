"""
Fight Detection — Unified launcher
Automatically sets up the hfarmai server and starts the detection pipeline.

Usage:
    python start.py                   # webcam (default)
    python start.py --source robot    # Unitree Go2 via WiFi
    python start.py --source video --video path.mp4
"""

import subprocess
import sys
import time
import os
import argparse
import signal
import requests

HFARMAI_REPO_URL = "https://github.com/hfarmai/unitree-go2-control"
HFARMAI_CLONE_DIR = os.path.expanduser("~/unitree-go2-control")
HFARMAI_PATH = os.path.join(HFARMAI_CLONE_DIR, "go2_control")
HFARMAI_SERVER = os.path.join(HFARMAI_PATH, "server.py")
HFARMAI_PORT = 8080
HFARMAI_HEALTH_URL = f"http://localhost:{HFARMAI_PORT}/api/video/stream"
HFARMAI_STARTUP_TIMEOUT = 45

_processes = []


def cleanup(sig=None, frame=None):
    print("\n  Shutting down...")
    for p in _processes:
        try:
            p.terminate()
            p.wait(timeout=3)
        except Exception:
            p.kill()
    print("  Done.")
    sys.exit(0)


signal.signal(signal.SIGINT, cleanup)
signal.signal(signal.SIGTERM, cleanup)


def check_hfarmai_installed() -> bool:
    return os.path.exists(HFARMAI_SERVER)


def wait_for_hfarmai_boot(timeout: float = 10) -> bool:
    status_url = f"http://localhost:{HFARMAI_PORT}/api/status"
    start = time.time()
    while time.time() - start < timeout:
        try:
            r = requests.get(status_url, timeout=1)
            if r.status_code == 200:
                return True
        except Exception:
            pass
        time.sleep(0.5)
    return False


def wait_for_hfarmai(robot_ip: str, timeout: float = HFARMAI_STARTUP_TIMEOUT) -> bool:
    print(f"  Waiting for hfarmai server (max {timeout}s)...")

    if not wait_for_hfarmai_boot():
        print("  Flask server not responding")
        return False

    print("  Flask OK — checking robot connection...")

    try:
        r = requests.get(f"http://localhost:{HFARMAI_PORT}/api/status", timeout=3)
        connected = r.json().get("connected", False)
    except Exception:
        connected = False

    if not connected:
        print(f"  Connecting to robot {robot_ip}...")
        try:
            r = requests.post(
                f"http://localhost:{HFARMAI_PORT}/api/connect",
                json={"ip": robot_ip},
                timeout=45
            )
            connected = r.status_code == 200
            if not connected:
                print(f"  /api/connect returned {r.status_code}: {r.text[:100]}")
        except Exception as e:
            print(f"  /api/connect error: {e}")

    if not connected:
        return False

    print("  Waiting for video stream (H264 sync)...")
    start = time.time()
    attempt = 0
    while time.time() - start < 30:
        attempt += 1
        try:
            r = requests.get(HFARMAI_HEALTH_URL, timeout=3, stream=True)
            if r.status_code == 200:
                content = next(r.iter_content(chunk_size=512), None)
                r.close()
                if content:
                    print(f"  Video stream ready (attempt {attempt})")
                    return True
        except Exception:
            pass
        time.sleep(1.0)

    print("  Video stream not available after 30s")
    return False


def ensure_hfarmai_installed():
    """Auto-clones and installs hfarmai if not present."""
    if check_hfarmai_installed():
        return

    print(f"  hfarmai server not found — cloning automatically...")
    print(f"  Repo: {HFARMAI_REPO_URL}")

    try:
        subprocess.run(["git", "--version"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ERROR: git is not installed.")
        print("  Install git from https://git-scm.com/ and try again.")
        sys.exit(1)

    try:
        subprocess.run(
            ["git", "clone", "--depth=1", HFARMAI_REPO_URL, HFARMAI_CLONE_DIR],
            check=True
        )
        print("  Clone complete.")
    except subprocess.CalledProcessError:
        print("  ERROR: clone failed. Check your internet connection.")
        sys.exit(1)

    if not check_hfarmai_installed():
        print(f"  ERROR: server.py not found in {HFARMAI_PATH} after clone.")
        sys.exit(1)

    req_file = os.path.join(HFARMAI_CLONE_DIR, "requirements.txt")
    if os.path.exists(req_file):
        print("  Installing hfarmai dependencies...")
        try:
            subprocess.run(
                [sys.executable, "-m", "pip", "install", "-r", req_file, "-q"],
                check=True
            )
            print("  Dependencies installed.")
        except subprocess.CalledProcessError as e:
            print(f"  Warning: pip install failed ({e}) — continuing anyway.")

    print("  hfarmai ready.")


def start_hfarmai_server() -> subprocess.Popen:
    ensure_hfarmai_installed()

    print("  Starting hfarmai server...")
    proc = subprocess.Popen(
        [sys.executable, HFARMAI_SERVER],
        cwd=HFARMAI_PATH,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        bufsize=1,
        text=True
    )
    _processes.append(proc)
    return proc


def stream_server_logs(proc: subprocess.Popen):
    import threading
    _IMPORTANT = ["error", "connected", "stream", "video", "failed", "refused"]

    def _read():
        for line in proc.stdout:
            l = line.rstrip()
            if any(k in l.lower() for k in _IMPORTANT):
                print(f"  [hfarmai] {l}")

    threading.Thread(target=_read, daemon=True).start()


def start_pipeline(source: str, video_path: str = None):
    script = os.path.join(os.path.dirname(__file__), "main.py")
    cmd = [sys.executable, script, "--source", source]
    if video_path:
        cmd += ["--video", video_path]

    print(f"  Starting pipeline: {' '.join(cmd)}")
    proc = subprocess.Popen(cmd, cwd=os.path.dirname(__file__))
    _processes.append(proc)
    return proc


def update_robot_ip_from_discovery():
    print("  Scanning for robot on the network...")
    from discovery import discover_robot
    candidates = discover_robot()
    if not candidates:
        print("  Robot not found on the network.")
        print("  Make sure the robot and PC are on the same WiFi network.")
        sys.exit(1)

    robot_ip = candidates[0]["ip"]
    stream_url = f"http://localhost:{HFARMAI_PORT}/api/video/stream"

    print(f"  Robot found: {robot_ip}")

    # Save to .env
    env_path = os.path.join(os.path.dirname(__file__), ".env")
    if os.path.exists(env_path):
        with open(env_path, "r") as f:
            content = f.read()
        import re
        for key, val in [("ROBOT_IP", robot_ip), ("ROBOT_STREAM_URL", stream_url), ("VIDEO_SOURCE", "robot")]:
            if re.search(rf"^{key}=.*$", content, re.MULTILINE):
                content = re.sub(rf"^{key}=.*$", f"{key}={val}", content, flags=re.MULTILINE)
            else:
                content += f"\n{key}={val}"
        with open(env_path, "w") as f:
            f.write(content)
        print("  .env updated with robot IP")

    return robot_ip


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fight Detection — Launcher")
    parser.add_argument("--source", type=str, default="webcam",
                        choices=["webcam", "robot", "video"])
    parser.add_argument("--video", type=str, default=None)
    parser.add_argument("--skip-discovery", action="store_true",
                        help="Use ROBOT_IP from .env without scanning the network")
    args = parser.parse_args()

    print("=" * 50)
    print("  Fight Detection — Launcher")
    print("=" * 50)

    if args.source == "robot":
        if not args.skip_discovery:
            robot_ip = update_robot_ip_from_discovery()
        else:
            from config import ROBOT_IP
            robot_ip = ROBOT_IP
            if not robot_ip:
                print("  ROBOT_IP not set in .env — remove --skip-discovery to auto-detect")
                sys.exit(1)
            print(f"  Using IP from .env: {robot_ip}")

        server_proc = start_hfarmai_server()
        stream_server_logs(server_proc)

        if not wait_for_hfarmai(robot_ip=robot_ip):
            print("  Robot connection failed — make sure it is powered on and on the same network")
            cleanup()

    pipeline_proc = start_pipeline(args.source, args.video)

    try:
        pipeline_proc.wait()
    except KeyboardInterrupt:
        cleanup()

    cleanup()
