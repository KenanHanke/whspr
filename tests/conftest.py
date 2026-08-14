"""Shared fixtures for the whspr test suite.

The server tests run real server processes (spawned with a fake, instantly
loading model unless stated otherwise) against per-test lock/socket paths so
they cannot collide with each other or with a real whspr installation.
"""

import os
import signal
import socket
import subprocess
import sys
import time

import pytest

import whspr.server as server


# A wrapper program that runs the real server main() with a configurable fake
# model, so lifecycle tests are fast and deterministic.  argv:
#   1: seconds the fake model load takes
#   2: "ok" or "fail" (whether the model load succeeds)
#   3: seconds each fake transcription takes
#   4: idle timeout to use, in seconds
#   5: path of a log file that receives one line per lifecycle event
#   6: wedged-job watchdog timeout, in seconds
FAKE_SERVER_PROGRAM = """
import sys, time, types
import whspr.server as server

load_delay = float(sys.argv[1])
load_ok = sys.argv[2] == "ok"
job_delay = float(sys.argv[3])
server.IDLE_TIMEOUT = float(sys.argv[4])
log_path = sys.argv[5]
server._WEDGED_JOB_TIMEOUT = float(sys.argv[6])

def log(event):
    with open(log_path, "a") as f:
        f.write(event + "\\n")

class FakeModel:
    def transcribe(self, path):
        log("job-start " + path)
        time.sleep(job_delay)
        with open(path, "r") as f:
            content = f.read()
        log("job-end " + path)
        return iter([types.SimpleNamespace(text=content)]), None

def fake_load_model():
    time.sleep(load_delay)
    if not load_ok:
        raise RuntimeError("fake model load failure")
    log("model-loaded")
    return FakeModel()

server.load_model = fake_load_model
log("server-starting")
status = server.main()
log("server-exited")
sys.exit(status)
"""


def wait_until(predicate, timeout=10.0, interval=0.05, message="condition"):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return
        time.sleep(interval)
    raise AssertionError(f"timed out after {timeout}s waiting for: {message}")


class ServerHarness:
    """Spawns fake-model server processes against an isolated runtime dir."""

    def __init__(self, runtime_dir, tmp_path):
        self.runtime_dir = runtime_dir
        self.tmp_path = tmp_path
        self.log_path = str(tmp_path / "server-events.log")
        self.processes = []

    def env(self):
        env = os.environ.copy()
        env["XDG_RUNTIME_DIR"] = self.runtime_dir
        return env

    def spawn(
        self,
        load_delay=0.0,
        load_ok=True,
        job_delay=0.0,
        idle_timeout=300.0,
        wedge_timeout=3600.0,
    ):
        process = subprocess.Popen(
            [
                sys.executable,
                "-c",
                FAKE_SERVER_PROGRAM,
                str(load_delay),
                "ok" if load_ok else "fail",
                str(job_delay),
                str(idle_timeout),
                self.log_path,
                str(wedge_timeout),
            ],
            env=self.env(),
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        self.processes.append(process)
        return process

    def spawn_and_wait_ready(self, **kwargs):
        process = self.spawn(**kwargs)
        wait_until(
            lambda: os.path.exists(server.SOCKET_PATH) and server.is_running(),
            message="server socket to appear",
        )
        return process

    def events(self):
        try:
            with open(self.log_path) as f:
                return [line.strip() for line in f if line.strip()]
        except FileNotFoundError:
            return []

    def make_fake_start(self, **spawn_kwargs):
        """A drop-in replacement for server.start() that spawns a fake server."""

        def fake_start():
            if not server.is_running():
                self.spawn(**spawn_kwargs)

        return fake_start

    def write_audio_stub(self, name="audio.txt", content="hello from the fake model"):
        path = self.tmp_path / name
        path.write_text(content)
        return str(path), content

    def terminate_all(self):
        for process in self.processes:
            if process.poll() is None:
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except (ProcessLookupError, PermissionError):
                    pass
            process.wait()


@pytest.fixture
def harness(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "runtime"
    runtime_dir.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setattr(server, "LOCK_PATH", os.path.join(str(runtime_dir), "whspr-server.lock"))
    monkeypatch.setattr(server, "SOCKET_PATH", os.path.join(str(runtime_dir), "whspr-server.sock"))
    # Keep tests away from the real machine-global legacy paths.
    monkeypatch.setattr(server, "_LEGACY_SOCKET_PATH", os.path.join(str(runtime_dir), "legacy.sock"))
    monkeypatch.setattr(server, "_LEGACY_LOCK_PATH", os.path.join(str(runtime_dir), "legacy.lock"))
    h = ServerHarness(str(runtime_dir), tmp_path)
    try:
        yield h
    finally:
        h.terminate_all()


def raw_request(payload_bytes, timeout=10.0):
    """Send raw bytes to the server socket and return everything it replies."""
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.settimeout(timeout)
        conn.connect(server.SOCKET_PATH)
        chunks = []
        try:
            if payload_bytes:
                conn.sendall(payload_bytes)
            while True:
                chunk = conn.recv(65536)
                if not chunk:
                    break
                chunks.append(chunk)
        except (socket.timeout, ConnectionError):
            # The server may hang up mid-send (e.g. on oversized messages);
            # whatever reply arrived before that is still returned.
            pass
        return b"".join(chunks)
