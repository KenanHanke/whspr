"""Command-line tests for `whspr --stop-server` (and argument handling).

These run the real `python -m whspr` entry point in a subprocess against
fake-model servers in an isolated runtime dir, so they are fast.
"""

import os
import subprocess
import sys
import threading
import time

import pytest

import whspr.server as server

from conftest import wait_until
from test_server_lifecycle import hold_server_lock


# Runs `python -m whspr` exactly (argv included), except that the legacy
# pre-1.1 server paths point into the test's runtime dir instead of the
# machine-wide /tmp, which the stop command also probes.
CLI_PROGRAM = """
import os, runpy, sys
import whspr.server as server
runtime_dir = os.environ["XDG_RUNTIME_DIR"]
server._LEGACY_SOCKET_PATH = os.path.join(runtime_dir, "legacy.sock")
server._LEGACY_LOCK_PATH = os.path.join(runtime_dir, "legacy.lock")
sys.argv[0] = "whspr"
runpy.run_module("whspr", run_name="__main__", alter_sys=True)
"""


def run_cli(harness, *args, timeout=60):
    return subprocess.run(
        [sys.executable, "-c", CLI_PROGRAM, *args],
        env=harness.env(),
        stdin=subprocess.DEVNULL,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def test_help_documents_stop_server(harness):
    result = run_cli(harness, "--help")
    assert result.returncode == 0
    assert "--stop-server" in result.stdout


def test_stop_server_stops_a_running_server(harness):
    process = harness.spawn_and_wait_ready()

    result = run_cli(harness, "--stop-server")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "Stopped the whspr server."
    assert not server.is_running()  # already free when the command returns
    assert not os.path.exists(server.SOCKET_PATH)
    wait_until(lambda: process.poll() is not None, message="server process to exit")
    assert process.returncode == 0


def test_stop_server_without_a_server_is_harmless_and_starts_none(harness):
    result = run_cli(harness, "--stop-server")

    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "The whspr server is not running."
    time.sleep(1.0)  # give any accidentally spawned server time to take the lock
    assert not server.is_running()


def test_stop_server_twice(harness):
    harness.spawn_and_wait_ready()
    assert "Stopped" in run_cli(harness, "--stop-server").stdout
    second = run_cli(harness, "--stop-server")
    assert second.returncode == 0
    assert "not running" in second.stdout


def test_stop_server_before_idle_timeout(harness):
    """The point of the command: stop long before the idle timeout would."""
    process = harness.spawn_and_wait_ready(idle_timeout=3600.0)
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content  # the server was just used

    started = time.monotonic()
    result = run_cli(harness, "--stop-server")
    assert result.returncode == 0, result.stderr
    wait_until(lambda: process.poll() is not None, message="server process to exit")
    assert time.monotonic() - started < 15.0


def test_next_dictation_after_stop_server_restarts_the_server(harness, monkeypatch):
    first = harness.spawn_and_wait_ready()
    assert run_cli(harness, "--stop-server").returncode == 0
    wait_until(lambda: first.poll() is not None, message="first server to exit")

    monkeypatch.setattr(server, "start", harness.make_fake_start())
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content
    assert server.is_running()
    server.stop()


def test_stop_server_mid_transcription_still_delivers_the_result(harness):
    process = harness.spawn_and_wait_ready(job_delay=3.0)
    path, content = harness.write_audio_stub()
    results = []
    in_flight = threading.Thread(target=lambda: results.append(server.transcribe(path)))
    in_flight.start()
    wait_until(
        lambda: any(e.startswith("job-start") for e in harness.events()),
        message="the job to start",
    )

    result = run_cli(harness, "--stop-server")
    assert result.returncode == 0, result.stderr
    assert "Stopped" in result.stdout

    in_flight.join(timeout=30.0)
    assert results == [content]  # the dictation underway was not lost
    wait_until(lambda: process.poll() is not None, message="server to exit after its job")


@pytest.mark.parametrize(
    "args",
    [
        ["--stop-server", "--paste"],
        ["--stop-server", "--cancel"],
        ["--cancel", "--stop-server"],
        ["--stop-server", "--finish-setup"],
    ],
)
def test_stop_server_refuses_to_be_combined(harness, args):
    process = harness.spawn_and_wait_ready()

    result = run_cli(harness, *args)

    assert result.returncode == 2  # argparse usage error
    assert "cannot be combined" in result.stderr
    assert process.poll() is None  # nothing was stopped (or started)


def test_stop_server_fails_cleanly_on_an_unresponsive_server(harness):
    """A server holding the lock without ever answering must produce an
    error and exit status 1 within the stop timeout, not a hang."""
    lock_file = hold_server_lock(os.getpid())
    try:
        started = time.monotonic()
        result = run_cli(harness, "--stop-server", timeout=60)
        assert result.returncode == 1
        assert "could not stop the server" in result.stderr
        assert "Traceback" not in result.stderr
        assert time.monotonic() - started < server._STOP_TIMEOUT + 10.0
    finally:
        lock_file.close()


def test_stop_server_reports_errors_without_a_traceback(monkeypatch, capsys):
    import whspr.__main__ as cli

    def refuse():
        raise PermissionError(13, "Permission denied")

    monkeypatch.setattr(cli.server, "stop", refuse)
    assert cli.stop_server() == 1
    captured = capsys.readouterr()
    assert "could not stop the server" in captured.err
    assert "Permission denied" in captured.err
    assert captured.out == ""
