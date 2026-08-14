"""Lifecycle, protocol, and robustness tests for whspr.server.

These use real server processes with a fake (instant) model, so they exercise
the genuine socket protocol, locking, threading, and shutdown logic quickly.
"""

import json
import os
import signal
import socket
import subprocess
import threading
import time

import pytest

import whspr.server as server

from conftest import raw_request, wait_until


def test_server_binds_socket_immediately_while_model_loads(harness):
    """The socket must be reachable long before a slow model finishes loading."""
    harness.spawn(load_delay=5.0)
    wait_until(
        lambda: os.path.exists(server.SOCKET_PATH) and server.is_running(),
        timeout=3.0,
        message="socket to appear well before the model is loaded",
    )
    assert "model-loaded" not in " ".join(harness.events())
    server.stop()


def test_transcribe_roundtrip(harness):
    harness.spawn_and_wait_ready()
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content


def test_transcribe_queued_before_model_ready(harness):
    """Requests sent while the model is still loading must be answered."""
    harness.spawn_and_wait_ready(load_delay=2.0)
    path, content = harness.write_audio_stub()
    start_time = time.monotonic()
    assert server.transcribe(path) == content
    assert time.monotonic() - start_time >= 1.0  # it really did wait for the model


def test_transcriptions_run_sequentially(harness):
    harness.spawn_and_wait_ready(job_delay=0.5)
    paths = [harness.write_audio_stub(f"audio-{i}.txt", f"text {i}") for i in range(3)]

    results = {}

    def run(path, content):
        results[path] = (server.transcribe(path), content)

    threads = [threading.Thread(target=run, args=p) for p in paths]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(results) == 3  # every client thread completed without raising
    for got, expected in results.values():
        assert got == expected

    # The event log must show strictly serialized jobs: no start before the
    # previous job's end.
    events = [e.split()[0] for e in harness.events() if e.startswith("job-")]
    assert events == ["job-start", "job-end"] * 3


def test_relative_paths_are_resolved_client_side(harness, monkeypatch):
    """The server has its own cwd, so clients must send absolute paths."""
    harness.spawn_and_wait_ready()
    path, content = harness.write_audio_stub()
    monkeypatch.chdir(os.path.dirname(path))
    assert server.transcribe(os.path.basename(path)) == content


def test_unicode_and_large_transcripts(harness):
    harness.spawn_and_wait_ready()
    text = ("Grüße aus München! こんにちは。 " * 5000).strip()  # ~500 KB of UTF-8
    path = harness.tmp_path / "big.txt"
    path.write_text(text, encoding="utf-8")
    assert server.transcribe(str(path)) == text


def test_stop_terminates_server_and_cleans_up(harness):
    process = harness.spawn_and_wait_ready()
    server.stop()
    wait_until(lambda: process.poll() is not None, message="server process to exit")
    assert process.returncode == 0
    assert not os.path.exists(server.SOCKET_PATH)
    assert not server.is_running()


def test_stop_without_server_is_a_noop(harness):
    assert not server.is_running()
    server.stop()  # must not raise


def test_second_server_instance_exits_immediately(harness):
    first = harness.spawn_and_wait_ready()
    second = harness.spawn()
    wait_until(lambda: second.poll() is not None, message="duplicate server to exit")
    assert first.poll() is None
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content
    server.stop()


def test_many_concurrent_server_starts_leave_exactly_one(harness):
    processes = [harness.spawn() for _ in range(8)]
    wait_until(
        lambda: sum(1 for p in processes if p.poll() is None) == 1,
        message="all but one duplicate server to exit",
    )
    assert server.is_running()
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content
    server.stop()
    wait_until(
        lambda: all(p.poll() is not None for p in processes),
        message="remaining server to exit after stop",
    )


def test_idle_timeout_stops_unused_server(harness):
    process = harness.spawn_and_wait_ready(idle_timeout=1.0)
    wait_until(
        lambda: process.poll() is not None,
        timeout=6.0,
        message="idle server to stop itself",
    )
    assert process.returncode == 0
    assert not server.is_running()
    assert not os.path.exists(server.SOCKET_PATH)


def test_idle_timeout_counts_from_last_use(harness):
    process = harness.spawn_and_wait_ready(idle_timeout=2.0)
    path, content = harness.write_audio_stub()
    time.sleep(1.2)
    assert server.transcribe(path) == content  # resets the idle clock
    time.sleep(1.2)
    assert process.poll() is None  # only ~1.2s since last use
    assert server.transcribe(path) == content
    wait_until(
        lambda: process.poll() is not None,
        timeout=8.0,
        message="server to stop after truly going idle",
    )


def test_idle_timeout_does_not_interrupt_slow_transcription(harness):
    process = harness.spawn_and_wait_ready(idle_timeout=1.0, job_delay=3.0)
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content
    wait_until(lambda: process.poll() is not None, message="server to stop afterwards")


def test_idle_timeout_does_not_drop_job_queued_during_slow_model_load(harness):
    harness.spawn_and_wait_ready(idle_timeout=1.0, load_delay=3.0)
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content


def test_transcribe_autostarts_server(harness, monkeypatch):
    monkeypatch.setattr(server, "start", harness.make_fake_start())
    assert not server.is_running()
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content
    assert server.is_running()
    server.stop()


def test_transcribe_recovers_after_server_is_killed(harness, monkeypatch):
    """A SIGKILLed server leaves a stale socket file; clients must recover."""
    process = harness.spawn_and_wait_ready()
    os.kill(process.pid, signal.SIGKILL)
    process.wait()
    assert os.path.exists(server.SOCKET_PATH)  # stale socket left behind
    assert not server.is_running()

    monkeypatch.setattr(server, "start", harness.make_fake_start())
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content
    server.stop()


def test_transcribe_recovers_when_server_stops_concurrently(harness, monkeypatch):
    """Simulates the idle-shutdown race: a request arriving mid-shutdown."""
    harness.spawn_and_wait_ready()
    monkeypatch.setattr(server, "start", harness.make_fake_start())
    path, content = harness.write_audio_stub()

    results = []
    thread = threading.Thread(
        target=lambda: results.append(server.transcribe(path))
    )
    server.stop()
    thread.start()
    thread.join(timeout=30.0)
    assert not thread.is_alive()
    assert results == [content]
    server.stop()


def test_transcribe_fails_cleanly_when_server_cannot_start(harness, monkeypatch):
    monkeypatch.setattr(server, "start", lambda: None)  # nothing ever starts
    monkeypatch.setattr(server, "_SERVER_WAIT_TIMEOUT", 1.5)
    path, _ = harness.write_audio_stub()
    start_time = time.monotonic()
    with pytest.raises(RuntimeError, match="did not become available"):
        server.transcribe(path)
    assert time.monotonic() - start_time < 10.0


def test_model_load_failure_returns_error_and_server_exits(harness):
    process = harness.spawn_and_wait_ready(load_ok=False)
    path, _ = harness.write_audio_stub()
    with pytest.raises(RuntimeError, match="model load failed"):
        server.transcribe(path)
    wait_until(
        lambda: process.poll() is not None,
        message="failed-model server to exit promptly",
    )
    assert not server.is_running()


def test_transcription_error_is_reported_not_fatal(harness):
    harness.spawn_and_wait_ready()
    with pytest.raises(RuntimeError):
        server.transcribe(str(harness.tmp_path / "does-not-exist.txt"))
    # The server must still be healthy afterwards.
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content
    server.stop()


def test_sigterm_shuts_down_cleanly(harness):
    process = harness.spawn_and_wait_ready()
    os.kill(process.pid, signal.SIGTERM)
    wait_until(lambda: process.poll() is not None, message="server to exit on SIGTERM")
    assert process.returncode == 0
    assert not os.path.exists(server.SOCKET_PATH)
    assert not server.is_running()


def test_malformed_requests_get_error_replies(harness):
    harness.spawn_and_wait_ready()

    reply = raw_request(b'{"type": "transcribe"}\n')
    assert json.loads(reply)["ok"] is False

    reply = raw_request(b'{"type": "transcribe", "path": 42}\n')
    assert json.loads(reply)["ok"] is False

    reply = raw_request(b'{"type": "no-such-thing"}\n')
    assert json.loads(reply)["ok"] is False

    reply = raw_request(b'"just a string"\n')
    assert json.loads(reply)["ok"] is False

    reply = raw_request(b"this is not json at all\n")
    assert json.loads(reply)["ok"] is False  # invalid JSON also gets a reply

    # After all that abuse the server still works.
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content
    server.stop()


def test_silent_and_rude_clients_do_not_wedge_the_server(harness):
    harness.spawn_and_wait_ready()

    # Client that connects and immediately disconnects.
    with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
        conn.connect(server.SOCKET_PATH)

    # Client that connects and sends nothing (left open in the background).
    lingering = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    lingering.connect(server.SOCKET_PATH)
    try:
        path, content = harness.write_audio_stub()
        assert server.transcribe(path) == content
    finally:
        lingering.close()
    server.stop()


def test_oversized_message_is_rejected(harness, monkeypatch):
    harness.spawn_and_wait_ready()
    blob = b"A" * (server._MAX_MESSAGE_BYTES + 65536 * 2)
    reply = raw_request(blob, timeout=20.0)
    # The server rejects and closes; depending on timing the client sees the
    # error reply or just the reset — never a crash or a hang.
    assert reply == b"" or json.loads(reply)["ok"] is False
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content
    server.stop()


def test_start_reaps_exited_server_child(harness, monkeypatch):
    """start() must not leave zombie children inside long-lived callers."""
    # Hold the lock ourselves so the spawned real server exits immediately
    # without loading any model.
    lock_file = open(server.LOCK_PATH, "a+")
    import fcntl

    fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)

    spawned = []
    real_popen = subprocess.Popen

    def recording_popen(*args, **kwargs):
        process = real_popen(*args, **kwargs)
        spawned.append(process)
        return process

    try:
        monkeypatch.setattr(server, "is_running", lambda: False)
        monkeypatch.setattr(server.subprocess, "Popen", recording_popen)
        server.start()

        assert len(spawned) == 1
        pid = spawned[0].pid
        # /proc/<pid> disappears only once the child has exited AND been
        # reaped; an unreaped child would linger there in state Z forever.
        wait_until(
            lambda: not os.path.exists(f"/proc/{pid}"),
            timeout=15.0,
            message="child to exit and be reaped",
        )
    finally:
        lock_file.close()


def test_client_times_out_on_server_that_never_replies(harness, monkeypatch):
    """Requirement: no hangs, ever — even a wedged server must not block clients."""
    mute = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    mute.bind(server.SOCKET_PATH)  # accepts connections, never replies
    mute.listen(8)
    try:
        monkeypatch.setattr(server, "_RESULT_TIMEOUT", 1.0)
        monkeypatch.setattr(server, "start", lambda: None)
        path, _ = harness.write_audio_stub()
        start_time = time.monotonic()
        with pytest.raises(RuntimeError, match="timed out waiting"):
            server.transcribe(path)
        assert time.monotonic() - start_time < 10.0
    finally:
        mute.close()


def test_stop_times_out_on_server_that_never_replies(harness, monkeypatch):
    mute = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    mute.bind(server.SOCKET_PATH)
    mute.listen(8)
    try:
        monkeypatch.setattr(server, "_STOP_TIMEOUT", 1.0)
        with pytest.raises(RuntimeError, match="timed out waiting"):
            server.stop()
    finally:
        mute.close()


def test_successor_binds_while_predecessor_drains_and_keeps_its_socket(harness):
    """A stopping server must hand over lock and socket path BEFORE its last
    job finishes draining — and must never unlink the successor's socket when
    it finally exits."""
    predecessor = harness.spawn_and_wait_ready(job_delay=4.0)
    path, content = harness.write_audio_stub()

    results = []
    in_flight = threading.Thread(target=lambda: results.append(server.transcribe(path)))
    in_flight.start()
    wait_until(
        lambda: any(e.startswith("job-start") for e in harness.events()),
        message="the long job to start",
    )

    server.stop()
    # The lock and socket path must be free while the job still drains.
    wait_until(
        lambda: not server.is_running(),
        timeout=5.0,
        message="predecessor to release the lock during its drain",
    )
    assert predecessor.poll() is None  # still draining its in-flight job

    harness.spawn_and_wait_ready()
    path2, content2 = harness.write_audio_stub("audio-successor.txt", "successor text")
    assert server.transcribe(path2) == content2

    # When the predecessor finally exits it must not remove the successor's
    # socket.
    wait_until(
        lambda: predecessor.poll() is not None,
        timeout=30.0,
        message="predecessor to finish draining and exit",
    )
    assert os.path.exists(server.SOCKET_PATH)
    assert server.transcribe(path2) == content2

    in_flight.join(timeout=30.0)
    assert not in_flight.is_alive()
    assert results == [content]  # the drained job still delivered its result
    server.stop()


def test_transcribe_gets_fresh_budget_when_server_dies_mid_request(harness, monkeypatch):
    """A server that takes the request but dies after the availability window
    must not fail the client instantly: the replacement gets a fresh budget."""
    monkeypatch.setattr(server, "_SERVER_WAIT_TIMEOUT", 1.5)

    wedge = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    wedge.bind(server.SOCKET_PATH)
    wedge.listen(1)

    def take_request_then_die():
        conn, _ = wedge.accept()
        server._recv_message(conn)
        time.sleep(3.0)  # longer than the whole 1.5s availability budget
        conn.close()
        wedge.close()
        os.unlink(server.SOCKET_PATH)

    thread = threading.Thread(target=take_request_then_die, daemon=True)
    thread.start()

    monkeypatch.setattr(server, "start", harness.make_fake_start())
    path, content = harness.write_audio_stub()
    assert server.transcribe(path) == content  # healed via the fresh budget
    thread.join(timeout=10)
    server.stop()


def test_transcribe_recovery_budget_is_bounded(harness, monkeypatch):
    """A server that keeps accepting requests and dying must not retry forever."""
    monkeypatch.setattr(server, "_SERVER_WAIT_TIMEOUT", 1.0)
    monkeypatch.setattr(server, "start", lambda: None)

    crasher = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    crasher.bind(server.SOCKET_PATH)
    crasher.listen(16)
    stop_serving = threading.Event()

    def accept_and_hang_up():
        while not stop_serving.is_set():
            try:
                crasher.settimeout(0.5)
                conn, _ = crasher.accept()
            except (socket.timeout, OSError):
                continue
            try:
                server._recv_message(conn)
            except (ValueError, OSError):
                pass
            conn.close()  # EOF before any reply, over and over

    thread = threading.Thread(target=accept_and_hang_up, daemon=True)
    thread.start()
    try:
        path, _ = harness.write_audio_stub()
        start_time = time.monotonic()
        with pytest.raises(RuntimeError, match="did not become available"):
            server.transcribe(path)
        elapsed = time.monotonic() - start_time
        # Bounded by (1 + _MAX_RECOVERIES) budget windows plus overhead.
        assert elapsed < (1 + server._MAX_RECOVERIES) * 1.0 + 10.0
    finally:
        stop_serving.set()
        thread.join(timeout=10)
        crasher.close()


def test_start_stops_legacy_fixed_path_server(harness, monkeypatch):
    """start() must shut down a lingering pre-1.1 server on the old /tmp path."""
    legacy_path = server._LEGACY_SOCKET_PATH
    try:
        os.unlink(legacy_path)
    except OSError:
        pass
    legacy = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    legacy.bind(legacy_path)
    legacy.listen(1)
    received = []

    def serve_stop_once():
        conn, _ = legacy.accept()
        with conn:
            received.append(server._recv_message(conn))
            server._send_message(conn, {"ok": True})

    thread = threading.Thread(target=serve_stop_once, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(server, "is_running", lambda: True)  # skip spawning
        server.start()
        thread.join(timeout=10.0)
        assert not thread.is_alive()
        assert received == [{"type": "stop"}]
        assert not os.path.exists(legacy_path)
    finally:
        legacy.close()
        try:
            os.unlink(legacy_path)
        except OSError:
            pass


def test_wedged_job_forces_exit_and_next_dictation_works(harness):
    """A job stuck beyond any client's patience must make the server abandon
    itself (freeing the lock) so the next dictation gets a working process."""
    process = harness.spawn_and_wait_ready(job_delay=120.0, wedge_timeout=1.5)
    path, content = harness.write_audio_stub()

    wedged_conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    wedged_conn.connect(server.SOCKET_PATH)
    try:
        server._send_message(wedged_conn, {"type": "transcribe", "path": path})
        wait_until(
            lambda: process.poll() is not None,
            timeout=15.0,
            message="wedged server to abandon itself",
        )
        assert process.returncode == 1
        assert not server.is_running()  # the flock died with the process
    finally:
        wedged_conn.close()

    # The stale socket file must not stop a healthy successor.
    assert os.path.exists(server.SOCKET_PATH)
    harness.spawn_and_wait_ready()
    assert server.transcribe(path) == content
    server.stop()


def test_is_running_reflects_lock_state(harness):
    assert not server.is_running()
    process = harness.spawn_and_wait_ready()
    assert server.is_running()
    server.stop()
    wait_until(lambda: process.poll() is not None, message="server to exit")
    assert not server.is_running()
