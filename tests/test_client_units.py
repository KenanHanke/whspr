"""In-process unit tests for whspr.client internals."""

import os
import socket
import threading
import time

import pytest

import whspr.client as client


@pytest.fixture
def client_runtime(tmp_path, monkeypatch):
    runtime_dir = tmp_path / "client-runtime"
    runtime_dir.mkdir()
    monkeypatch.setenv("XDG_RUNTIME_DIR", str(runtime_dir))
    monkeypatch.setattr(
        client, "SOCKET_PATH", os.path.join(str(runtime_dir), "whspr-recorder.sock")
    )
    monkeypatch.setattr(
        client, "LOCK_PATH", os.path.join(str(runtime_dir), "whspr-recorder.lock")
    )
    return runtime_dir


def test_stop_reply_wait_is_bounded_when_recorder_never_replies(
    client_runtime, monkeypatch
):
    """A recorder that accepts the stop request but never finalizes must not
    hang the stop keypress forever."""
    mute_recorder = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    mute_recorder.bind(client.SOCKET_PATH)
    mute_recorder.listen(1)

    accepted = []

    def accept_and_ignore():
        conn, _ = mute_recorder.accept()
        accepted.append(conn)  # keep it open, never reply

    thread = threading.Thread(target=accept_and_ignore, daemon=True)
    thread.start()
    try:
        monkeypatch.setattr(client, "_STOP_REPLY_TIMEOUT", 0.5)
        start_time = time.monotonic()
        with pytest.raises(TimeoutError, match="did not finalize"):
            client.request_stop_and_wait(connect_timeout=5.0)
        assert time.monotonic() - start_time < 5.0
    finally:
        for conn in accepted:
            conn.close()
        mute_recorder.close()


def touch(path, age_seconds=0.0):
    path.write_bytes(b"RIFF" + b"\x00" * 100)
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(path, (old, old))


def find_dead_pid():
    pid = 4_000_000
    while os.path.exists(f"/proc/{pid}"):
        pid -= 1
    return pid


def test_sweep_deletes_only_old_recordings_of_dead_processes(client_runtime):
    dead_pid = find_dead_pid()
    live_pid = 1  # init always exists

    own = client_runtime / f"whspr-recording-{os.getpid()}.wav"
    live = client_runtime / f"whspr-recording-{live_pid}.wav"
    dead_old = client_runtime / f"whspr-recording-{dead_pid}.wav"
    dead_fresh = client_runtime / f"whspr-recording-{dead_pid - 1}.wav"
    unrelated = client_runtime / "not-a-recording.wav"

    touch(own, age_seconds=3 * 3600)
    touch(live, age_seconds=3 * 3600)
    touch(dead_old, age_seconds=3 * 3600)
    touch(dead_fresh)  # dead pid but recent: may still be in flight
    touch(unrelated, age_seconds=3 * 3600)

    client._sweep_stale_recordings()

    assert own.exists()  # never our own file
    assert live.exists()  # recorder still alive
    assert not dead_old.exists()  # genuinely stale: swept
    assert dead_fresh.exists()  # could still be queued for transcription
    assert unrelated.exists()  # unknown names are left alone


def test_new_recording_paths_land_in_the_runtime_dir(client_runtime):
    path = client._new_recording_path()
    assert path.startswith(str(client_runtime))
    assert str(os.getpid()) in os.path.basename(path)
