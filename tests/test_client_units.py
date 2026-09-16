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


class _FakeSoundProc:
    """Stand-in for an aplay Popen, with a controllable exit status."""

    def __init__(self, returncode=0, stderr_text=""):
        self.returncode = returncode
        self._stderr_text = stderr_text
        self.killed = False

    @property
    def stderr(self):
        return _FakeStderr(self._stderr_text) if self._stderr_text else None

    def wait(self, timeout=None):
        return self.returncode

    def kill(self):
        self.killed = True


class _FakeStderr:
    def __init__(self, text):
        self._text = text

    def read(self):
        return self._text


@pytest.fixture
def stubbed_sounds(monkeypatch):
    """Record every sound played, without touching the audio stack."""
    played = []
    monkeypatch.setattr(client, "play_wav_blocking", lambda path: played.append(path))
    monkeypatch.setattr(client, "play_wav_background", lambda path: _FakeSoundProc())
    return played


def test_stop_side_failure_plays_cancelled_sound_and_reraises(
    tmp_path, monkeypatch, stubbed_sounds
):
    """The interface is sound-based: a failed dictation must be audible, and
    the recording preserved (regression: the laptop 'model load failed' bug
    left the user with silence and a discarded traceback)."""
    recording = tmp_path / "whspr-recording-x.wav"
    recording.write_bytes(b"RIFF" + b"\x00" * 100)
    monkeypatch.setattr(client, "request_stop_and_wait", lambda *a, **k: str(recording))
    monkeypatch.setattr(
        client.server,
        "transcribe",
        lambda p: (_ for _ in ()).throw(
            RuntimeError("model load failed: No module named 'nvidia'")
        ),
    )
    copied = []
    monkeypatch.setattr(client.pyperclip, "copy", copied.append)

    with pytest.raises(RuntimeError, match="model load failed"):
        client.stop_transcribe_copy_and_notify()

    assert client.CANCELLED_SOUND in stubbed_sounds
    assert client.FINISHED_SOUND not in stubbed_sounds
    assert copied == []
    assert recording.exists()  # the dictation is preserved, not discarded


def test_clipboard_backend_failure_is_audible_and_reraised(
    tmp_path, monkeypatch, stubbed_sounds
):
    """No pyperclip backend (no xclip/wl-clipboard) must not fail silently."""
    recording = tmp_path / "whspr-recording-y.wav"
    recording.write_bytes(b"RIFF" + b"\x00" * 100)
    monkeypatch.setattr(client, "request_stop_and_wait", lambda *a, **k: str(recording))
    monkeypatch.setattr(client.server, "transcribe", lambda p: "hello")

    def no_backend(text):
        raise client.pyperclip.PyperclipException("no clipboard backend")

    monkeypatch.setattr(client.pyperclip, "copy", no_backend)

    with pytest.raises(client.pyperclip.PyperclipException):
        client.stop_transcribe_copy_and_notify()
    assert client.CANCELLED_SOUND in stubbed_sounds


def test_stop_sound_failure_does_not_abort_successful_dictation(
    tmp_path, monkeypatch, stubbed_sounds
):
    """Once the transcript is on the clipboard, a broken notification sound
    must not present the successful dictation as a failure."""
    recording = tmp_path / "whspr-recording-z.wav"
    recording.write_bytes(b"RIFF" + b"\x00" * 100)
    monkeypatch.setattr(client, "request_stop_and_wait", lambda *a, **k: str(recording))
    monkeypatch.setattr(
        client,
        "play_wav_background",
        lambda path: _FakeSoundProc(returncode=1, stderr_text="Device or resource busy"),
    )
    monkeypatch.setattr(client.server, "transcribe", lambda p: "hello world")
    copied = []
    monkeypatch.setattr(client.pyperclip, "copy", copied.append)
    pasted = []
    monkeypatch.setattr(client, "_paste_with_ydotool", lambda: pasted.append(True))

    client.stop_transcribe_copy_and_notify(paste=True)  # must not raise

    assert copied == ["hello world"]
    assert client.FINISHED_SOUND in stubbed_sounds  # success still signalled
    assert pasted == [True]
    assert not recording.exists()  # deleted on success


def test_broken_audio_device_never_fails_a_successful_dictation(tmp_path, monkeypatch):
    """A busy/wedged audio device breaks EVERY aplay identically. Once the
    transcript is on the clipboard the dictation has succeeded, so no sound
    failure — including the final finished sound — may turn it into an error."""
    recording = tmp_path / "whspr-recording-b.wav"
    recording.write_bytes(b"RIFF" + b"\x00" * 100)
    monkeypatch.setattr(client, "request_stop_and_wait", lambda *a, **k: str(recording))
    monkeypatch.setattr(
        client,
        "play_wav_background",
        lambda path: _FakeSoundProc(returncode=1, stderr_text="Device or resource busy"),
    )

    def broken_device(path):
        raise RuntimeError("aplay failed with exit code 1: Device or resource busy")

    monkeypatch.setattr(client, "play_wav_blocking", broken_device)
    monkeypatch.setattr(client.server, "transcribe", lambda p: "hello world")
    copied = []
    monkeypatch.setattr(client.pyperclip, "copy", copied.append)
    pasted = []
    monkeypatch.setattr(client, "_paste_with_ydotool", lambda: pasted.append(True))

    client.stop_transcribe_copy_and_notify(paste=True)  # must NOT raise

    assert copied == ["hello world"]  # the transcript still reached the clipboard
    assert pasted == [True]
    assert not recording.exists()  # and the successful recording was cleaned up


def test_missing_aplay_binary_does_not_prevent_transcription(tmp_path, monkeypatch):
    """If aplay is missing entirely, launching the stop sound raises before
    transcription. That must not lose the dictation — the transcript must
    still be produced and copied (regression: the STOP-sound launch gated it)."""
    recording = tmp_path / "whspr-recording-a.wav"
    recording.write_bytes(b"RIFF" + b"\x00" * 100)
    monkeypatch.setattr(client, "request_stop_and_wait", lambda *a, **k: str(recording))

    def no_aplay(path):
        raise FileNotFoundError(2, "No such file or directory", "aplay")

    monkeypatch.setattr(client, "play_wav_background", no_aplay)
    monkeypatch.setattr(client.server, "transcribe", lambda p: "hello world")
    copied = []
    monkeypatch.setattr(client.pyperclip, "copy", copied.append)

    client.stop_transcribe_copy_and_notify()  # must NOT raise

    assert copied == ["hello world"]  # transcription happened despite no aplay
    assert not recording.exists()


def test_paste_missing_ydotool_reports_to_stderr(monkeypatch, capsys):
    def missing(*args, **kwargs):
        raise FileNotFoundError(2, "No such file or directory", "ydotool")

    monkeypatch.setattr(client.subprocess, "Popen", missing)
    client._paste_with_ydotool()  # must not raise
    assert "ydotool" in capsys.readouterr().err


def test_paste_nonzero_exit_warns_about_ydotoold(monkeypatch, capsys):
    class ExitsNonzero:
        def __init__(self, *args, **kwargs):
            pass

        def wait(self):
            return 1

    monkeypatch.setattr(client.subprocess, "Popen", ExitsNonzero)
    client._paste_with_ydotool()
    # The reaper thread prints asynchronously; give it a moment.
    for _ in range(50):
        if "ydotoold" in capsys.readouterr().err:
            return
        time.sleep(0.02)
    pytest.fail("expected a ydotoold warning on nonzero ydotool exit")


# --- recorder process hardening ------------------------------------------------

PID_REPORTING_ARECORD = """#!/bin/bash
# Fake arecord: report our pid, "record" a little, then run until signalled.
for last in "$@"; do :; done
echo $$ > "$ARECORD_PID_FILE"
head -c 1000 /dev/zero > "$last"
trap 'exit 0' INT TERM
while true; do sleep 0.05; done
"""

SILENT_APLAY = "#!/bin/bash\nexit 0\n"


def start_shimmed_recorder(tmp_path, runtime_dir):
    import subprocess
    import sys

    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    for name, script in (("arecord", PID_REPORTING_ARECORD), ("aplay", SILENT_APLAY)):
        (bin_dir / name).write_text(script)
        (bin_dir / name).chmod(0o755)
    pid_file = tmp_path / "arecord.pid"
    env = {
        **os.environ,
        "PATH": f"{bin_dir}:{os.environ['PATH']}",
        "XDG_RUNTIME_DIR": str(runtime_dir),
        "ARECORD_PID_FILE": str(pid_file),
    }
    # client.main() directly: the recorder side, without spawning a server.
    recorder = subprocess.Popen(
        [sys.executable, "-c", "import whspr.client as c; c.main()"],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.monotonic() + 20.0
    while not (pid_file.exists() and pid_file.read_text().strip()):
        assert time.monotonic() < deadline, "fake arecord never started"
        time.sleep(0.05)
    return recorder, int(pid_file.read_text())


def process_is_gone(pid):
    try:
        with open(f"/proc/{pid}/stat") as stat:
            return stat.read().split(") ")[1].startswith("Z")  # zombie: dead
    except FileNotFoundError:
        return True


def test_recording_and_recorder_socket_are_private(tmp_path, client_runtime):
    recorder, arecord_pid = start_shimmed_recorder(tmp_path, client_runtime)
    try:
        socket_path = client_runtime / "whspr-recorder.sock"
        recording = client_runtime / f"whspr-recording-{recorder.pid}.wav"
        deadline = time.monotonic() + 10.0
        while not socket_path.exists():
            assert time.monotonic() < deadline
            time.sleep(0.05)
        assert os.stat(socket_path).st_mode & 0o777 == 0o600
        assert os.stat(recording).st_mode & 0o777 == 0o600
    finally:
        recorder.kill()
        recorder.wait()
        if not process_is_gone(arecord_pid):
            os.kill(arecord_pid, 9)


@pytest.mark.parametrize("signal_name", ["SIGTERM", "SIGKILL"])
def test_killed_recorder_does_not_orphan_arecord(tmp_path, client_runtime, signal_name):
    """arecord must not keep the microphone open after its recorder dies."""
    import signal

    recorder, arecord_pid = start_shimmed_recorder(tmp_path, client_runtime)
    try:
        recorder.send_signal(getattr(signal, signal_name))
        recorder.wait(timeout=10)
        deadline = time.monotonic() + 10.0
        while not process_is_gone(arecord_pid):
            assert time.monotonic() < deadline, "arecord outlived its recorder"
            time.sleep(0.05)
    finally:
        if recorder.poll() is None:
            recorder.kill()
            recorder.wait()
        if not process_is_gone(arecord_pid):
            os.kill(arecord_pid, 9)
