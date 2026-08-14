"""Full dictation-flow tests: `python -m whspr` twice, with shimmed audio I/O.

`arecord`/`aplay` are replaced by shims on PATH (no sound hardware needed) and
the clipboard runs against a headless Xvfb display via xclip, so this covers
__main__, client.py, server.py, and a real Whisper model end to end.
"""

import os
import shutil
import subprocess
import sys
import time

import pytest

import whspr.server as server

from conftest import wait_until
from test_real_model import ENGLISH_TEXT, keywords_found, make_speech_wav

pytestmark = pytest.mark.real

DISPLAY = ":99"


ARECORD_SHIM = """#!/bin/bash
# Fake arecord: instantly "record" a prepared WAV file, then wait for SIGINT
# like the real arecord would while recording.
for last in "$@"; do :; done
cp "$WHSPR_TEST_RECORDING" "$last"
trap 'exit 0' INT TERM
while true; do sleep 0.05; done
"""

APLAY_SHIM = """#!/bin/bash
exit 0
"""


@pytest.fixture(scope="module")
def xvfb():
    if shutil.which("Xvfb") is None or shutil.which("xclip") is None:
        pytest.skip("Xvfb/xclip not available")
    proc = subprocess.Popen(
        ["Xvfb", DISPLAY, "-screen", "0", "640x480x16"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until(
            lambda: os.path.exists("/tmp/.X11-unix/X" + DISPLAY.lstrip(":")),
            timeout=10.0,
            message="Xvfb display to appear",
        )
        yield DISPLAY
    finally:
        proc.terminate()
        proc.wait()


@pytest.fixture
def dictation_env(harness, tmp_path, xvfb):
    shim_dir = tmp_path / "bin"
    shim_dir.mkdir()
    (shim_dir / "arecord").write_text(ARECORD_SHIM)
    (shim_dir / "aplay").write_text(APLAY_SHIM)
    os.chmod(shim_dir / "arecord", 0o755)
    os.chmod(shim_dir / "aplay", 0o755)

    recording = make_speech_wav(tmp_path, ENGLISH_TEXT, voice="en-us")

    env = os.environ.copy()
    env.update(
        {
            "PATH": f"{shim_dir}:{env['PATH']}",
            "DISPLAY": xvfb,
            "XDG_RUNTIME_DIR": harness.runtime_dir,
            "CUDA_VISIBLE_DEVICES": "",
            "WHSPR_TEST_RECORDING": recording,
        }
    )

    yield env

    # A stray recording would wedge later tests; cancel it only if one exists
    # (a bare --cancel would itself spawn a fresh server via __main__).
    if os.path.exists(recorder_socket_path(env)):
        subprocess.run(
            [sys.executable, "-m", "whspr", "--cancel"],
            env=env,
            timeout=60,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    # Stop the transcription server, retrying briefly in case one is still
    # mid-startup (spawned detached, it may not have bound its socket yet).
    deadline = time.monotonic() + 15.0
    while time.monotonic() < deadline:
        try:
            server.stop()
        except RuntimeError:
            pass
        if not server.is_running():
            break
        time.sleep(0.3)
    assert not server.is_running(), "a test server survived teardown"


def recorder_socket_path(env):
    return os.path.join(env["XDG_RUNTIME_DIR"], "whspr-recorder.sock")


def leftover_recordings(env):
    import glob

    return glob.glob(os.path.join(env["XDG_RUNTIME_DIR"], "whspr-recording-*.wav"))


def run_whspr(env, args, tmp_path, name, timeout):
    """Run `python -m whspr` with output redirected to a file, not pipes.

    pyperclip's xclip child daemonizes while holding inherited stdout/stderr
    open, so capturing via pipes would wait for EOF forever.
    """
    log_path = tmp_path / f"{name}.log"
    with open(log_path, "w") as log:
        result = subprocess.run(
            [sys.executable, "-m", "whspr", *args],
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
            timeout=timeout,
        )
    return result.returncode, log_path.read_text()


def read_clipboard(env):
    result = subprocess.run(
        ["xclip", "-selection", "clipboard", "-o"],
        env=env,
        capture_output=True,
        text=True,
        timeout=10,
    )
    return result.stdout


def set_clipboard(env, text):
    subprocess.run(
        ["xclip", "-selection", "clipboard"],
        env=env,
        input=text,
        text=True,
        timeout=10,
        check=True,
    )


def start_recorder(dictation_env, tmp_path):
    log = open(tmp_path / "recorder.log", "w")
    recorder = subprocess.Popen(
        [sys.executable, "-m", "whspr"],
        env=dictation_env,
        stdout=log,
        stderr=subprocess.STDOUT,
    )
    log.close()
    wait_until(
        lambda: os.path.exists(recorder_socket_path(dictation_env)),
        timeout=30.0,
        message="recorder to start listening",
    )
    return recorder


def recorder_log(tmp_path):
    try:
        return (tmp_path / "recorder.log").read_text()
    except OSError:
        return "<no recorder log>"


def test_full_dictation_flow_copies_transcript_to_clipboard(dictation_env, tmp_path):
    set_clipboard(dictation_env, "sentinel-before-dictation")

    recorder = start_recorder(dictation_env, tmp_path)
    try:
        returncode, log = run_whspr(dictation_env, [], tmp_path, "stopper", 600)
        assert returncode == 0, log

        recorder.wait(timeout=30)
        assert recorder.returncode == 0, recorder_log(tmp_path)

        transcript = read_clipboard(dictation_env)
        assert keywords_found(transcript, ["quick", "brown", "fox", "lazy", "dog"], 3), (
            f"clipboard: {transcript!r}"
        )
        # The recording is deleted once the transcript is safely delivered.
        assert leftover_recordings(dictation_env) == []
    finally:
        if recorder.poll() is None:
            recorder.kill()
            recorder.wait()


def test_cancel_flow_discards_recording(dictation_env, tmp_path):
    set_clipboard(dictation_env, "sentinel-cancel-test")

    recorder = start_recorder(dictation_env, tmp_path)
    try:
        returncode, log = run_whspr(dictation_env, ["--cancel"], tmp_path, "cancel", 120)
        assert returncode == 0, log

        recorder.wait(timeout=30)
        assert recorder.returncode == 0, recorder_log(tmp_path)

        assert read_clipboard(dictation_env) == "sentinel-cancel-test"
        assert leftover_recordings(dictation_env) == []  # cancelled WAV deleted
    finally:
        if recorder.poll() is None:
            recorder.kill()
            recorder.wait()


def test_cancel_without_recording_is_quick_and_starts_no_server(dictation_env, tmp_path):
    assert not server.is_running()
    returncode, log = run_whspr(dictation_env, ["--cancel"], tmp_path, "noop-cancel", 60)
    assert returncode == 0, log
    time.sleep(1.0)  # give any accidentally spawned server time to take the lock
    assert not server.is_running()


def test_paste_flag_without_ydotool_still_succeeds(dictation_env, tmp_path):
    """--paste must degrade gracefully when ydotool is not installed."""
    set_clipboard(dictation_env, "sentinel-paste-test")

    recorder = start_recorder(dictation_env, tmp_path)
    try:
        returncode, log = run_whspr(dictation_env, ["--paste"], tmp_path, "paster", 600)
        assert returncode == 0, log

        recorder.wait(timeout=30)
        assert recorder.returncode == 0, recorder_log(tmp_path)

        transcript = read_clipboard(dictation_env)
        assert keywords_found(transcript, ["quick", "brown", "fox", "lazy", "dog"], 3), (
            f"clipboard: {transcript!r}"
        )
    finally:
        if recorder.poll() is None:
            recorder.kill()
            recorder.wait()


DYING_ARECORD_SHIM = """#!/bin/bash
# Fake arecord that captures a moment of audio and then crashes, like a mic
# being unplugged mid-dictation.
for last in "$@"; do :; done
cp "$WHSPR_TEST_RECORDING" "$last"
sleep 0.3
echo "arecord: pcm_read error: Input/output error" >&2
exit 1
"""


def test_dead_microphone_reports_error_instead_of_silence(dictation_env, tmp_path):
    """If arecord dies mid-recording the stop press must surface an error,
    never a silently truncated transcript."""
    shim_dir = tmp_path / "dying-bin"
    shim_dir.mkdir()
    (shim_dir / "arecord").write_text(DYING_ARECORD_SHIM)
    (shim_dir / "aplay").write_text(APLAY_SHIM)
    os.chmod(shim_dir / "arecord", 0o755)
    os.chmod(shim_dir / "aplay", 0o755)
    env = dict(dictation_env)
    env["PATH"] = f"{shim_dir}:{env['PATH']}"

    set_clipboard(env, "sentinel-dead-mic")

    recorder = subprocess.Popen(
        [sys.executable, "-m", "whspr"],
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    try:
        wait_until(
            lambda: os.path.exists(recorder_socket_path(env)),
            timeout=30.0,
            message="recorder to start listening",
        )
        time.sleep(1.0)  # let the fake arecord die

        returncode, log = run_whspr(env, [], tmp_path, "dead-mic-stopper", 120)
        assert returncode != 0
        assert "arecord failed" in log, log

        recorder.wait(timeout=30)
        assert recorder.returncode != 0  # the recorder also reports failure

        assert read_clipboard(env) == "sentinel-dead-mic"  # nothing fake copied
        assert leftover_recordings(env) == []  # partial capture cleaned up
    finally:
        if recorder.poll() is None:
            recorder.kill()
            recorder.wait()


FAILING_APLAY_SHIM = """#!/bin/bash
# Simulates a busy/broken audio OUTPUT device: every playback fails, while
# the microphone (arecord) is unaffected.
echo "aplay: audio open error: Device or resource busy" >&2
exit 1
"""


def test_broken_audio_output_still_dictates(dictation_env, tmp_path):
    """A busy/broken speaker must not make the tool unusable: the whole
    dictation (record -> transcribe -> clipboard) must still succeed with no
    beeps, rather than crash-looping on the start sound."""
    shim_dir = tmp_path / "silent-bin"
    shim_dir.mkdir()
    (shim_dir / "arecord").write_text(ARECORD_SHIM)
    (shim_dir / "aplay").write_text(FAILING_APLAY_SHIM)
    os.chmod(shim_dir / "arecord", 0o755)
    os.chmod(shim_dir / "aplay", 0o755)
    env = dict(dictation_env)
    env["PATH"] = f"{shim_dir}:{env['PATH']}"

    set_clipboard(env, "sentinel-broken-audio")

    log = open(tmp_path / "recorder.log", "w")
    recorder = subprocess.Popen(
        [sys.executable, "-m", "whspr"], env=env, stdout=log, stderr=subprocess.STDOUT
    )
    log.close()
    try:
        # The recorder must come up despite the start sound failing.
        wait_until(
            lambda: os.path.exists(recorder_socket_path(env)),
            timeout=30.0,
            message="recorder to start listening without a working speaker",
        )

        returncode, out = run_whspr(env, [], tmp_path, "broken-audio-stopper", 600)
        assert returncode == 0, out  # success despite every aplay failing

        recorder.wait(timeout=30)
        assert recorder.returncode == 0, recorder_log(tmp_path)

        transcript = read_clipboard(env)
        assert keywords_found(transcript, ["quick", "brown", "fox", "lazy", "dog"], 3), (
            f"clipboard: {transcript!r}"
        )
        assert leftover_recordings(env) == []
    finally:
        if recorder.poll() is None:
            recorder.kill()
            recorder.wait()


def test_finish_setup_preloads_model(dictation_env, tmp_path):
    returncode, log = run_whspr(
        dictation_env, ["--finish-setup"], tmp_path, "finish-setup", 600
    )
    assert returncode == 0, log
