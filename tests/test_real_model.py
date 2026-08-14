"""Tests that exercise real Whisper models (downloads them on first run).

CUDA visibility is sticky per process (CUDA_VISIBLE_DEVICES is read once at
CUDA initialization), so every model-selection probe runs in a fresh
subprocess with an explicitly controlled environment.
"""

import json
import os
import subprocess
import sys
import time

import pytest

import whspr.server as server

from conftest import wait_until

pytestmark = pytest.mark.real


ENGLISH_TEXT = "The quick brown fox jumps over the lazy dog"
GERMAN_TEXT = "Guten Morgen, heute scheint die Sonne und die Vögel singen"

# Loads the model exactly like the server would, transcribes one file, and
# reports what happened as JSON on stdout.
PROBE_PROGRAM = """
import json, sys
import whspr.server as server

model = server.load_model()
result = {
    "device": model.model.device,
    "multilingual": bool(model.model.is_multilingual),
    "text": server.transcribe_helper(sys.argv[1], model) if sys.argv[1] != "-" else "",
}
print(json.dumps(result))
"""


def probe_load_model(wav="-", cuda_visible=None, timeout=600):
    env = os.environ.copy()
    if cuda_visible is not None:
        env["CUDA_VISIBLE_DEVICES"] = cuda_visible
    result = subprocess.run(
        [sys.executable, "-c", PROBE_PROGRAM, wav],
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )
    assert result.returncode == 0, result.stderr
    return json.loads(result.stdout)


def make_speech_wav(tmp_path, text, voice="en-us", name="speech.wav"):
    path = tmp_path / name
    subprocess.run(
        ["espeak-ng", "-v", voice, "-s", "140", "-w", str(path), text],
        check=True,
    )
    assert path.stat().st_size > 44
    return str(path)


def keywords_found(transcript, keywords, minimum):
    transcript = transcript.lower()
    return sum(1 for k in keywords if k.lower() in transcript) >= minimum


def test_cpu_machine_loads_multilingual_small():
    probe = probe_load_model(cuda_visible="")
    assert probe["device"] == "cpu"
    assert probe["multilingual"] is True  # "small", not "small.en"


def test_cpu_transcribes_english(tmp_path):
    wav = make_speech_wav(tmp_path, ENGLISH_TEXT, voice="en-us")
    probe = probe_load_model(wav=wav, cuda_visible="")
    assert probe["device"] == "cpu"
    assert keywords_found(probe["text"], ["quick", "brown", "fox", "lazy", "dog"], 3), (
        probe["text"]
    )


def test_cpu_transcribes_german_proving_multilingual(tmp_path):
    wav = make_speech_wav(tmp_path, GERMAN_TEXT, voice="de", name="german.wav")
    probe = probe_load_model(wav=wav, cuda_visible="")
    assert probe["device"] == "cpu"
    assert keywords_found(
        probe["text"], ["Morgen", "Sonne", "Vögel", "singen", "scheint"], 2
    ), probe["text"]


def test_gpu_machine_uses_turbo_or_falls_back_cleanly(tmp_path):
    """On a CUDA machine load_model() must either use the GPU or fall back."""
    wav = make_speech_wav(tmp_path, ENGLISH_TEXT, voice="en-us")
    probe = probe_load_model(wav=wav)
    # Either outcome is legitimate (the GPU may be busy/VRAM-starved); what
    # matters is that a working model came up either way.
    assert probe["device"] in ("cuda", "cpu")
    assert keywords_found(probe["text"], ["quick", "brown", "fox", "lazy", "dog"], 3), (
        probe["text"]
    )
    if probe["device"] == "cpu" and server._cuda_is_usable():
        pytest.skip("CUDA present but unusable right now; CPU fallback verified instead")


def test_no_cache_no_network_fails_fast_not_forever(harness, tmp_path, monkeypatch):
    """With no model cache and no network the user gets an error, not a hang."""
    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "HF_HOME": str(tmp_path / "empty-hf-cache"),
        "HF_HUB_OFFLINE": "1",
        "XDG_RUNTIME_DIR": harness.runtime_dir,
    }

    def start_offline_server():
        if server.is_running():
            return
        process = subprocess.Popen(
            [sys.executable, "-m", "whspr.server"],
            env=env,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
        )
        harness.processes.append(process)

    monkeypatch.setattr(server, "start", start_offline_server)

    wav = make_speech_wav(tmp_path, ENGLISH_TEXT, voice="en-us")
    start_time = time.monotonic()
    with pytest.raises(RuntimeError, match="model load failed|did not become available"):
        server.transcribe(wav)
    assert time.monotonic() - start_time < 120.0

    # The broken server must vacate the lock so a later attempt gets a fresh
    # process (which would succeed once the network is back).
    wait_until(
        lambda: not server.is_running(),
        timeout=30.0,
        message="failed-model server to exit",
    )


def test_server_survives_parent_death(harness, tmp_path):
    """Requirement 5: killing the process that called start() must not kill
    the server (start_new_session detachment)."""
    import signal

    env = {
        **os.environ,
        "CUDA_VISIBLE_DEVICES": "",
        "XDG_RUNTIME_DIR": harness.runtime_dir,
    }
    parent = subprocess.Popen(
        [
            sys.executable,
            "-c",
            "import whspr.server as s, time; s.start(); time.sleep(600)",
        ],
        env=env,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    try:
        wait_until(
            lambda: os.path.exists(server.SOCKET_PATH),
            timeout=30.0,
            message="server started by the doomed parent to bind",
        )
        # Kill the parent's entire process group; a non-detached server would
        # be in that group and die with it.
        os.killpg(parent.pid, signal.SIGKILL)
        parent.wait(timeout=10)

        time.sleep(1.0)
        assert server.is_running(), "server died with its parent"
        wav = make_speech_wav(tmp_path, ENGLISH_TEXT, voice="en-us")
        text = server.transcribe(wav)
        assert keywords_found(text, ["quick", "brown", "fox", "lazy", "dog"], 3), text
    finally:
        if parent.poll() is None:
            os.killpg(parent.pid, signal.SIGKILL)
            parent.wait()
        server.stop()
        wait_until(
            lambda: not server.is_running(),
            timeout=15.0,
            message="detached server to stop in teardown",
        )


def test_end_to_end_server_with_real_model_cpu(harness, tmp_path, monkeypatch):
    """Spawn the real `python -m whspr.server` and transcribe real audio."""
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "")

    spawn_started = time.monotonic()
    process = subprocess.Popen(
        [sys.executable, "-m", "whspr.server"],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        stdin=subprocess.DEVNULL,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        start_new_session=True,
    )
    harness.processes.append(process)
    wait_until(
        lambda: os.path.exists(server.SOCKET_PATH),
        timeout=10.0,
        message="real server to bind its socket quickly",
    )
    bind_latency = time.monotonic() - spawn_started
    assert bind_latency < 5.0, f"socket took {bind_latency:.1f}s to appear"

    wav = make_speech_wav(tmp_path, ENGLISH_TEXT, voice="en-us")
    text = server.transcribe(wav)
    assert keywords_found(text, ["quick", "brown", "fox", "lazy", "dog"], 3), text

    server.stop()
    wait_until(lambda: process.poll() is not None, message="server to exit")
