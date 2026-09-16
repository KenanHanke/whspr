"""Tests that exercise real Whisper models (downloads them on first run).

CUDA visibility is sticky per process (CUDA_VISIBLE_DEVICES is read once at
CUDA initialization), so every model-selection probe runs in a fresh
subprocess with an explicitly controlled environment.
"""

import json
import os
import shutil
import subprocess
import sys
import time

import pytest

import whspr.server as server

from conftest import wait_until

pytestmark = pytest.mark.real


ENGLISH_TEXT = "The quick brown fox jumps over the lazy dog"
GERMAN_TEXT = "Guten Morgen, heute scheint die Sonne und die Vögel singen"
FRENCH_TEXT = "Bonjour, je voudrais réserver une table pour deux personnes ce soir"

LONG_OPENING = "At the very beginning, a purple elephant walked into the room."
LONG_FILLER = " ".join(
    [
        "Please send the quarterly report to the finance department by Friday.",
        "The weather tomorrow will be sunny with a light breeze from the west.",
        "Remember to water the plants and feed the cat before you leave.",
        "Our meeting has been moved to three o'clock in the large conference room.",
    ]
)
LONG_CLOSING = "At the very end, a green giraffe waved goodbye to everyone."

# Loads the model exactly like the server would, transcribes one file, and
# reports what happened as JSON on stdout.
PROBE_PROGRAM = """
import json, resource, sys
import whspr.server as server
from whspr._parakeet import ParakeetModel

model = server.load_model()
if isinstance(model, ParakeetModel):
    backend, device = "parakeet", "cpu"
else:
    backend, device = "whisper", model.model.device
result = {
    "backend": backend,
    "device": device,
    "text": server.transcribe_helper(sys.argv[1], model) if sys.argv[1] != "-" else "",
    "max_rss_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
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


def concatenate_wavs(paths, output):
    import wave

    with wave.open(output, "wb") as out:
        for i, path in enumerate(paths):
            with wave.open(path, "rb") as part:
                if i == 0:
                    out.setparams(part.getparams())
                out.writeframes(part.readframes(part.getnframes()))


def wav_seconds(path):
    import wave

    with wave.open(path, "rb") as wav:
        return wav.getnframes() / wav.getframerate()


def keywords_found(transcript, keywords, minimum):
    transcript = transcript.lower()
    return sum(1 for k in keywords if k.lower() in transcript) >= minimum


def test_cpu_machine_loads_parakeet():
    probe = probe_load_model(cuda_visible="")
    assert probe["backend"] == "parakeet"
    assert probe["device"] == "cpu"


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


def test_cpu_transcribes_french_detecting_the_language_itself(tmp_path):
    wav = make_speech_wav(tmp_path, FRENCH_TEXT, voice="fr", name="french.wav")
    probe = probe_load_model(wav=wav, cuda_visible="")
    assert probe["backend"] == "parakeet"
    assert keywords_found(
        probe["text"], ["bonjour", "réserver", "table", "personnes", "soir"], 3
    ), probe["text"]


@pytest.mark.parametrize("peak_dbfs", [-55, -62])
def test_cpu_transcribes_quiet_speech(tmp_path, peak_dbfs):
    """Low microphone gain or a distant speaker: Parakeet garbles speech this
    quiet unless it is amplified first (Whisper small coped with it)."""
    import wave

    import numpy as np
    from faster_whisper import decode_audio

    speech = decode_audio(make_speech_wav(tmp_path, ENGLISH_TEXT, voice="en-us"))
    speech *= 10 ** (peak_dbfs / 20) / np.max(np.abs(speech))
    path = tmp_path / "quiet.wav"
    with wave.open(str(path), "wb") as wav:  # 16-bit, as arecord records it
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(np.round(speech * 32767).astype("<i2").tobytes())
    probe = probe_load_model(wav=str(path), cuda_visible="")
    assert keywords_found(probe["text"], ["quick", "brown", "fox", "lazy", "dog"], 3), (
        probe["text"]
    )


def test_cpu_transcribes_silence_as_empty_text(tmp_path):
    import wave

    path = tmp_path / "silence.wav"
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(16000)
        wav.writeframes(b"\x00\x00" * 16000 * 3)
    probe = probe_load_model(wav=str(path), cuda_visible="")
    assert probe["text"] == ""


def test_cpu_transcribes_a_recording_too_long_for_one_pass(tmp_path):
    """Parakeet's encoder cannot take much more than 10 minutes in one pass
    (and needs ~5 GB for 6 minutes), so a long dictation must be chunked:
    every part must come through, in order, with bounded memory."""
    opening = make_speech_wav(tmp_path, LONG_OPENING, name="opening.wav")
    filler = make_speech_wav(tmp_path, LONG_FILLER, name="filler.wav")
    closing = make_speech_wav(tmp_path, LONG_CLOSING, name="closing.wav")
    long_wav = str(tmp_path / "long.wav")
    concatenate_wavs([opening] + [filler] * 40 + [closing], long_wav)
    assert wav_seconds(long_wav) > 12 * 60

    probe = probe_load_model(wav=long_wav, cuda_visible="")

    text = probe["text"].lower()
    assert "purple elephant" in text[:300], text[:300]
    assert "green giraffe" in text[-300:], text[-300:]
    assert text.index("purple elephant") < text.index("green giraffe")
    assert 36 <= text.count("quarterly report") <= 40  # nothing lost or doubled
    assert probe["max_rss_mb"] < 3000, probe["max_rss_mb"]


def test_library_api_transcribes_compressed_audio_on_cpu(tmp_path):
    """`whspr.transcribe()` accepts any audio format, as with Whisper."""
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg is needed to create the compressed test file")
    wav = make_speech_wav(tmp_path, ENGLISH_TEXT, voice="en-us")
    mp3 = str(tmp_path / "speech.mp3")
    subprocess.run(["ffmpeg", "-v", "error", "-i", wav, mp3], check=True)

    result = subprocess.run(
        [sys.executable, "-c", "import sys, whspr; print(whspr.transcribe(sys.argv[1]))", mp3],
        env={**os.environ, "CUDA_VISIBLE_DEVICES": ""},
        capture_output=True,
        text=True,
        timeout=600,
    )
    assert result.returncode == 0, result.stderr
    assert keywords_found(result.stdout, ["quick", "brown", "fox", "lazy", "dog"], 3), (
        result.stdout
    )


def test_cpu_model_works_without_the_gpu_extras(tmp_path):
    """Installs without [gpu] have no `nvidia` package; the real CPU model
    must load and transcribe regardless."""
    from test_server_units import BLOCK_NVIDIA_PREAMBLE

    program = BLOCK_NVIDIA_PREAMBLE + (
        "import whspr.server as server\n"
        "model = server.load_model()\n"
        "print(server.transcribe_helper(sys.argv[1], model))\n"
    )
    wav = make_speech_wav(tmp_path, ENGLISH_TEXT, voice="en-us")
    result = subprocess.run(
        [sys.executable, "-c", program, wav], capture_output=True, text=True, timeout=600
    )
    assert result.returncode == 0, result.stderr
    assert keywords_found(result.stdout, ["quick", "brown", "fox", "lazy", "dog"], 3), (
        result.stdout
    )


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
