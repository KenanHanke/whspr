"""In-process unit tests for whspr._parakeet (no real model needed)."""

import os
import wave

import numpy as np
import pytest

import whspr._parakeet as parakeet
from whspr._parakeet import ParakeetModel, split_audio

RATE = parakeet.SAMPLE_RATE
MAX = int(parakeet._MAX_CHUNK_SECONDS * RATE)
SEARCH = int(parakeet._CUT_SEARCH_SECONDS * RATE)
MIN = int(parakeet._MIN_AUDIO_SECONDS * RATE)


def noise(seconds, seed=0, level=0.3):
    rng = np.random.default_rng(seed)
    return (rng.standard_normal(int(seconds * RATE)) * level).astype(np.float32)


def assert_valid_split(audio, chunks):
    # Every sample exactly once, in order: nothing dropped or duplicated.
    assert np.array_equal(np.concatenate(chunks), audio)
    assert all(len(chunk) <= MAX for chunk in chunks)
    for chunk in chunks[:-1]:
        assert len(chunk) >= MAX - SEARCH  # cuts only within the search window
    if len(audio) >= MIN:
        assert len(chunks[-1]) >= MIN  # never an untranscribable remainder


# --- split_audio -------------------------------------------------------------


@pytest.mark.parametrize("length", [0, 1, MIN - 1, MIN, RATE, MAX - 1, MAX])
def test_short_audio_is_a_single_chunk(length):
    audio = noise(length / RATE)
    chunks = split_audio(audio)
    assert len(chunks) == 1
    assert np.array_equal(chunks[0], audio)


@pytest.mark.parametrize(
    "length", [MAX + 1, MAX + MIN - 1, MAX + MIN, MAX + MIN + 1, 2 * MAX, 2 * MAX + 1]
)
def test_boundary_lengths_split_validly(length):
    audio = noise(length / RATE, seed=length)
    assert_valid_split(audio, split_audio(audio))


def test_tiny_overhang_is_not_left_as_its_own_chunk():
    """One sample past the limit must not yield a 1-sample final chunk (onnx-asr
    raises IndexError on that)."""
    audio = noise((MAX + 1) / RATE)
    chunks = split_audio(audio)
    assert len(chunks) == 2
    assert len(chunks[-1]) >= MIN


def test_cuts_land_in_pauses():
    # Continuous "speech" with a single short pause inside each search window,
    # placed a little before each chunk limit.
    limit = parakeet._MAX_CHUNK_SECONDS
    pauses = [limit - 8.0, 2 * limit - 15.5, 3 * limit - 22.0]
    audio = noise(pauses[-1] + 12.0)
    for pause in pauses:
        begin = int(pause * RATE)
        audio[begin : begin + int(0.4 * RATE)] = 0.0

    chunks = split_audio(audio)
    assert_valid_split(audio, chunks)
    cuts = np.cumsum([len(chunk) for chunk in chunks])[:-1] / RATE
    assert len(cuts) == len(pauses)
    for cut, pause in zip(cuts, pauses):
        assert pause <= cut <= pause + 0.4, (cut, pause)


def test_very_long_recording_splits_validly_and_quickly():
    import time

    minutes = 45
    audio = noise(minutes * 60)  # a 45-minute dictation
    started = time.monotonic()
    chunks = split_audio(audio)
    assert time.monotonic() - started < 5.0
    assert_valid_split(audio, chunks)
    assert len(chunks) >= minutes * 60 / parakeet._MAX_CHUNK_SECONDS


def test_digital_silence_splits_validly():
    audio = np.zeros(int(150 * RATE), dtype=np.float32)
    assert_valid_split(audio, split_audio(audio))


def test_clipped_audio_splits_validly():
    audio = np.ones(int(150 * RATE), dtype=np.float32)
    assert_valid_split(audio, split_audio(audio))


# --- ParakeetModel.transcribe --------------------------------------------------


class FakeAsr:
    def __init__(self, replies=None):
        self.calls = []
        self.replies = replies

    def recognize(self, waveform, sample_rate):
        assert sample_rate == RATE
        assert isinstance(waveform, np.ndarray)
        assert waveform.dtype == np.float32 and waveform.ndim == 1
        self.calls.append(len(waveform))
        if self.replies is not None:
            return self.replies[len(self.calls) - 1]
        return f" chunk{len(self.calls)} "


def write_wav(path, samples, rate=RATE, channels=1):
    pcm = (np.clip(samples, -1.0, 1.0) * 32767).astype("<i2")
    if channels > 1:
        pcm = np.repeat(pcm[:, None], channels, axis=1)
    with wave.open(str(path), "wb") as wav:
        wav.setnchannels(channels)
        wav.setsampwidth(2)
        wav.setframerate(rate)
        wav.writeframes(pcm.tobytes())
    return str(path)


THREE_CHUNKS_SECONDS = 2 * parakeet._MAX_CHUNK_SECONDS + 10.0


def test_transcribe_joins_chunk_texts_with_single_spaces(tmp_path):
    path = write_wav(tmp_path / "long.wav", noise(THREE_CHUNKS_SECONDS))
    asr = FakeAsr()
    assert ParakeetModel(asr).transcribe(path) == "chunk1 chunk2 chunk3"
    assert len(asr.calls) == 3
    assert all(n <= MAX for n in asr.calls)
    assert sum(asr.calls) == int(THREE_CHUNKS_SECONDS * RATE)


def test_transcribe_skips_empty_chunk_results(tmp_path):
    path = write_wav(tmp_path / "long.wav", noise(THREE_CHUNKS_SECONDS))
    asr = FakeAsr(replies=["  First.", "   ", "Third. "])
    assert ParakeetModel(asr).transcribe(path) == "First. Third."


@pytest.mark.parametrize("samples", [0, 1, 2, MIN - 1])
def test_transcribe_returns_empty_text_for_too_short_audio(tmp_path, samples):
    path = write_wav(tmp_path / "tiny.wav", noise(samples / RATE))
    asr = FakeAsr()
    assert ParakeetModel(asr).transcribe(path) == ""
    assert asr.calls == []  # the model never sees an input it would choke on


def test_transcribe_sends_audio_of_exactly_the_minimum_length(tmp_path):
    path = write_wav(tmp_path / "short.wav", noise(MIN / RATE))
    asr = FakeAsr()
    assert ParakeetModel(asr).transcribe(path) == "chunk1"
    assert asr.calls == [MIN]


# --- quiet audio -------------------------------------------------------------


def test_quiet_audio_is_amplified_uniformly():
    quiet = noise(1, level=0.001)  # peaks near -47 dBFS
    louder = parakeet._amplify_if_quiet(quiet)
    assert louder.dtype == np.float32
    assert np.allclose(louder, quiet * parakeet._QUIET_GAIN)


def test_amplification_is_moderate_so_near_silence_stays_quiet():
    one_bit = np.full(RATE, 1 / 32768, dtype=np.float32)
    louder = parakeet._amplify_if_quiet(one_bit)
    assert np.max(louder) < 0.001  # still below -60 dBFS


@pytest.mark.parametrize("peak", [parakeet._QUIET_PEAK * 1.01, 0.05, 0.3, 0.9, 1.0])
def test_audio_at_normal_levels_is_left_untouched(peak):
    """Gain changes transcripts subtly even where it is not needed, so audio
    at or above the quiet threshold must reach the model exactly as is."""
    audio = noise(1, level=peak / 10)
    audio[0] = peak  # a known peak
    assert parakeet._amplify_if_quiet(audio) is audio


def test_digital_silence_is_left_untouched():
    silence = np.zeros(RATE, dtype=np.float32)
    assert parakeet._amplify_if_quiet(silence) is silence


def test_transcribe_amplifies_quiet_chunks_before_recognition(tmp_path):
    path = write_wav(tmp_path / "quiet.wav", noise(2, level=0.001))
    peaks = []

    class PeakRecordingAsr(FakeAsr):
        def recognize(self, waveform, sample_rate):
            peaks.append(float(np.max(np.abs(waveform))))
            return super().recognize(waveform, sample_rate)

    ParakeetModel(PeakRecordingAsr()).transcribe(path)
    [peak] = peaks
    assert peak > parakeet._QUIET_PEAK  # it reached the model amplified


def test_usable_physical_cores_counts_hyperthread_siblings_once():
    cores = parakeet._usable_physical_cores()
    assert 1 <= cores <= len(os.sched_getaffinity(0))


def test_usable_physical_cores_without_topology_counts_every_cpu(monkeypatch):
    import builtins

    real_open = builtins.open

    def no_sysfs(path, *args, **kwargs):
        if str(path).startswith("/sys/devices/system/cpu/"):
            raise FileNotFoundError(path)
        return real_open(path, *args, **kwargs)

    monkeypatch.setattr(builtins, "open", no_sysfs)
    assert parakeet._usable_physical_cores() == len(os.sched_getaffinity(0))


def test_transcribe_resamples_and_downmixes_any_input(tmp_path):
    """Recordings from other tools (44.1 kHz stereo, say) must be converted
    to what the model expects, as faster-whisper did before."""
    three_seconds_at_44k = noise(3 * 44100 / RATE)
    path = write_wav(tmp_path / "cd.wav", three_seconds_at_44k, rate=44100, channels=2)
    asr = FakeAsr()
    assert ParakeetModel(asr).transcribe(path) == "chunk1"
    assert abs(asr.calls[0] - 3 * RATE) <= RATE // 100


def test_transcribe_raises_for_missing_file(tmp_path):
    with pytest.raises(Exception):
        ParakeetModel(FakeAsr()).transcribe(str(tmp_path / "missing.wav"))


def test_transcribe_raises_for_non_audio_file(tmp_path):
    path = tmp_path / "not-audio.wav"
    path.write_text("this is not audio")
    with pytest.raises(Exception):
        ParakeetModel(FakeAsr()).transcribe(str(path))


def test_module_import_stays_cheap():
    """The client imports the server (and with it this module) on every
    keypress, before the start sound; heavy imports must stay lazy."""
    import subprocess
    import sys

    program = (
        "import sys, whspr.client\n"
        "heavy = {'numpy', 'onnx_asr', 'onnxruntime', 'faster_whisper', 'ctranslate2'}\n"
        "print(sorted(heavy & set(sys.modules)))\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", program], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "[]"
