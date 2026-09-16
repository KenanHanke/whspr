# src/whspr/_parakeet.py
"""CPU speech recognition with NVIDIA's Parakeet TDT 0.6B v3.

The model runs through onnx-asr, i.e. on plain onnxruntime (no PyTorch or
NeMo), using its int8-quantized ONNX export: roughly a 670 MB download and
1 GB of RAM, while transcribing much faster than real time on a laptop CPU.
It recognizes 25 European languages and detects the language by itself.

Heavy dependencies are imported where they are used, keeping this module
cheap to import from the latency-sensitive client.
"""

import os

MODEL_NAME = "nemo-parakeet-tdt-0.6b-v3"
QUANTIZATION = "int8"
SAMPLE_RATE = 16000

# The encoder attends over its entire input at once, so memory grows
# quadratically with length (~5 GB for 6 minutes of audio), and the exported
# model fails outright somewhere past 10 minutes.  Longer recordings are
# therefore transcribed in chunks of at most this many seconds...
_MAX_CHUNK_SECONDS = 60.0
# ...each ending at the quietest moment within this many seconds before the
# limit, so that cuts land in pauses between words rather than inside them.
_CUT_SEARCH_SECONDS = 15.0
# The stretch of audio whose loudness decides where the quietest moment is.
_PAUSE_WINDOW_SECONDS = 0.3
_PAUSE_HOP_SECONDS = 0.01
# Shorter audio cannot hold a word, and the model chokes on a few samples
# (onnx-asr raises IndexError for a single one); it is transcribed as "".
_MIN_AUDIO_SECONDS = 0.1
# Speech peaking below about -50 dBFS (low microphone gain, a distant
# speaker) comes out garbled or empty, so chunks peaking below _QUIET_PEAK
# are amplified by _QUIET_GAIN first.  Louder audio is left alone (gain
# would subtly change its transcripts), and the fixed, moderate gain never
# blows silence or hum up into imagined words.
_QUIET_PEAK = 10 ** (-40 / 20)  # -40 dBFS
_QUIET_GAIN = 10 ** (30 / 20)  # +30 dB
# onnxruntime's default of one spin-waiting thread per core makes the
# model's many tiny decoder steps burn every core for no gain in speed;
# a few sleeping threads, one per physical core, are as fast at a fraction
# of the CPU time.
_MAX_THREADS = 8


class ParakeetModel:
    """A loaded Parakeet model; see `transcribe()`."""

    def __init__(self, asr):
        self._asr = asr

    @classmethod
    def load(cls):
        """Load the model, downloading it on first use.

        onnx-asr looks for already-downloaded files before touching the
        network, so a warm cache works offline.
        """
        import onnx_asr

        threads = min(_MAX_THREADS, _usable_physical_cores())
        # Pinned to the CPU: if onnxruntime-gpu happens to be installed,
        # onnx-asr would otherwise try its CUDA/TensorRT providers first.
        cpu_only = ["CPUExecutionProvider"]
        asr = onnx_asr.load_model(
            MODEL_NAME,
            quantization=QUANTIZATION,
            providers=cpu_only,
            sess_options=_session_options(threads),
            # Only used for input that is not 16 kHz already, which never
            # reaches it; one thread each keeps its seven sessions cheap.
            resampler_config={"sess_options": _session_options(1), "providers": cpu_only},
        )
        return cls(asr)

    def transcribe(self, path):
        """Transcribe an audio file of any format PyAV can decode; return text."""
        # faster-whisper's decoder (PyAV) handles every common format and
        # resamples to 16 kHz mono; onnx-asr's own reader only takes PCM WAV.
        from faster_whisper import decode_audio

        audio = decode_audio(path, sampling_rate=SAMPLE_RATE)
        min_length = int(_MIN_AUDIO_SECONDS * SAMPLE_RATE)
        texts = []
        for chunk in split_audio(audio):
            if len(chunk) >= min_length:
                chunk = _amplify_if_quiet(chunk)
                text = self._asr.recognize(chunk, sample_rate=SAMPLE_RATE).strip()
                if text:
                    texts.append(text)
        return " ".join(texts)


def _session_options(threads):
    import onnxruntime as rt

    options = rt.SessionOptions()
    options.intra_op_num_threads = threads
    options.inter_op_num_threads = 1
    options.add_session_config_entry("session.intra_op.allow_spinning", "0")
    # Without the arena, memory a long dictation needed is given back
    # instead of staying with the idle server.
    options.enable_cpu_mem_arena = False
    return options


def _usable_physical_cores():
    """Physical cores this process may run on (hyperthreads counted once)."""
    try:
        cpus = os.sched_getaffinity(0)
    except (AttributeError, OSError):
        return os.cpu_count() or 1
    cores = set()
    for cpu in cpus:
        try:
            path = f"/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list"
            with open(path) as siblings:
                cores.add(siblings.read().strip())
        except OSError:
            return max(len(cpus), 1)  # no topology info: count every CPU
    return max(len(cores), 1)


def _amplify_if_quiet(chunk):
    import numpy as np

    peak = float(np.max(np.abs(chunk)))
    if peak == 0.0 or peak >= _QUIET_PEAK:
        return chunk
    return chunk * np.float32(_QUIET_GAIN)


def split_audio(audio, sample_rate=SAMPLE_RATE):
    """Split audio into consecutive chunks the model can take in one pass.

    The chunks together cover every sample exactly once, in order.  Each is
    at most `_MAX_CHUNK_SECONDS` long; all but the last are cut at a pause,
    and the last is never shorter than `_MIN_AUDIO_SECONDS` unless the
    whole input is.
    """
    max_length = int(_MAX_CHUNK_SECONDS * sample_rate)
    search = int(_CUT_SEARCH_SECONDS * sample_rate)
    min_length = int(_MIN_AUDIO_SECONDS * sample_rate)
    window = int(_PAUSE_WINDOW_SECONDS * sample_rate)
    hop = max(int(_PAUSE_HOP_SECONDS * sample_rate), 1)

    chunks = []
    start = 0
    while len(audio) - start > max_length:
        # Never leave a remainder too short to transcribe.
        latest = min(start + max_length, len(audio) - min_length)
        cut = _quietest_point(audio, start + max_length - search, latest, window, hop)
        chunks.append(audio[start:cut])
        start = cut
    chunks.append(audio[start:])
    return chunks


def _quietest_point(audio, lo, hi, window, hop):
    """The index in [lo, hi] centred in the quietest `window` samples."""
    import numpy as np

    half = window // 2
    begin = max(lo - half, 0)
    end = min(hi + half, len(audio))
    power = np.square(audio[begin:end], dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(power)))

    centres = np.arange(lo, hi + 1, hop)
    window_starts = np.clip(centres - half, begin, end) - begin
    window_ends = np.clip(centres + half, begin, end) - begin
    # Mean rather than total power: windows clipped at the edges are shorter.
    mean_power = (cumulative[window_ends] - cumulative[window_starts]) / np.maximum(
        window_ends - window_starts, 1
    )
    return int(centres[np.argmin(mean_power)])
