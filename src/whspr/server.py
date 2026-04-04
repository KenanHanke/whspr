from ._cuda_bootstrap import ensure_cuda_runtime_loaded

ensure_cuda_runtime_loaded()

from faster_whisper import WhisperModel


def transcribe(path) -> str:
    model = WhisperModel(
        "turbo",
        device="cuda",
        compute_type="float16",
    )
    segments, info = model.transcribe(path)
    text = "".join(seg.text for seg in segments)
    return text.strip()
