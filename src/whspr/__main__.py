from __future__ import annotations

from ._cuda_bootstrap import ensure_cuda_runtime_loaded

ensure_cuda_runtime_loaded()

from faster_whisper import WhisperModel

model = WhisperModel(
    "turbo",
    device="cuda",
    compute_type="float16",
)

def transcribe(path) -> str:
    segments, info = model.transcribe(path)
    text = "".join(seg.text for seg in segments)
    return text

def main() -> None:
    ...

if __name__ == "__main__":
    main()
