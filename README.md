# whspr

A minimalist dictation tool for local speech recognition using OpenAI's Whisper and
NVIDIA's Parakeet models. Its interface is fully keyboard-driven and sound-based so as
not to interfere with windowing or application focus.

Processing is done locally. If `whspr[gpu]` optional dependencies are installed and an
Nvidia GPU is available, the model `whisper-large-v3-turbo` will be used via `faster-whisper`;
otherwise, [`parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
will be run on the CPU via `onnx-asr`. `whspr` is currently only available on Linux and
can be installed from [PyPI](https://pypi.org/project/whspr/).

Transcription happens in a background server process that keeps the model warm
between dictations and shuts itself down automatically after five minutes of
inactivity; it is started (and restarted) on demand, so this is invisible in use.
To free the model's memory without waiting that long, stop the server with
`whspr --stop-server`.

## Usage

Bind the following commands to your preferred keyboard shortcuts (examples given here).

```bash
whspr            # Super+C
whspr --paste    # Super+V
whspr --cancel   # Super+X
```

In the example given, `Super+C` and `Super+V` will both start or stop dictation and copy the
result to the clipboard. The difference is that `Super+V` will additionally paste the result
into the currently focussed application. `Super+X` will cancel any dictation currently in progress.
Sounds will indicate when `whspr` is listening and when it has finished processing.

The background server can also be stopped from the command line before its five minutes
of inactivity have passed; a transcription that is already underway still completes, and
the next dictation starts the server again:

```bash
whspr --stop-server
```

`whspr` can also be accessed from within Python:

```python
from whspr import transcribe
result = transcribe("path/to/audio.mp3")
```

## Installation

`whspr` depends on:

- Python 3.10 or newer.
- the `aplay`, `arecord`, `ydotool` commands. The former two are part of the
  `alsa-utils` package and installed on most distros by default. `ydotool` is optional
  and only required for the `--paste` flag (see [Usage](#usage)).
- a clipboard backend compatible with `pyperclip`, e.g. `wl-clipboard` on Wayland or `xclip` on X11.
- for optional GPU-accelerated speech recognition, an Nvidia GPU and drivers are required.

On Ubuntu, simply run:

```bash
sudo apt update && sudo apt install -y alsa-utils wl-clipboard xclip ydotool pipx
pipx install 'whspr[gpu]'  # gpu support is optional; omit [gpu] if it's not desired
whspr --finish-setup     # optional to pre-load the model from the internet before its first use
```
