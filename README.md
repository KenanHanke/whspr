# whspr

A minimalist dictation tool for local speech recognition using OpenAI's Whisper models.
Its interface is fully keyboard-driven and sound-based so as not to interfere with
windowing or application focus.

Processing is done locally using `faster-whisper`. If `whspr[gpu]` optional dependencies
are installed and an Nvidia GPU is available, the current version of `whisper-large-turbo`
will be used; otherwise, `whisper-small.en` will be used. `whspr` is currently only
available on Linux.

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

## Installation

`whspr` depends on:

- the `aplay`, `arecord`, `ydotool` commands. The former two are part of the
  `alsa-utils` package and installed on most systems already. `ydotool` is optional
  and only required for the `--paste` flag (see [Usage](#usage)).
- a clipboard backend compatible with `pyperclip`, e.g. `wl-clipboard` on Wayland or `xclip` on X11.
- for optional GPU-accelerated speech recognition, an Nvidia GPU and drivers are required.

On Ubuntu, run:

```bash
sudo apt update && sudo apt install -y alsa-utils wl-clipboard xclip ydotool
pip install whspr[gpu]  # gpu support is optional; omit [gpu] if it's not desired
```

`whspr` can also be used from within Python:

```python
from whspr import transcribe
result = transcribe("path/to/audio.mp3")
```
