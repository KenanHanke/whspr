# whspr

An interactive dictation software package for Linux optimized for low latency and high accuracy.

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
  and only required for the `--paste` option (see [Usage](#usage)).
- a clipboard backend compatible with `pyperclip` to be available, e.g. `wl-clipboard` on Wayland or `xclip` on X11.
- for optional local speech recognition, an Nvidia GPU and drivers are required.

On Ubuntu, run:

```bash
sudo apt update && sudo apt install -y alsa-utils wl-clipboard xclip ydotool
pip install whspr[gpu]  # gpu support is optional; omit [gpu] if it's not desired
```
