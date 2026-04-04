# linux_speech_io.py
"""
Small Linux-only helpers for a speech-recognition workflow built around
`arecord`, `aplay`, a Unix domain socket, and a lock file.

Expected flow:
1) One process acquires the lock and calls `record_until_stop()`.
2) A second process fails to acquire the lock, so it calls
   `stop_transcribe_copy_and_notify()`.
3) The recorder receives the stop request over the Unix socket, stops
   `arecord`, finalizes the WAV file, and replies that the recording is ready.
"""

from __future__ import annotations

import errno
import fcntl
import signal
import socket
import subprocess
import time
from pathlib import Path
from typing import Any
from importlib.resources import files
import pyperclip
from . import server


START_SOUND = str(files("whspr").joinpath("data/sounds/start.wav"))
STOP_SOUND = str(files("whspr").joinpath("data/sounds/stop.wav"))
FINISHED_SOUND = str(files("whspr").joinpath("data/sounds/finished.wav"))

RECORDING_PATH = "/tmp/whspr-recording.wav"
SOCKET_PATH = "/tmp/whspr-recorder.sock"
LOCK_PATH = "/tmp/whspr-recorder.lock"

# Speech-friendly defaults for ASR.
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_FORMAT = "S16_LE"

_STOP_REQUEST = b"STOP\n"


def _unlink_if_exists(path: str | Path) -> None:
    """Remove a filesystem entry if it exists."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _read_pipe_text(pipe: Any) -> str:
    """Read and normalize subprocess stderr text."""
    if pipe is None:
        return ""
    text = pipe.read()
    return text.strip() if text else ""


def _raise_process_error(name: str, proc: subprocess.Popen[str]) -> None:
    """Raise a RuntimeError with stderr attached for a failed subprocess."""
    stderr = _read_pipe_text(proc.stderr)
    message = f"{name} failed with exit code {proc.returncode}"
    if stderr:
        message += f": {stderr}"
    raise RuntimeError(message)


def _play_wav_blocking(path: str) -> None:
    """Play a WAV file and wait until playback finishes."""
    proc = subprocess.run(
        ["aplay", "-q", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        message = f"aplay failed with exit code {proc.returncode}"
        if proc.stderr:
            message += f": {proc.stderr.strip()}"
        raise RuntimeError(message)


def _play_wav_background(path: str) -> subprocess.Popen[str]:
    """Play a WAV file in the background and return the running process."""
    return subprocess.Popen(
        ["aplay", "-q", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _wait_for_success(name: str, proc: subprocess.Popen[str]) -> None:
    """Wait for a background subprocess and raise on failure."""
    proc.wait()
    if proc.returncode != 0:
        _raise_process_error(name, proc)


def _start_arecord(recording_path: str) -> subprocess.Popen[str]:
    """
    Start continuous recording to a WAV file.

    The command is intentionally explicit about sample format, channel count,
    and sample rate so the recorded file is predictable for ASR.
    """
    return subprocess.Popen(
        [
            "arecord",
            "-q",
            "-t",
            "wav",
            "-f",
            SAMPLE_FORMAT,
            "-c",
            str(CHANNELS),
            "-r",
            str(SAMPLE_RATE),
            recording_path,
        ],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _stop_arecord_cleanly(proc: subprocess.Popen[str]) -> None:
    """
    Ask arecord to stop in a way that lets it finalize the WAV file cleanly.

    We try SIGINT first, then SIGTERM, and only fall back to SIGKILL if the
    recorder becomes unresponsive.
    """
    if proc.poll() is not None:
        return

    proc.send_signal(signal.SIGINT)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    proc.send_signal(signal.SIGTERM)
    try:
        proc.wait(timeout=5)
        return
    except subprocess.TimeoutExpired:
        pass

    proc.kill()
    proc.wait(timeout=5)


def _recv_all(sock: socket.socket) -> bytes:
    """Read from a socket until the peer closes the connection."""
    chunks: list[bytes] = []
    while True:
        chunk = sock.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
    return b"".join(chunks)


def _ensure_recording_exists(recording_path: str) -> None:
    """Basic sanity-check that the recorder produced a non-empty WAV file."""
    path = Path(recording_path)
    if not path.exists():
        raise RuntimeError(f"Recording file was not created: {recording_path}")
    if path.stat().st_size <= 44:
        raise RuntimeError(f"Recording file looks empty: {recording_path}")


def record_until_stop() -> str:
    """
    Play `start_sound`, then start continuously recording microphone audio to
    `recording_path`. While recording, wait for a stop request on the Unix
    socket at `socket_path`. When the stop request arrives, stop `arecord`,
    finalize the file, and reply over the same open socket connection with:

        READY /tmp/whspr-recording.wav

    Returns the recording path once the file is ready.
    """
    _play_wav_blocking(START_SOUND)

    _unlink_if_exists(SOCKET_PATH)
    Path(RECORDING_PATH).parent.mkdir(parents=True, exist_ok=True)

    server_sock: socket.socket | None = None
    conn: socket.socket | None = None
    recorder: subprocess.Popen[str] | None = None

    try:
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(str(SOCKET_PATH))
        server_sock.listen(1)

        recorder = _start_arecord(RECORDING_PATH)

        # Catch obvious startup failures early instead of hanging on accept().
        time.sleep(0.1)
        if recorder.poll() is not None:
            _raise_process_error("arecord", recorder)

        conn, _ = server_sock.accept()
        with conn:
            stop_request = conn.recv(4096)
            if not stop_request:
                raise RuntimeError("Stop client disconnected before sending a stop request.")

            _stop_arecord_cleanly(recorder)
            _ensure_recording_exists(RECORDING_PATH)

            conn.sendall(f"READY {RECORDING_PATH}\n".encode("utf-8"))

        return RECORDING_PATH

    except Exception as exc:
        if conn is not None:
            try:
                conn.sendall(f"ERROR {exc}\n".encode("utf-8"))
            except OSError:
                pass
        raise

    finally:
        if recorder is not None and recorder.poll() is None:
            _stop_arecord_cleanly(recorder)

        if server_sock is not None:
            server_sock.close()

        _unlink_if_exists(SOCKET_PATH)


def request_stop_and_wait(
    connect_timeout: float = 15.0,
    retry_interval: float = 0.05,
) -> str:
    """
    Connect to the recorder's Unix socket from another Python process, send the
    stop signal, then wait for the recorder to reply that the WAV file is ready.

    Returns the ready recording path.
    """
    deadline = time.monotonic() + connect_timeout
    last_error: OSError | None = None

    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            break

        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as sock:
                sock.settimeout(remaining)
                sock.connect(str(SOCKET_PATH))
                sock.sendall(_STOP_REQUEST)
                sock.shutdown(socket.SHUT_WR)

                response = _recv_all(sock).decode("utf-8").strip()

            if response.startswith("READY "):
                return response[len("READY ") :]

            if response.startswith("ERROR "):
                raise RuntimeError(response[len("ERROR ") :])

            raise RuntimeError(f"Unexpected recorder response: {response!r}")

        except OSError as exc:
            last_error = exc
            if exc.errno in (errno.ENOENT, errno.ECONNREFUSED, errno.ECONNRESET):
                time.sleep(min(retry_interval, max(remaining, 0.0)))
                continue
            raise

    raise TimeoutError(
        f"Could not connect to recorder socket {SOCKET_PATH!r} within "
        f"{connect_timeout} seconds"
    ) from last_error


def stop_transcribe_copy_and_notify():
    """
    1) Call `request_stop_and_wait()`.
    2) Start playing `stop_sound` in the background.
    3) While that sound is still playing, run:
           pyperclip.copy(server.transcribe(recording_path))
    4) After both playback and transcription have finished, play
       `finished_sound`.
    """
    request_stop_and_wait()
    
    stop_proc = _play_wav_background(STOP_SOUND)
    try:
        transcript = str(server.transcribe(RECORDING_PATH))
        pyperclip.copy(transcript)
    finally:
        _wait_for_success("aplay", stop_proc)

    _play_wav_blocking(FINISHED_SOUND)


def main():
    """
    Try to acquire an exclusive non-blocking lock on `lock_path`.

    - If locking succeeds, this process becomes the recorder and calls
      `record_until_stop()`.
    - If locking fails because another process already holds the lock, this
      process behaves as the stop/transcribe side and calls
      `stop_transcribe_copy_and_notify()`.
    """
    Path(LOCK_PATH).parent.mkdir(parents=True, exist_ok=True)

    # Open in append mode so the file exists and is writable.
    with open(LOCK_PATH, "a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError as exc:
            if exc.errno not in (errno.EACCES, errno.EAGAIN):
                raise

            stop_transcribe_copy_and_notify()
            return

        try:
            record_until_stop()
            return
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()