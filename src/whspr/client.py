# src/whspr/client.py
"""
Small Linux-only recording client for a speech-recognition workflow built
around `arecord`, `aplay`, a Unix domain socket, and a lock file.

Expected flow:
1) One process acquires the lock and calls `record_until_stop()`.
2) A second process fails to acquire the lock, so it calls
   `stop_transcribe_copy_and_notify()`.
3) The recorder receives the stop request over the Unix socket, stops
   `arecord`, finalizes the WAV file, and replies that the recording is ready.
4) The second process waits for the ready reply, then transcribes the recording
    and copies the transcript to the clipboard while playing a sound.
"""

from __future__ import annotations

import errno
import fcntl
import glob
import os
import re
import signal
import socket
import subprocess
import sys
import tempfile
import threading
import time
from pathlib import Path
from importlib.resources import files
import pyperclip
from . import server
from ._paths import runtime_file


START_SOUND = str(files("whspr").joinpath("data/sounds/start.wav"))
STOP_SOUND = str(files("whspr").joinpath("data/sounds/stop.wav"))
FINISHED_SOUND = str(files("whspr").joinpath("data/sounds/finished.wav"))
CANCELLED_SOUND = str(files("whspr").joinpath("data/sounds/cancelled.wav"))

# Per-user paths: recordings are private, and concurrent users must not
# collide on a shared machine.
SOCKET_PATH = runtime_file("whspr-recorder.sock")
LOCK_PATH = runtime_file("whspr-recorder.lock")

# Speech-friendly defaults for ASR.
SAMPLE_RATE = 16000
CHANNELS = 1
SAMPLE_FORMAT = "S16_LE"

_STOP_REQUEST = b"STOP\n"

# Sounds are ~0.3s long; a playback lasting longer than this means the audio
# stack is wedged, and a stuck aplay must never wedge a dictation.
_PLAYBACK_TIMEOUT = 10.0

# How often the recorder checks whether arecord died while waiting for a stop.
_RECORDER_POLL_INTERVAL = 0.5

# How long the stop side waits for the recorder's READY/ERROR reply once
# connected.  Must exceed _stop_arecord_cleanly's full SIGINT/SIGTERM/SIGKILL
# escalation (up to ~15s) so a slow-but-progressing stop never loses the
# finished recording to a timeout.
_STOP_REPLY_TIMEOUT = 30.0

# Only recordings this old are swept as stale.  A dead recorder's file can
# still be legitimately in flight (the stop side queues it and the server may
# transcribe it much later, e.g. after a slow model download), so the age must
# comfortably exceed the server client's 30-minute result timeout.
_STALE_RECORDING_AGE = 2 * 3600.0


def _new_recording_path() -> str:
    """A recording path unique to this recorder process.

    Unique names keep an in-flight transcription safe from being overwritten
    when the user immediately starts the next dictation.
    """
    return runtime_file(f"whspr-recording-{os.getpid()}.wav")


def _sweep_stale_recordings() -> None:
    """Delete recordings left behind by recorder processes that are gone.

    Successful and cancelled dictations clean up after themselves, but a
    SIGKILLed recorder cannot; without a sweep those files would sit in the
    (RAM-backed) runtime dir until logout.
    """
    directory = os.path.dirname(_new_recording_path())
    for path in glob.glob(os.path.join(directory, "whspr-recording-*.wav")):
        match = re.search(r"whspr-recording-(\d+)", os.path.basename(path))
        if not match:
            continue
        pid = int(match.group(1))
        if pid == os.getpid() or os.path.exists(f"/proc/{pid}"):
            continue
        try:
            if time.time() - os.stat(path).st_mtime < _STALE_RECORDING_AGE:
                continue
            os.unlink(path)
        except OSError:
            pass


def _unlink_if_exists(path: str | Path) -> None:
    """Remove a filesystem entry if it exists."""
    try:
        Path(path).unlink()
    except FileNotFoundError:
        pass


def _wait_for_playback(name: str, proc: subprocess.Popen[str]) -> None:
    """Wait for a sound-playing subprocess, kill it if stuck, raise on failure."""
    try:
        proc.wait(timeout=_PLAYBACK_TIMEOUT)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        raise RuntimeError(f"{name} did not finish playing within {_PLAYBACK_TIMEOUT}s")
    if proc.returncode != 0:
        stderr = proc.stderr.read().strip() if proc.stderr else ""
        message = f"{name} failed with exit code {proc.returncode}"
        if stderr:
            message += f": {stderr}"
        raise RuntimeError(message)


def play_wav_blocking(path: str) -> None:
    """Play a WAV file and wait until playback finishes."""
    _wait_for_playback("aplay", play_wav_background(path))


def play_wav_background(path: str) -> subprocess.Popen[str]:
    """Play a WAV file in the background and return the running process."""
    return subprocess.Popen(
        ["aplay", "-q", path],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.PIPE,
        text=True,
    )


def _start_arecord(recording_path: str):
    """Start continuous recording to a WAV file.

    The command is intentionally explicit about sample format, channel count,
    and sample rate so the recorded file is predictable for ASR.  stderr goes
    to a temp file rather than a pipe: a pipe nobody drains could fill up
    during a long recording and freeze arecord mid-dictation.

    Returns the process and the (readable) stderr log file.
    """
    stderr_log = tempfile.TemporaryFile(mode="w+", encoding="utf-8", errors="replace")
    process = subprocess.Popen(
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
        stderr=stderr_log,
    )
    return process, stderr_log


def _arecord_error(proc: subprocess.Popen, stderr_log) -> RuntimeError:
    """Build an error describing why arecord exited."""
    try:
        stderr_log.seek(0)
        stderr = stderr_log.read().strip()
    except (OSError, ValueError):
        stderr = ""
    message = f"arecord failed with exit code {proc.returncode}"
    if stderr:
        message += f": {stderr}"
    return RuntimeError(message)


def _stop_arecord_cleanly(proc: subprocess.Popen) -> None:
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


def _play_sound_best_effort(path: str, warn: bool = True) -> None:
    """Play a sound without ever letting its failure change the outcome.

    Used for the notification sounds that play once a dictation has already
    finished (succeeded or failed): a broken/busy audio device breaks every
    aplay call identically, and that must never turn a completed dictation
    into an error.  `warn=False` stays silent, for the error path where the
    real exception is about to be raised anyway.
    """
    try:
        play_wav_blocking(path)
    except (RuntimeError, OSError) as exc:
        if warn:
            print(
                f"whspr: could not play {os.path.basename(path)}: {exc}",
                file=sys.stderr,
            )


def _paste_with_ydotool() -> None:
    try:
        process = subprocess.Popen(
            ["ydotool", "key", "29:1", "47:1", "47:0", "29:0"],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            # Let ydotool's own error through to stderr — the common failure is
            # "failed to connect socket .ydotool_socket" (ydotoold not running),
            # which is invisible if discarded, even when debugging from a shell.
            start_new_session=True,
            close_fds=True,
        )
    except (FileNotFoundError, OSError) as exc:
        print(f"whspr: could not run ydotool for --paste: {exc}", file=sys.stderr)
        return

    def reap():
        if process.wait() != 0:
            print(
                "whspr: --paste via ydotool failed (is the ydotoold daemon running?)",
                file=sys.stderr,
            )

    # Reap in the background so no zombie lingers inside long-lived callers.
    threading.Thread(target=reap, daemon=True).start()


def _accept_stop_connection(server_sock, recorder):
    """Wait for the stop client while watching the recorder's health.

    If arecord dies — whether right at startup (mic busy or missing) or
    mid-recording — play the cancelled sound right away so the user knows
    their dictation is not being captured, then keep waiting so the eventual
    stop press receives a proper ERROR reply instead of becoming a fresh,
    equally doomed recording.
    """
    recorder_died = False
    server_sock.settimeout(_RECORDER_POLL_INTERVAL)
    while True:
        try:
            conn, _ = server_sock.accept()
            return conn, recorder_died
        except socket.timeout:
            if not recorder_died and recorder.poll() is not None:
                recorder_died = True
                _play_sound_best_effort(CANCELLED_SOUND)


def record_until_stop() -> str:
    """
    Play the start sound, then continuously record microphone audio to a
    per-dictation WAV file. While recording, wait for a stop request on the
    Unix socket at `SOCKET_PATH`. When the stop request arrives, stop
    `arecord`, finalize the file, and reply over the same open socket
    connection with:

        READY <recording-path>

    If the recording failed, reply with `ERROR <reason>` instead.
    Returns the recording path once the file is ready.

    The start sound is best-effort: a busy or broken audio *output* device
    must not stop the (input-side) recording from happening.
    """
    _play_sound_best_effort(START_SOUND)

    _sweep_stale_recordings()
    recording_path = _new_recording_path()
    _unlink_if_exists(SOCKET_PATH)
    Path(recording_path).parent.mkdir(parents=True, exist_ok=True)

    server_sock: socket.socket | None = None
    recorder: subprocess.Popen | None = None
    stderr_log = None

    try:
        server_sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        server_sock.bind(SOCKET_PATH)
        server_sock.listen(1)

        recorder, stderr_log = _start_arecord(recording_path)

        conn, recorder_died = _accept_stop_connection(server_sock, recorder)
        try:
            stop_request = conn.recv(4096)
            if not stop_request:
                raise RuntimeError("Stop client disconnected before sending a stop request.")

            if recorder_died or recorder.poll() is not None:
                raise _arecord_error(recorder, stderr_log)

            _stop_arecord_cleanly(recorder)
            _ensure_recording_exists(recording_path)

            conn.sendall(f"READY {recording_path}\n".encode("utf-8"))
        except Exception as exc:
            # Reply while the connection is still open so the stop side can
            # report the real failure instead of an empty response.  The
            # partial recording is worthless once the failure is delivered.
            try:
                conn.sendall(f"ERROR {exc}\n".encode("utf-8"))
            except OSError:
                pass
            _unlink_if_exists(recording_path)
            raise
        finally:
            conn.close()

        return recording_path

    finally:
        if recorder is not None and recorder.poll() is None:
            _stop_arecord_cleanly(recorder)

        if stderr_log is not None:
            stderr_log.close()

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
                # Once connected, the reply gets its own larger budget: the
                # recorder may legitimately need up to ~15s to stop a stuck
                # arecord before it can send READY.
                sock.settimeout(_STOP_REPLY_TIMEOUT)
                sock.sendall(_STOP_REQUEST)
                sock.shutdown(socket.SHUT_WR)

                try:
                    response = _recv_all(sock).decode("utf-8").strip()
                except socket.timeout as exc:
                    raise TimeoutError(
                        "recorder did not finalize the recording within "
                        f"{_STOP_REPLY_TIMEOUT:.0f} seconds"
                    ) from exc

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


def cancel_recording():
    try:
        with open(LOCK_PATH, "a+") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                # This means the lock is held by a recording process, so we can proceed to cancel it.
                pass
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                return
    except OSError:
        return

    try:
        recording_path = request_stop_and_wait(connect_timeout=5.0)
    except (TimeoutError, RuntimeError, OSError):
        return  # the recording ended on its own before we could cancel it

    _unlink_if_exists(recording_path)
    _play_sound_best_effort(CANCELLED_SOUND)


def stop_transcribe_copy_and_notify(paste=False):
    """
    1) Call `request_stop_and_wait()` to obtain the finished recording.
    2) Start playing `STOP_SOUND` in the background.
    3) While that sound is still playing, transcribe the recording and copy
       the transcript to the clipboard.
    4) After both playback and transcription have finished, play
       `FINISHED_SOUND`.

    The recording file is deleted only after the transcript has safely landed
    in the clipboard, so a failure never discards the user's dictation.

    Because the interface is sound-based, any failure plays `CANCELLED_SOUND`
    so the user is never left with silence after the stop beep; the underlying
    error is still re-raised so it shows up when run from a terminal.
    """
    recording_path = None
    try:
        recording_path = request_stop_and_wait()

        stop_proc = _start_notification_sound(STOP_SOUND)
        try:
            transcript = str(server.transcribe(recording_path))
            pyperclip.copy(transcript)
        finally:
            # Once the transcript is safely in the clipboard, a failure of the
            # notification sound must not fail the dictation; only warn.
            _finish_notification_sound(stop_proc)
    except Exception as exc:
        # Silent here: the real error is re-raised right after, so a failed
        # cancelled sound must not add a second, misleading message.
        _play_sound_best_effort(CANCELLED_SOUND, warn=False)
        if recording_path and os.path.exists(recording_path):
            print(
                f"whspr: dictation failed ({exc}); "
                f"the recording is preserved at {recording_path}",
                file=sys.stderr,
            )
        raise

    # The dictation has fully succeeded (transcript on the clipboard); a broken
    # audio device must not now turn that success into a failure.
    _unlink_if_exists(recording_path)
    if paste:
        _paste_with_ydotool()
    _play_sound_best_effort(FINISHED_SOUND)


def _start_notification_sound(path):
    """Launch a background notification sound, or return None if it cannot even
    start (e.g. aplay missing, or a transient fork/fd exhaustion).

    This runs before transcription, so a failure to *launch* the sound must
    not prevent the dictation any more than a failure to *play* it does.
    """
    try:
        return play_wav_background(path)
    except OSError as exc:
        print(
            f"whspr: could not play {os.path.basename(path)}: {exc}",
            file=sys.stderr,
        )
        return None


def _finish_notification_sound(stop_proc):
    """Wait for the background stop sound; warn (never raise) if it failed.

    The transcript is already in the clipboard by the time this runs, so a
    broken notification sound must not present a successful dictation as a
    failure.
    """
    if stop_proc is None:
        return
    try:
        _wait_for_playback("aplay", stop_proc)
    except RuntimeError as exc:
        print(f"whspr: could not play the stop sound: {exc}", file=sys.stderr)


def main(paste=False):
    """
    Try to acquire an exclusive non-blocking lock on `LOCK_PATH`.

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

            stop_transcribe_copy_and_notify(paste)
            return

        try:
            record_until_stop()
            return
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


if __name__ == "__main__":
    main()
