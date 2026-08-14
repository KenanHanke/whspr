# src/whspr/server.py
"""
Background transcription server for whspr.

The server owns the Whisper model so that repeated dictations do not pay the
model start-up cost every time.  It listens on a per-user Unix domain socket
and answers newline-delimited JSON requests, one request per connection:

    {"type": "transcribe", "path": "/abs/audio.wav"}  ->  {"ok": true, "text": "..."}
    {"type": "stop"}                                  ->  {"ok": true}

Failures are reported as {"ok": false, "error": "..."}; replies rejected only
because the server is shutting down additionally carry "code": "stopping",
which tells clients to retry against the server's replacement.

Lifecycle:

* `start()` spawns `python -m whspr.server` fully detached from the caller,
  so the server survives even if the caller is killed.
* A lock file guarantees a single server instance per user.  The socket is
  bound immediately at start-up while the model loads in a background thread,
  so clients can connect and queue work before the model is ready.
* Transcriptions run sequentially on a single worker thread.
* The server exits on a stop request, and automatically once it has gone
  unused for `IDLE_TIMEOUT` seconds.  `transcribe()` transparently (re)starts
  the server when needed, so the auto-shutdown is invisible to callers.
"""

import fcntl
import json
import os
import queue
import signal
import socket
import subprocess
import sys
import threading
import time

from ._paths import runtime_file

LOCK_PATH = runtime_file("whspr-server.lock")
SOCKET_PATH = runtime_file("whspr-server.sock")

# Fixed paths used by whspr before 1.1; old servers there never stop on their
# own, so start() asks any leftover one to shut down.
_LEGACY_SOCKET_PATH = "/tmp/whspr-server.sock"
_LEGACY_LOCK_PATH = "/tmp/whspr-server.lock"

IDLE_TIMEOUT = 5 * 60.0  # seconds without use after which the server exits

_ACCEPT_POLL_INTERVAL = 0.5    # how often the accept loop checks for idleness/stop
_MODEL_POLL_INTERVAL = 0.2     # how often a queued job re-checks for the model
_FAILED_MODEL_LINGER = 1.0     # how quickly a server whose model failed exits
_REQUEST_TIMEOUT = 30.0        # server: max seconds for one client socket operation
_DIAL_TIMEOUT = 5.0            # client: max seconds to connect and send one request
_SERVER_WAIT_TIMEOUT = 60.0    # client: max seconds of consecutive unreachability
_MAX_RECOVERIES = 3            # client: max mid-request server losses to retry past
_SPAWN_RETRY_INTERVAL = 2.0    # client: min seconds between server spawn attempts
_RESULT_TIMEOUT = 30 * 60.0    # client: max seconds to wait for a transcription result
_STOP_TIMEOUT = 10.0           # client: max seconds to wait for a stop acknowledgement
_MAX_MESSAGE_BYTES = 8 * 1024 * 1024  # cap on a single protocol message

# A job outliving every client's patience means a wedged native call (stuck
# CUDA init, hung download, ...).  Such a server can only be abandoned — it
# must vacate the lock so the next dictation gets a working process.
_WEDGED_JOB_TIMEOUT = _RESULT_TIMEOUT + 120.0


class _ServerUnavailableError(Exception):
    """The server socket is absent, refusing, or hung up before replying.

    `server_was_reached` distinguishes "could not reach a server at all"
    (connect/send failed) from "a server took the request but vanished before
    replying" (it crashed or shut down mid-request); the latter earns the
    client a fresh availability budget for the replacement server.
    """

    def __init__(self, message, server_was_reached=False):
        super().__init__(message)
        self.server_was_reached = server_was_reached


class _ServerStoppingError(RuntimeError):
    """Raised inside the server when work is interrupted by a shutdown."""


# Reply for requests that a shutting-down server cannot take; the "stopping"
# code tells clients to retry against the server's replacement.
_STOPPING_REPLY = {"ok": False, "error": "server is stopping", "code": "stopping"}


def _send_message(conn, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
    conn.sendall(data)


def _recv_message(conn):
    buffer = bytearray()
    while b"\n" not in buffer:
        if len(buffer) > _MAX_MESSAGE_BYTES:
            raise ValueError("message exceeds size limit")
        chunk = conn.recv(65536)
        if not chunk:
            raise ConnectionError("connection closed before a full message arrived")
        buffer.extend(chunk)
    line = bytes(buffer).split(b"\n", 1)[0]
    return json.loads(line.decode("utf-8"))


def _close_quietly(conn):
    try:
        conn.close()
    except OSError:
        pass


def _try_send(conn, payload):
    try:
        _send_message(conn, payload)
    except OSError:
        pass


def _unlink_quietly(path):
    try:
        os.unlink(path)
    except OSError:
        pass


class _Job:
    def __init__(self, path, conn):
        self.path = path
        self.conn = conn


class _ServerState:
    """Shared state between the accept loop, request handlers, and the worker.

    Work accounting invariant: the accept loop calls begin_work() exactly once
    per accepted connection, and whoever ultimately answers and closes that
    connection calls end_work() exactly once — the handler for connections it
    finishes itself, the worker for queued jobs, or the shutdown drain for
    jobs it rejects.  should_exit() relies on this balance to never stop the
    server while a request is queued or running.
    """

    def __init__(self):
        self.stop_event = threading.Event()
        # Written (only) by signal handlers, which must not touch Event's
        # non-reentrant lock: a signal landing inside stop_event.set() on the
        # main thread would self-deadlock re-entering set().
        self.stop_requested = False
        self.jobs = queue.Queue()
        self._model_ready = threading.Event()
        self._model = None
        self._model_error = None
        self._activity_lock = threading.Lock()
        self._inflight = 0
        self._last_activity = time.monotonic()
        self._job_started_at = None

    def note_job_started(self):
        with self._activity_lock:
            self._job_started_at = time.monotonic()

    def note_job_finished(self):
        with self._activity_lock:
            self._job_started_at = None

    def wedged(self):
        """True if the current job has been running impossibly long."""
        with self._activity_lock:
            started_at = self._job_started_at
        return started_at is not None and time.monotonic() - started_at > _WEDGED_JOB_TIMEOUT

    def begin_work(self):
        with self._activity_lock:
            self._inflight += 1
            self._last_activity = time.monotonic()

    def end_work(self):
        with self._activity_lock:
            self._inflight -= 1
            self._last_activity = time.monotonic()

    def set_model(self, model):
        self._model = model
        self._model_ready.set()

    def set_model_error(self, error):
        self._model_error = error
        self._model_ready.set()

    def model_failed(self):
        return self._model_ready.is_set() and self._model_error is not None

    def wait_for_model(self):
        """Block until the model is available; raise if it failed or we are stopping."""
        while not self._model_ready.wait(_MODEL_POLL_INTERVAL):
            if self.stop_event.is_set():
                raise _ServerStoppingError("server is stopping")
        if self._model_error is not None:
            raise RuntimeError(f"model load failed: {self._model_error}")
        return self._model

    def should_exit(self):
        """True once the server has no in-flight work and has idled long enough.

        A server whose model failed to load exits almost immediately instead,
        so the next dictation gets a fresh process and a fresh load attempt.
        """
        with self._activity_lock:
            if self._inflight > 0:
                return False
            idle_for = time.monotonic() - self._last_activity
        if self.model_failed():
            return idle_for > _FAILED_MODEL_LINGER
        return idle_for > IDLE_TIMEOUT


def _load_model_into(state):
    try:
        state.set_model(load_model())
    except Exception as exc:
        state.set_model_error(exc)


def _run_jobs(state):
    """Worker loop: complete queued transcriptions sequentially."""
    while True:
        job = state.jobs.get()
        if job is None:
            return
        state.note_job_started()
        try:
            model = state.wait_for_model()
            text = transcribe_helper(job.path, model)
            _try_send(job.conn, {"ok": True, "text": text})
        except _ServerStoppingError:
            _try_send(job.conn, _STOPPING_REPLY)
        except Exception as exc:
            _try_send(job.conn, {"ok": False, "error": str(exc)})
        finally:
            state.note_job_finished()
            _close_quietly(job.conn)
            state.end_work()


def _handle_connection(conn, state):
    """Parse one request; transcribe jobs hand their connection to the worker."""
    handed_off = False
    try:
        conn.settimeout(_REQUEST_TIMEOUT)
        try:
            request = _recv_message(conn)
        except ValueError:
            _try_send(conn, {"ok": False, "error": "malformed request"})
            return
        except OSError:
            return  # peer is silent, gone, or broken; no reply is possible
        if not isinstance(request, dict):
            _try_send(conn, {"ok": False, "error": "malformed request"})
            return
        request_type = request.get("type")

        if request_type == "transcribe":
            path = request.get("path")
            if not isinstance(path, str) or not path:
                _try_send(conn, {"ok": False, "error": "missing or invalid path"})
                return
            if state.stop_event.is_set():
                _try_send(conn, _STOPPING_REPLY)
                return
            state.jobs.put(_Job(path, conn))
            handed_off = True
            return

        if request_type == "stop":
            _try_send(conn, {"ok": True})
            state.stop_event.set()
            return

        _try_send(conn, {"ok": False, "error": "unknown request type"})
    finally:
        if not handed_off:
            _close_quietly(conn)
            state.end_work()


def _reject_pending_jobs(state):
    while True:
        try:
            job = state.jobs.get_nowait()
        except queue.Empty:
            return
        if job is None:
            continue
        _try_send(job.conn, _STOPPING_REPLY)
        _close_quietly(job.conn)
        state.end_work()


def _install_stop_signal_handlers(state):
    def request_stop(signum, frame):
        # Just a flag: an Event.set() here could self-deadlock if the signal
        # interrupted the main thread inside stop_event.set().
        state.stop_requested = True

    try:
        signal.signal(signal.SIGTERM, request_stop)
        signal.signal(signal.SIGINT, request_stop)
    except ValueError:
        pass  # not running in the main thread; rely on stop requests alone


def _serve(listener, release_lock):
    state = _ServerState()
    _install_stop_signal_handlers(state)

    threading.Thread(target=_load_model_into, args=(state,), daemon=True).start()
    worker = threading.Thread(target=_run_jobs, args=(state,), daemon=True)
    worker.start()

    try:
        while not (state.stop_event.is_set() or state.stop_requested):
            try:
                conn, _ = listener.accept()
            except socket.timeout:
                if state.wedged():
                    # A native call is stuck beyond any client's patience;
                    # only abandoning the process frees the lock (the flock
                    # dies with us, and the next server unlinks the socket).
                    os._exit(1)
                if state.should_exit():
                    break
                continue
            except OSError:
                break
            state.begin_work()
            try:
                threading.Thread(
                    target=_handle_connection, args=(conn, state), daemon=True
                ).start()
            except RuntimeError:
                _close_quietly(conn)
                state.end_work()
    except KeyboardInterrupt:
        pass
    finally:
        state.stop_event.set()
        # Stop accepting, remove the socket, then hand back the lock BEFORE
        # draining: late clients must fail over to a replacement server
        # immediately instead of waiting for this one's last job to finish.
        _close_quietly(listener)
        _unlink_quietly(SOCKET_PATH)
        release_lock()
        _reject_pending_jobs(state)
        state.jobs.put(None)
        worker.join(_WEDGED_JOB_TIMEOUT)
        if worker.is_alive():
            os._exit(1)  # wedged mid-job; the lock is already released
        _reject_pending_jobs(state)  # jobs that slipped in while shutting down


def main():
    """Run the server in the foreground; return an exit status once it stops.

    Returns immediately (status 0) if another server already holds the lock.
    """
    try:
        lock_file = open(LOCK_PATH, "a+")
    except OSError as exc:
        print(f"whspr server: cannot open lock file {LOCK_PATH}: {exc}", file=sys.stderr)
        return 1
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            return 0  # another server instance is already running

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()

        def release_lock():
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            except OSError:
                pass

        _unlink_quietly(SOCKET_PATH)
        listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        try:
            # Bind under a tight umask: connecting to a pathname socket needs
            # write permission, so this keeps other users out even in /tmp.
            old_umask = os.umask(0o177)
            try:
                listener.bind(SOCKET_PATH)
                listener.listen(16)
                listener.settimeout(_ACCEPT_POLL_INTERVAL)
            except OSError as exc:
                print(
                    f"whspr server: cannot bind socket {SOCKET_PATH}: {exc}",
                    file=sys.stderr,
                )
                _unlink_quietly(SOCKET_PATH)
                return 1
            finally:
                os.umask(old_umask)
            # _serve owns the socket from here: it unlinks SOCKET_PATH and
            # releases the lock at the START of its shutdown, so a successor
            # may already be bound to a fresh socket at this path by the time
            # _serve returns — never unlink again after this point.
            _serve(listener, release_lock)
            return 0
        finally:
            _close_quietly(listener)
    finally:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()


def start():
    """Start the server as a fully detached background process.

    A no-op if a server is already running; the server's own lock also makes
    concurrent duplicate starts harmless.
    """
    _stop_legacy_server()

    if is_running():
        return

    module_name = __spec__.name if __spec__ is not None else __name__
    with open(os.devnull, "rb") as devnull_in, open(os.devnull, "ab") as devnull_out:
        process = subprocess.Popen(
            [sys.executable, "-m", module_name],
            stdin=devnull_in,
            stdout=devnull_out,
            stderr=devnull_out,
            close_fds=True,
            start_new_session=True,
        )
    # Reap the child in the background so it cannot linger as a zombie inside
    # long-lived callers that use whspr as a library.
    threading.Thread(target=process.wait, daemon=True).start()


def _stop_legacy_server():
    """Best-effort shutdown of a server left over from whspr < 1.1.

    Old servers used fixed /tmp paths and ran (holding the loaded model in
    memory) until explicitly stopped, so after an upgrade one could linger
    until reboot.  Newer servers never use the legacy socket path.
    """
    if SOCKET_PATH == _LEGACY_SOCKET_PATH or not os.path.exists(_LEGACY_SOCKET_PATH):
        return
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.settimeout(2.0)
            conn.connect(_LEGACY_SOCKET_PATH)
            _send_message(conn, {"type": "stop"})
            _recv_message(conn)
    except (OSError, ValueError):
        pass  # no live legacy server; just clean up whatever is left
    _unlink_quietly(_LEGACY_SOCKET_PATH)
    _unlink_quietly(_LEGACY_LOCK_PATH)


def is_running():
    """Check whether a server instance currently holds the lock file."""
    try:
        with open(LOCK_PATH, "a+") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except OSError:
                return True
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
            return False
    except OSError:
        return False


def stop():
    """Ask a running server to shut down; a no-op if none is reachable."""
    try:
        response = _request({"type": "stop"}, _STOP_TIMEOUT)
    except _ServerUnavailableError:
        return
    if not (isinstance(response, dict) and response.get("ok")):
        error = response.get("error") if isinstance(response, dict) else None
        raise RuntimeError(str(error or "failed to stop the whspr server"))


def transcribe(path):
    """Transcribe an audio file via the server, starting the server if needed.

    Retries the connection until a server is reachable (it may still be
    starting up, or may have auto-stopped after idling), then waits for the
    result.  `_SERVER_WAIT_TIMEOUT` bounds *consecutive* unreachability, not
    total time: a server that takes the request and then goes away mid-job
    (crash or shutdown) grants a fresh budget for reaching its replacement,
    at most `_MAX_RECOVERIES` times.  The path is resolved client-side
    because the server process has its own working directory.
    """
    request = {"type": "transcribe", "path": os.path.abspath(os.fspath(path))}

    deadline = time.monotonic() + _SERVER_WAIT_TIMEOUT
    recoveries_left = _MAX_RECOVERIES
    last_spawn = None
    retry_delay = 0.05
    while True:
        cause = None
        server_was_reached = False
        try:
            response = _request(request, _RESULT_TIMEOUT)
        except _ServerUnavailableError as exc:
            cause = exc
            server_was_reached = exc.server_was_reached
        else:
            stopping = (
                isinstance(response, dict)
                and not response.get("ok")
                and response.get("code") == "stopping"
            )
            if not stopping:
                break
            server_was_reached = True

        # The server is unreachable or shutting down: (re)start it, then retry
        # against the replacement until the deadline runs out.
        now = time.monotonic()
        if server_was_reached and recoveries_left > 0:
            recoveries_left -= 1
            deadline = now + _SERVER_WAIT_TIMEOUT
        if now >= deadline:
            raise RuntimeError(
                "the whspr server did not become available within "
                f"{_SERVER_WAIT_TIMEOUT:.0f} seconds"
            ) from cause
        if last_spawn is None or now - last_spawn >= _SPAWN_RETRY_INTERVAL:
            start()
            last_spawn = now
        time.sleep(min(retry_delay, max(deadline - now, 0.0)))
        retry_delay = min(retry_delay * 2, 0.5)

    if not isinstance(response, dict):
        raise RuntimeError("received a malformed reply from the whspr server")
    if not response.get("ok"):
        raise RuntimeError(str(response.get("error") or "transcription failed"))
    text = response.get("text")
    if not isinstance(text, str):
        raise RuntimeError("received a malformed reply from the whspr server")
    return text


def _request(payload, reply_timeout):
    """Send a single request over a fresh connection and return the reply."""
    conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        conn.settimeout(_DIAL_TIMEOUT)
        try:
            conn.connect(SOCKET_PATH)
            _send_message(conn, payload)
        except (FileNotFoundError, ConnectionError, socket.timeout) as exc:
            raise _ServerUnavailableError(str(exc)) from exc

        conn.settimeout(reply_timeout)
        try:
            return _recv_message(conn)
        except ConnectionError as exc:
            raise _ServerUnavailableError(str(exc), server_was_reached=True) from exc
        except socket.timeout as exc:
            raise RuntimeError(
                "timed out waiting for the whspr server to reply"
            ) from exc
        except ValueError as exc:
            raise RuntimeError(
                "received a malformed reply from the whspr server"
            ) from exc
    finally:
        conn.close()


def load_model():
    """Load the best Whisper model for this machine.

    The device is detected *before* any model files are fetched, so CPU-only
    machines never download the much larger GPU model.  Already-downloaded
    model files are used without hitting the network.
    """
    from ._cuda_bootstrap import ensure_cuda_runtime_loaded

    ensure_cuda_runtime_loaded()

    if _cuda_is_usable():
        try:
            model = _load_whisper_model(
                "large-v3-turbo", device="cuda", compute_type="float16"
            )
            _warm_up(model)
            return model
        except Exception as exc:
            # Broken CUDA stack (driver, cuBLAS, cuDNN, VRAM): fall back to
            # the CPU, but say why, or the silent downgrade is undebuggable.
            print(
                f"whspr server: CUDA model unavailable ({exc}); using the CPU model",
                file=sys.stderr,
            )
    return _load_whisper_model("small", device="cpu", compute_type="int8")


def _cuda_is_usable():
    """True only if a CUDA device is visible AND its support libraries exist.

    A visible device merely means the NVIDIA driver is installed; ctranslate2
    loads cuBLAS/cuDNN lazily, so without this check a driver-only machine
    (whspr installed without the [gpu] extra) would download the large GPU
    model just to fail and fall back to the CPU.
    """
    try:
        import ctranslate2

        from ._cuda_bootstrap import cuda_support_libraries_present

        return ctranslate2.get_cuda_device_count() > 0 and cuda_support_libraries_present()
    except Exception:
        return False


def _load_whisper_model(name, **kwargs):
    from faster_whisper import WhisperModel

    try:
        # Fast path: reuse already-downloaded model files without network I/O.
        return WhisperModel(name, local_files_only=True, **kwargs)
    except Exception:
        return WhisperModel(name, **kwargs)


def _warm_up(model):
    """Transcribe a tiny bundled sound so CUDA failures surface at load time.

    cuDNN and cuBLAS are loaded lazily during the first transcription, so a
    CUDA model can construct fine yet be unable to transcribe anything.
    """
    from importlib.resources import as_file, files

    sample = files("whspr").joinpath("data/sounds/finished.wav")
    with as_file(sample) as sample_path:
        transcribe_helper(str(sample_path), model)


def transcribe_helper(path, model):
    segments, info = model.transcribe(path)
    text = "".join(seg.text for seg in segments)

    # remove common hallucinations
    hallucinations = [
        "Thank you.",
        "Hello, I know I'll be right back.",
    ]
    for hallucination in hallucinations:
        text = text.replace(" " + hallucination, "")
        text = text.replace(hallucination + " ", "")
        text = text.replace(hallucination, "")

    text = text.strip()
    return text


if __name__ == "__main__":
    sys.exit(main())
