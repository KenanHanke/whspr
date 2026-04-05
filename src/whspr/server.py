import fcntl
import json
import os
import queue
import socket
import subprocess
import sys
import threading


LOCK_PATH = "/tmp/whspr-server.lock"
SOCKET_PATH = "/tmp/whspr-server.sock"


class _Job:
    def __init__(self, path, conn):
        self.path = path
        self.conn = conn


def _send_message(conn, payload):
    data = json.dumps(payload, ensure_ascii=False).encode("utf-8") + b"\n"
    conn.sendall(data)


def _recv_message(conn):
    chunks = []
    while True:
        chunk = conn.recv(4096)
        if not chunk:
            break
        chunks.append(chunk)
        if b"\n" in chunk:
            break

    if not chunks:
        raise ConnectionError("empty message")

    raw = b"".join(chunks).split(b"\n", 1)[0]
    return json.loads(raw.decode("utf-8"))


def _request(payload):
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
            conn.connect(SOCKET_PATH)
            _send_message(conn, payload)
            return _recv_message(conn)
    except FileNotFoundError as exc:
        raise RuntimeError("server is not running") from exc
    except ConnectionRefusedError as exc:
        raise RuntimeError("server is not running") from exc
    except OSError as exc:
        raise RuntimeError(f"server request failed: {exc}") from exc


def _wait_for_model(model_ready, stop_event):
    while True:
        if model_ready.wait(0.1):
            return
        if stop_event.is_set():
            raise RuntimeError("server is stopping")


def _reject_pending_jobs(jobs):
    while True:
        try:
            job = jobs.get_nowait()
        except queue.Empty:
            return

        if job is None:
            continue

        try:
            _send_message(job.conn, {"ok": False, "error": "server is stopping"})
        except OSError:
            pass
        finally:
            try:
                job.conn.close()
            except OSError:
                pass


def main():
    lock_file = open(LOCK_PATH, "a+")
    try:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            return

        lock_file.seek(0)
        lock_file.truncate()
        lock_file.write(f"{os.getpid()}\n")
        lock_file.flush()

        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass

        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as listener:
            listener.bind(SOCKET_PATH)
            listener.listen()
            listener.settimeout(0.1)

            stop_event = threading.Event()
            model_ready = threading.Event()
            jobs = queue.Queue()
            model_box = {}

            def model_loader():
                try:
                    model_box["model"] = load_model()
                except Exception as exc:
                    model_box["error"] = exc
                finally:
                    model_ready.set()

            def worker():
                while True:
                    job = jobs.get()
                    if job is None:
                        return

                    try:
                        _wait_for_model(model_ready, stop_event)
                        if "error" in model_box:
                            raise RuntimeError(f"model load failed: {model_box['error']}")

                        text = transcribe_helper(job.path, model_box["model"])
                        _send_message(job.conn, {"ok": True, "text": text})
                    except Exception as exc:
                        try:
                            _send_message(job.conn, {"ok": False, "error": str(exc)})
                        except OSError:
                            pass
                    finally:
                        try:
                            job.conn.close()
                        except OSError:
                            pass

            threading.Thread(target=model_loader, daemon=True).start()
            worker_thread = threading.Thread(target=worker, daemon=True)
            worker_thread.start()

            def handle_client(conn):
                try:
                    request = _recv_message(conn)
                    request_type = request.get("type")

                    if request_type == "transcribe":
                        path = request.get("path")
                        if not isinstance(path, str):
                            raise ValueError("missing or invalid path")
                        if stop_event.is_set():
                            raise RuntimeError("server is stopping")
                        jobs.put(_Job(path, conn))
                        return

                    if request_type == "stop":
                        _send_message(conn, {"ok": True})
                        conn.close()
                        stop_event.set()
                        _reject_pending_jobs(jobs)
                        jobs.put(None)
                        return

                    raise ValueError("unknown request type")
                except Exception as exc:
                    try:
                        _send_message(conn, {"ok": False, "error": str(exc)})
                    except OSError:
                        pass
                    try:
                        conn.close()
                    except OSError:
                        pass

            while not stop_event.is_set():
                try:
                    conn, _ = listener.accept()
                except socket.timeout:
                    continue
                except OSError:
                    break

                threading.Thread(target=handle_client, args=(conn,), daemon=True).start()

            stop_event.set()
            _reject_pending_jobs(jobs)
            jobs.put(None)
            worker_thread.join()
    finally:
        try:
            os.unlink(SOCKET_PATH)
        except FileNotFoundError:
            pass
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
        except OSError:
            pass
        lock_file.close()


def start():
    if is_running():
        return

    module_name = __spec__.name if __spec__ is not None else __name__
    with open(os.devnull, "rb") as devnull_in, open(os.devnull, "ab") as devnull_out:
        subprocess.Popen(
            [sys.executable, "-m", module_name],
            stdin=devnull_in,
            stdout=devnull_out,
            stderr=devnull_out,
            close_fds=True,
            start_new_session=True,
        )


def is_running():
    try:
        with open(LOCK_PATH, "a+") as lock_file:
            try:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            else:
                fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)
                return False
    except OSError:
        return False


def stop():
    if not is_running():
        return

    response = _request({"type": "stop"})
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "failed to stop server"))


def transcribe(path):
    response = _request({"type": "transcribe", "path": path})
    if not response.get("ok"):
        raise RuntimeError(response.get("error", "transcription failed"))
    return response["text"]


def load_model():
    from ._cuda_bootstrap import ensure_cuda_runtime_loaded

    ensure_cuda_runtime_loaded()

    from faster_whisper import WhisperModel

    model = WhisperModel(
        "turbo",
        device="cuda",
        compute_type="float16",
    )
    return model


def transcribe_helper(path, model):
    segments, info = model.transcribe(path)
    text = "".join(seg.text for seg in segments)
    return text.strip()


if __name__ == "__main__":
    main()