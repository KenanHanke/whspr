"""In-process unit tests for whspr.server internals."""

import json
import os
import socket
import threading
import types

import pytest

import whspr.server as server


# --- wire protocol -----------------------------------------------------------


def test_recv_message_handles_chunked_delivery():
    a, b = socket.socketpair()
    try:
        payload = {"type": "transcribe", "path": "/tmp/x.wav"}
        raw = json.dumps(payload).encode("utf-8") + b"\n"

        def drip_feed():
            for i in range(len(raw)):
                a.sendall(raw[i : i + 1])

        thread = threading.Thread(target=drip_feed)
        thread.start()
        assert server._recv_message(b) == payload
        thread.join()
    finally:
        a.close()
        b.close()


def test_recv_message_raises_on_eof_before_newline():
    a, b = socket.socketpair()
    try:
        a.sendall(b'{"partial": ')
        a.close()
        with pytest.raises(ConnectionError):
            server._recv_message(b)
    finally:
        b.close()


def test_recv_message_rejects_oversized_input(monkeypatch):
    monkeypatch.setattr(server, "_MAX_MESSAGE_BYTES", 200_000)
    a, b = socket.socketpair()
    try:
        blob = b"A" * 400_000

        def flood():
            try:
                a.sendall(blob)
            except OSError:
                pass

        thread = threading.Thread(target=flood)
        thread.start()
        with pytest.raises(ValueError):
            server._recv_message(b)
        b.close()
        thread.join()
    finally:
        a.close()


def test_send_message_is_single_line_json():
    a, b = socket.socketpair()
    try:
        server._send_message(a, {"ok": True, "text": "line1\nline2 ümlaut"})
        data = b.recv(65536)
        assert data.endswith(b"\n")
        assert data.count(b"\n") == 1
        assert json.loads(data.decode("utf-8")) == {"ok": True, "text": "line1\nline2 ümlaut"}
    finally:
        a.close()
        b.close()


# --- runtime paths -----------------------------------------------------------


def test_runtime_paths_prefer_xdg_runtime_dir(tmp_path, monkeypatch):
    from whspr._paths import runtime_file

    monkeypatch.setenv("XDG_RUNTIME_DIR", str(tmp_path))
    assert runtime_file("whspr-server.sock") == str(tmp_path / "whspr-server.sock")


def test_runtime_paths_fall_back_to_tmp_with_uid(monkeypatch):
    from whspr._paths import runtime_file

    monkeypatch.delenv("XDG_RUNTIME_DIR", raising=False)
    expected = f"/tmp/whspr-server-{os.getuid()}.sock"
    assert runtime_file("whspr-server.sock") == expected


def test_runtime_paths_ignore_nonexistent_xdg_dir(monkeypatch):
    from whspr._paths import runtime_file

    monkeypatch.setenv("XDG_RUNTIME_DIR", "/no/such/directory/for/sure")
    expected = f"/tmp/whspr-server-{os.getuid()}.lock"
    assert runtime_file("whspr-server.lock") == expected


def test_client_paths_are_per_user():
    import whspr.client as client

    for path in (client._new_recording_path(), client.SOCKET_PATH, client.LOCK_PATH):
        runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
        if runtime_dir and os.path.isdir(runtime_dir):
            assert path.startswith(runtime_dir)
        else:
            assert f"-{os.getuid()}" in os.path.basename(path)


def test_recording_paths_are_unique_per_recorder_process():
    import whspr.client as client

    path = client._new_recording_path()
    assert str(os.getpid()) in os.path.basename(path)


# --- model selection ---------------------------------------------------------


class RecordingLoader:
    def __init__(self, fail_on=()):
        self.calls = []
        self.fail_on = set(fail_on)

    def __call__(self, name, **kwargs):
        self.calls.append((name, kwargs))
        if name in self.fail_on:
            raise RuntimeError(f"cannot load {name}")
        return types.SimpleNamespace(name=name, kwargs=kwargs)


@pytest.fixture
def no_cuda_bootstrap(monkeypatch):
    import whspr._cuda_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "ensure_cuda_runtime_loaded", lambda: None)


def test_load_model_uses_multilingual_small_on_cpu(monkeypatch, no_cuda_bootstrap):
    loader = RecordingLoader()
    monkeypatch.setattr(server, "_cuda_is_usable", lambda: False)
    monkeypatch.setattr(server, "_load_whisper_model", loader)

    model = server.load_model()

    assert loader.calls == [("small", {"device": "cpu", "compute_type": "int8"})]
    assert model.name == "small"  # multilingual "small", NOT "small.en"


def test_load_model_uses_turbo_on_gpu(monkeypatch, no_cuda_bootstrap):
    loader = RecordingLoader()
    warmed = []
    monkeypatch.setattr(server, "_cuda_is_usable", lambda: True)
    monkeypatch.setattr(server, "_load_whisper_model", loader)
    monkeypatch.setattr(server, "_warm_up", warmed.append)

    model = server.load_model()

    assert loader.calls == [
        ("large-v3-turbo", {"device": "cuda", "compute_type": "float16"})
    ]
    assert model.name == "large-v3-turbo"
    assert warmed == [model]  # the CUDA stack was actually exercised


def test_load_model_falls_back_to_cpu_when_cuda_model_fails(
    monkeypatch, no_cuda_bootstrap
):
    loader = RecordingLoader(fail_on={"large-v3-turbo"})
    monkeypatch.setattr(server, "_cuda_is_usable", lambda: True)
    monkeypatch.setattr(server, "_load_whisper_model", loader)

    model = server.load_model()

    assert [call[0] for call in loader.calls] == ["large-v3-turbo", "small"]
    assert model.name == "small"


def test_load_model_falls_back_to_cpu_when_warm_up_fails(
    monkeypatch, no_cuda_bootstrap
):
    loader = RecordingLoader()
    monkeypatch.setattr(server, "_cuda_is_usable", lambda: True)
    monkeypatch.setattr(server, "_load_whisper_model", loader)

    def broken_warm_up(model):
        raise RuntimeError("cuDNN missing")

    monkeypatch.setattr(server, "_warm_up", broken_warm_up)

    model = server.load_model()

    assert [call[0] for call in loader.calls] == ["large-v3-turbo", "small"]
    assert model.name == "small"


import importlib.util


@pytest.mark.skipif(
    importlib.util.find_spec("nvidia") is None,
    reason="[gpu] extras not installed",
)
def test_cuda_support_libraries_found_with_gpu_extras():
    import whspr._cuda_bootstrap as bootstrap

    assert bootstrap.cuda_support_libraries_present() is True


def test_cuda_support_libraries_absent_without_wheels(monkeypatch):
    import whspr._cuda_bootstrap as bootstrap

    # With no wheels and nothing loadable: unusable.
    monkeypatch.setattr(bootstrap, "_candidate_lib_dirs", lambda: [])
    monkeypatch.setattr(bootstrap, "_loadable", lambda soname: False)
    assert bootstrap.cuda_support_libraries_present() is False


def test_cuda_probe_rejects_incompatible_library_versions(monkeypatch):
    """A CUDA 11-era toolkit must not count: ctranslate2 4.x needs cuBLAS 12."""
    import whspr._cuda_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "_candidate_lib_dirs", lambda: [])
    cuda11_libs = {"libcublas.so.11", "libcudnn.so.8"}
    monkeypatch.setattr(bootstrap, "_loadable", lambda soname: soname in cuda11_libs)
    # cuDNN 8 alone is acceptable (older ctranslate2 4.x), but cuBLAS 11 is
    # never linkable by any ctranslate2 4.x wheel, so the probe must fail.
    assert bootstrap.cuda_support_libraries_present() is False


def test_cuda_unusable_without_support_libraries(monkeypatch):
    """A visible GPU driver alone must not trigger the big turbo download."""
    import whspr._cuda_bootstrap as bootstrap

    monkeypatch.setattr(bootstrap, "cuda_support_libraries_present", lambda: False)
    assert server._cuda_is_usable() is False


def test_load_whisper_model_prefers_local_files(monkeypatch):
    import faster_whisper

    calls = []

    class FakeWhisperModel:
        def __init__(self, name, local_files_only=False, **kwargs):
            calls.append(local_files_only)
            if local_files_only:
                raise RuntimeError("not cached")

    monkeypatch.setattr(faster_whisper, "WhisperModel", FakeWhisperModel)
    server._load_whisper_model("small", device="cpu", compute_type="int8")
    assert calls == [True, False]  # cache first, then the network


# --- transcribe_helper -------------------------------------------------------


class FakeSegmentModel:
    def __init__(self, texts):
        self.texts = texts

    def transcribe(self, path):
        segments = iter(types.SimpleNamespace(text=t) for t in self.texts)
        return segments, None


def test_transcribe_helper_joins_and_strips_segments():
    model = FakeSegmentModel([" Hello", " world."])
    assert server.transcribe_helper("ignored", model) == "Hello world."


def test_transcribe_helper_removes_known_hallucinations():
    model = FakeSegmentModel([" Real text.", " Thank you."])
    assert server.transcribe_helper("ignored", model) == "Real text."


# --- server state ------------------------------------------------------------


def test_should_exit_respects_inflight_work(monkeypatch):
    monkeypatch.setattr(server, "IDLE_TIMEOUT", 0.0)
    state = server._ServerState()
    state.begin_work()
    assert not state.should_exit()
    state.end_work()
    assert state.should_exit()


def test_should_exit_quickly_after_model_failure(monkeypatch):
    monkeypatch.setattr(server, "IDLE_TIMEOUT", 3600.0)
    monkeypatch.setattr(server, "_FAILED_MODEL_LINGER", 0.0)
    state = server._ServerState()
    assert not state.should_exit()  # huge idle timeout, healthy model
    state.set_model_error(RuntimeError("boom"))
    assert state.should_exit()


def test_wait_for_model_raises_when_stopping():
    state = server._ServerState()
    state.stop_event.set()
    with pytest.raises(RuntimeError, match="stopping"):
        state.wait_for_model()


def test_wait_for_model_reports_load_failure():
    state = server._ServerState()
    state.set_model_error(RuntimeError("no weights"))
    with pytest.raises(RuntimeError, match="model load failed"):
        state.wait_for_model()
