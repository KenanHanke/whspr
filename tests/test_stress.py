"""Chaos test: the server must survive a storm of mixed, partly hostile traffic."""

import random
import socket
import threading

import whspr.server as server

from conftest import wait_until


def test_server_survives_traffic_storm(harness):
    harness.spawn_and_wait_ready(job_delay=0.01)

    errors = []
    successes = []
    lock = threading.Lock()

    def good_client(i):
        try:
            path, content = harness.write_audio_stub(f"storm-{i}.txt", f"payload {i}")
            result = server.transcribe(path)
            with lock:
                if result == content:
                    successes.append(i)
                else:
                    errors.append(f"wrong result for {i}: {result!r}")
        except Exception as exc:
            with lock:
                errors.append(f"client {i}: {exc}")

    def rude_client(i):
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as conn:
                conn.settimeout(5.0)
                conn.connect(server.SOCKET_PATH)
                choice = i % 4
                if choice == 0:
                    pass  # connect and hang up instantly
                elif choice == 1:
                    conn.sendall(b"garbage without a newline")
                elif choice == 2:
                    conn.sendall(b'{"type": "transcribe"}\n')
                    conn.recv(4096)
                else:
                    conn.sendall(b'{"unknown": true}\n')
                    conn.recv(4096)
        except (OSError, ValueError):
            pass  # rude clients are allowed to fail; the server must not

    def duplicate_server(_):
        process = harness.spawn()
        process.wait(timeout=30)

    threads = []
    for i in range(30):
        threads.append(threading.Thread(target=good_client, args=(i,)))
    for i in range(30):
        threads.append(threading.Thread(target=rude_client, args=(i,)))
    for i in range(3):
        threads.append(threading.Thread(target=duplicate_server, args=(i,)))

    random.shuffle(threads)
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=120)
    assert not any(t.is_alive() for t in threads), "storm clients wedged"

    assert errors == []
    assert len(successes) == 30

    # Every job must have been strictly serialized despite the storm.
    events = [e.split()[0] for e in harness.events() if e.startswith("job-")]
    assert events == ["job-start", "job-end"] * 30

    # And the server still shuts down cleanly.
    server.stop()
    wait_until(lambda: not server.is_running(), message="server to stop after storm")
