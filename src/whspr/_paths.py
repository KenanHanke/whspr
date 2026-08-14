# src/whspr/_paths.py
"""Per-user runtime file locations shared by the whspr client and server."""

import os


def runtime_file(name):
    """Return a per-user path for a whspr runtime file (socket, lock, ...).

    Prefers XDG_RUNTIME_DIR (private, per-user, wiped on logout); falls back
    to /tmp with a uid suffix so concurrent users cannot collide there either.
    """
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and os.path.isdir(runtime_dir):
        return os.path.join(runtime_dir, name)
    stem, ext = os.path.splitext(name)
    return os.path.join("/tmp", f"{stem}-{os.getuid()}{ext}")
