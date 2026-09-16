# src/whspr/__init__.py

import threading

from . import server as _server

_MODEL = None
_MODEL_LOCK = threading.Lock()  # concurrent first calls must share one load

def transcribe(path):
    global _MODEL
    with _MODEL_LOCK:
        if _MODEL is None:
            _MODEL = _server.load_model()
    return _server.transcribe_helper(path, _MODEL) # type: ignore
