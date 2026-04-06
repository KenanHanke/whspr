# src/whspr/__init__.py

from . import server as _server

_MODEL = None

def transcribe(path):
    global _MODEL
    if _MODEL is None:
        _MODEL = _server.load_model()
    return _server.transcribe_helper(path, _MODEL) # type: ignore
