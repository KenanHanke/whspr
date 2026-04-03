from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
import sys
from typing import Iterable

_BOOTSTRAPPED = False
_DLL_DIR_HANDLES = []   # Windows: keep add_dll_directory handles alive
_PRELOADED_HANDLES = [] # Linux: keep CDLL handles alive


def _package_dir(module_name: str) -> Path | None:
    """
    Return the installed package directory for a module/package, using importlib
    instead of relying on module.__file__.
    """
    spec = importlib.util.find_spec(module_name)
    if spec is None:
        return None

    # Package / namespace-package case
    if spec.submodule_search_locations:
        for location in spec.submodule_search_locations:
            p = Path(location)
            if p.exists():
                return p

    # Regular module case
    if spec.origin:
        p = Path(spec.origin).resolve().parent
        if p.exists():
            return p

    return None


def _existing_dirs(paths: Iterable[Path]) -> list[Path]:
    out: list[Path] = []
    for p in paths:
        if p.exists() and p.is_dir():
            out.append(p)
    return out


def _candidate_lib_dirs() -> list[Path]:
    """
    Look for NVIDIA pip package lib/bin directories. We keep this tolerant:
    some packages may be absent, depending on platform and environment.
    """
    candidates: list[Path] = []

    for mod in (
        "nvidia.cuda_runtime",
        "nvidia.cuda_nvrtc",
        "nvidia.cublas",
        "nvidia.cudnn",
    ):
        pkg_dir = _package_dir(mod)
        if not pkg_dir:
            continue

        # Linux wheels usually store .so files in lib/
        # Windows wheels commonly use bin/
        candidates.extend(_existing_dirs([pkg_dir / "lib", pkg_dir / "bin"]))

    # Deduplicate while preserving order
    seen: set[Path] = set()
    uniq: list[Path] = []
    for d in candidates:
        r = d.resolve()
        if r not in seen:
            seen.add(r)
            uniq.append(r)
    return uniq


def _prepend_env_path(var_name: str, new_dirs: list[Path]) -> None:
    current = os.environ.get(var_name, "")
    current_parts = [p for p in current.split(os.pathsep) if p]

    merged: list[str] = []
    seen: set[str] = set()

    for p in [str(d) for d in new_dirs] + current_parts:
        if p not in seen:
            seen.add(p)
            merged.append(p)

    os.environ[var_name] = os.pathsep.join(merged)


def _glob_unique(lib_dirs: list[Path], patterns: list[str]) -> list[Path]:
    found: list[Path] = []
    seen: set[Path] = set()

    for lib_dir in lib_dirs:
        for pattern in patterns:
            for path in sorted(lib_dir.glob(pattern)):
                real = path.resolve()
                if real.is_file() and real not in seen:
                    seen.add(real)
                    found.append(real)

    return found


def _bootstrap_linux(lib_dirs: list[Path]) -> None:
    # Helpful for child processes and diagnostics, but don't rely on it alone.
    _prepend_env_path("LD_LIBRARY_PATH", lib_dirs)

    # Preload likely CUDA dependencies by absolute path before importing ctranslate2.
    # Order matters a bit; load lower-level/common pieces first.
    load_order = [
        "libcudart.so*",
        "libnvrtc.so*",
        "libcublasLt.so*",
        "libcublas.so*",
        "libcudnn*.so*",
    ]

    libs = _glob_unique(lib_dirs, load_order)
    mode = getattr(os, "RTLD_NOW", 0) | getattr(os, "RTLD_GLOBAL", 0)

    for lib in libs:
        try:
            handle = ctypes.CDLL(str(lib), mode=mode)
            _PRELOADED_HANDLES.append(handle)
        except OSError:
            # Keep going; the final import will produce the actionable failure.
            pass


def _bootstrap_windows(lib_dirs: list[Path]) -> None:
    if not hasattr(os, "add_dll_directory"):
        return

    for lib_dir in lib_dirs:
        try:
            h = os.add_dll_directory(str(lib_dir)) # type: ignore
            _DLL_DIR_HANDLES.append(h)
        except OSError:
            pass


def ensure_cuda_runtime_loaded() -> None:
    """
    Make NVIDIA pip-installed shared libraries discoverable before importing
    faster_whisper / ctranslate2.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    lib_dirs = _candidate_lib_dirs()

    if os.name == "nt":
        _bootstrap_windows(lib_dirs)
    elif sys.platform.startswith("linux"):
        _bootstrap_linux(lib_dirs)

    _BOOTSTRAPPED = True


def diagnostic_report() -> str:
    lib_dirs = _candidate_lib_dirs()
    lines = [
        f"platform={sys.platform}",
        f"lib_dirs={','.join(str(d) for d in lib_dirs) if lib_dirs else '<none>'}",
        f"LD_LIBRARY_PATH={os.environ.get('LD_LIBRARY_PATH', '')}",
    ]
    return "\n".join(lines)