# src/whspr/_cuda_bootstrap.py
"""Make pip-installed NVIDIA libraries loadable before ctranslate2 needs them."""

from __future__ import annotations

import ctypes
import importlib.util
import os
from pathlib import Path
from typing import Iterable

_BOOTSTRAPPED = False
_PRELOADED_HANDLES = []  # keep CDLL handles alive


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
    Look for NVIDIA pip package lib directories. We keep this tolerant:
    some packages may be absent, depending on the environment.
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

        candidates.extend(_existing_dirs([pkg_dir / "lib"]))

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


def ensure_cuda_runtime_loaded() -> None:
    """
    Make NVIDIA pip-installed shared libraries discoverable before importing
    faster_whisper / ctranslate2.
    """
    global _BOOTSTRAPPED
    if _BOOTSTRAPPED:
        return

    lib_dirs = _candidate_lib_dirs()

    # Helpful for child processes and diagnostics, but don't rely on it alone.
    _prepend_env_path("LD_LIBRARY_PATH", lib_dirs)

    # Preload likely CUDA dependencies by absolute path before importing
    # ctranslate2.  Order matters a bit; load lower-level pieces first.
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

    _BOOTSTRAPPED = True


def cuda_support_libraries_present() -> bool:
    """Whether the cuBLAS and cuDNN versions ctranslate2 needs can be loaded.

    ctranslate2 loads these lazily (cuBLAS at model load, cuDNN at first
    inference), so a working NVIDIA driver alone is not proof that GPU
    inference can succeed.  Checks the pip-installed NVIDIA wheels first,
    then dlopen-probes for system installs — in both cases pinned to the
    versions every ctranslate2 4.x wheel links against (CUDA 12 cuBLAS;
    cuDNN 9 for >=4.5, cuDNN 8 for older), because a version-blind search
    would accept e.g. a CUDA 11 toolkit that ctranslate2 is guaranteed to
    reject.
    """
    lib_dirs = _candidate_lib_dirs()
    checks = [
        (["libcublas.so.12*"], ["libcublas.so.12"]),
        (["libcudnn.so.9*", "libcudnn.so.8*"], ["libcudnn.so.9", "libcudnn.so.8"]),
    ]
    for wheel_patterns, sonames in checks:
        if _glob_unique(lib_dirs, wheel_patterns):
            continue
        if any(_loadable(soname) for soname in sonames):
            continue
        return False
    return True


def _loadable(soname: str) -> bool:
    try:
        ctypes.CDLL(soname)
        return True
    except OSError:
        return False
