"""Resolve ffmpeg and ffprobe executable paths."""

import os
import shutil
from pathlib import Path
from typing import Optional


def _resolve_from_env(var_names: list[str]) -> Optional[str]:
    for var in var_names:
        value = os.getenv(var)
        if value and Path(value).exists():
            return value
    return None


def _resolve_bundled_executable(executable_name: str) -> Optional[str]:
    backend_root = Path(__file__).resolve().parents[1]
    project_root = backend_root.parent
    search_roots = [
        project_root,
        project_root.parent,
        project_root.parent / "work",
        project_root.parent / "work" / "AutoClip-Deploy" / "work",
    ]

    for root in search_roots:
        if not root.exists():
            continue
        direct_matches = [
            root / "ffmpeg" / "bin" / executable_name,
            root / "ffmpeg" / "ffmpeg-8.1.2-essentials_build" / "bin" / executable_name,
        ]
        for candidate in direct_matches:
            if candidate.exists():
                return str(candidate)

        for candidate in root.glob(f"**/{executable_name}"):
            if candidate.exists():
                return str(candidate)
    return None


def _resolve_executable(env_vars: list[str], command: str) -> str:
    env_path = _resolve_from_env(env_vars)
    if env_path:
        return env_path

    which = shutil.which(command)
    if which:
        return which

    executable_name = f"{command}.exe" if os.name == "nt" else command
    bundled = _resolve_bundled_executable(executable_name)
    if bundled:
        return bundled

    return command


def get_ffmpeg_path() -> str:
    return _resolve_executable(["AUTOCLIP_FFMPEG_PATH", "FFMPEG_PATH"], "ffmpeg")


def get_ffprobe_path() -> str:
    return _resolve_executable(["AUTOCLIP_FFPROBE_PATH", "FFPROBE_PATH"], "ffprobe")
