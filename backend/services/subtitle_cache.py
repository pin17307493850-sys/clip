import hashlib
import json
import shutil
from pathlib import Path
from typing import Optional

from backend.core.path_utils import get_cache_directory


SAMPLE_SIZE = 4 * 1024 * 1024


def _subtitle_cache_dir() -> Path:
    cache_dir = get_cache_directory() / "subtitles"
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir


def compute_video_fingerprint(video_path: Path) -> str:
    """Compute a fast, stable-enough fingerprint for subtitle reuse."""
    video_path = Path(video_path)
    stat = video_path.stat()
    digest = hashlib.sha256()
    digest.update(str(stat.st_size).encode("utf-8"))

    with video_path.open("rb") as f:
        digest.update(f.read(SAMPLE_SIZE))
        if stat.st_size > SAMPLE_SIZE:
            f.seek(max(0, stat.st_size - SAMPLE_SIZE))
            digest.update(f.read(SAMPLE_SIZE))

    return digest.hexdigest()


def get_cached_subtitle(video_path: Path) -> Optional[Path]:
    fingerprint = compute_video_fingerprint(Path(video_path))
    srt_path = _subtitle_cache_dir() / f"{fingerprint}.srt"
    if srt_path.exists() and srt_path.stat().st_size > 0:
        return srt_path
    return None


def copy_cached_subtitle(video_path: Path, target_path: Path) -> Optional[Path]:
    cached = get_cached_subtitle(video_path)
    if not cached:
        return None
    target_path = Path(target_path)
    target_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(cached, target_path)
    return target_path


def cache_subtitle(video_path: Path, subtitle_path: Path, model_name: str = "") -> Optional[Path]:
    subtitle_path = Path(subtitle_path)
    if not subtitle_path.exists() or subtitle_path.stat().st_size == 0:
        return None

    fingerprint = compute_video_fingerprint(Path(video_path))
    cache_dir = _subtitle_cache_dir()
    cached_srt = cache_dir / f"{fingerprint}.srt"
    shutil.copy2(subtitle_path, cached_srt)

    metadata = {
        "fingerprint": fingerprint,
        "source_video": str(video_path),
        "source_subtitle": str(subtitle_path),
        "model_name": model_name,
    }
    (cache_dir / f"{fingerprint}.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return cached_srt


def delete_cached_subtitle(video_path: Path) -> bool:
    """Delete the reusable subtitle cache for a video fingerprint."""
    try:
        fingerprint = compute_video_fingerprint(Path(video_path))
    except Exception:
        return False

    deleted = False
    cache_dir = _subtitle_cache_dir()
    for suffix in (".srt", ".json"):
        cache_file = cache_dir / f"{fingerprint}{suffix}"
        if cache_file.exists():
            cache_file.unlink()
            deleted = True
    return deleted
