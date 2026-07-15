"""Utilities for removing duplicate clip candidates."""

import logging
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


def _time_to_seconds(value: Any) -> Optional[float]:
    if value is None:
        return None
    text = str(value).strip().replace(",", ".")
    if not text:
        return None
    parts = text.split(":")
    try:
        if len(parts) == 3:
            return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
        if len(parts) == 2:
            return int(parts[0]) * 60 + float(parts[1])
        return float(parts[0])
    except Exception:
        return None


def _clip_range(clip: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    start = _time_to_seconds(clip.get("start_time"))
    end = _time_to_seconds(clip.get("end_time"))
    if start is None or end is None or end <= start:
        return None
    return start, end


def _seconds_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    milliseconds = int(round((seconds - int(seconds)) * 1000))
    whole = int(seconds)
    if milliseconds >= 1000:
        whole += 1
        milliseconds -= 1000
    hours = whole // 3600
    minutes = (whole % 3600) // 60
    secs = whole % 60
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"


def _range_key(clip: Dict[str, Any]) -> Optional[Tuple[int, int]]:
    clip_range = _clip_range(clip)
    if clip_range is None:
        return None
    start, end = clip_range
    return round(start), round(end)


def _quality_key(clip: Dict[str, Any]) -> Tuple[float, float, int, int]:
    score = float(clip.get("final_score", clip.get("score", 0)) or 0)
    clip_range = _clip_range(clip)
    duration = (clip_range[1] - clip_range[0]) if clip_range else 0
    text_len = len(str(clip.get("content") or "")) + len(str(clip.get("recommend_reason") or ""))
    has_title = 1 if (clip.get("generated_title") or clip.get("title")) else 0
    return score, duration, text_len, has_title


def dedupe_clips_by_time(clips: List[Dict[str, Any]], source: str = "clips") -> List[Dict[str, Any]]:
    """Keep one clip for each rounded start/end range.

    The LLM can occasionally return duplicate candidates with different ids but
    identical time ranges. Keeping the highest-quality candidate here prevents
    duplicate exports, collections, and database rows downstream.
    """
    if not clips:
        return []

    best_by_range: Dict[Tuple[int, int], Dict[str, Any]] = {}
    passthrough: List[Dict[str, Any]] = []
    duplicates = 0

    for clip in clips:
        key = _range_key(clip)
        if key is None:
            passthrough.append(clip)
            continue

        existing = best_by_range.get(key)
        if existing is None:
            best_by_range[key] = clip
            continue

        duplicates += 1
        if _quality_key(clip) > _quality_key(existing):
            best_by_range[key] = clip

    deduped = passthrough + list(best_by_range.values())
    deduped.sort(key=lambda item: (_time_to_seconds(item.get("start_time")) or 0, _time_to_seconds(item.get("end_time")) or 0))

    if duplicates:
        logger.info("Removed %s duplicate clip candidates from %s", duplicates, source)
    return deduped


def _parse_advised_ranges(text: str, parent_start: float, parent_end: float) -> List[Tuple[float, float]]:
    if not text:
        return []
    normalized = (
        str(text)
        .replace("—", "-")
        .replace("–", "-")
        .replace("至", "-")
        .replace("到", "-")
        .replace("～", "-")
        .replace("~", "-")
    )
    pattern = re.compile(r"(\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d{1,3})?)\s*-\s*(\d{1,2}:\d{2}(?::\d{2})?(?:[,.]\d{1,3})?)")
    ranges: List[Tuple[float, float]] = []
    for start_text, end_text in pattern.findall(normalized):
        start = _time_to_seconds(start_text)
        end = _time_to_seconds(end_text)
        if start is None or end is None or end <= start:
            continue
        if start < parent_start - 5 or end > parent_end + 5:
            continue
        if end - start < 8:
            continue
        ranges.append((max(parent_start, start), min(parent_end, end)))

    unique: List[Tuple[float, float]] = []
    for item in ranges:
        key = (round(item[0]), round(item[1]))
        if key not in {(round(existing[0]), round(existing[1])) for existing in unique}:
            unique.append(item)
    return unique


def expand_long_clips_from_advice(
    clips: List[Dict[str, Any]],
    source: str = "clips",
    min_duration_seconds: float = 120.0,
) -> List[Dict[str, Any]]:
    """Split long clips when the model provided concrete sub-ranges.

    Product livestreams often contain a long product explanation that is useful
    only after being split into selling-point modules. The LLM already writes
    those suggested time ranges into duration_advice; this turns them into real
    clips instead of dropping that signal on the floor.
    """
    if not clips:
        return []

    split_parent_keys = set()
    for clip in clips:
        clip_range = _clip_range(clip)
        duration = clip.get("duration_seconds", clip.get("duration"))
        try:
            duration_value = float(duration)
        except Exception:
            duration_value = (clip_range[1] - clip_range[0]) if clip_range else 0
        if not clip_range or duration_value < min_duration_seconds:
            continue
        ranges = _parse_advised_ranges(str(clip.get("duration_advice") or ""), clip_range[0], clip_range[1])
        if len(ranges) >= 2:
            split_parent_keys.add((round(clip_range[0]), round(clip_range[1])))

    expanded: List[Dict[str, Any]] = []
    split_count = 0

    for clip in clips:
        clip_range = _clip_range(clip)
        duration = clip.get("duration_seconds", clip.get("duration"))
        try:
            duration_value = float(duration)
        except Exception:
            duration_value = (clip_range[1] - clip_range[0]) if clip_range else 0

        if not clip_range or duration_value < min_duration_seconds:
            expanded.append(clip)
            continue

        parent_key = (round(clip_range[0]), round(clip_range[1]))
        advice = str(clip.get("duration_advice") or "")
        ranges = _parse_advised_ranges(advice, clip_range[0], clip_range[1])
        if len(ranges) < 2:
            if parent_key in split_parent_keys:
                logger.info(
                    "Skipping long duplicate parent clip because another duplicate supplied split ranges: %s",
                    clip.get("id"),
                )
                continue
            expanded.append(clip)
            continue

        split_count += len(ranges)
        for index, (start, end) in enumerate(ranges, start=1):
            child = deepcopy(clip)
            parent_id = str(clip.get("id", "clip"))
            child["id"] = f"{parent_id}-{index}"
            child["parent_clip_id"] = parent_id
            child["split_index"] = index
            child["split_count"] = len(ranges)
            child["start_time"] = _seconds_to_srt_time(start)
            child["end_time"] = _seconds_to_srt_time(end)
            child["duration_seconds"] = round(end - start, 2)
            child["duration"] = round(end - start, 2)
            child["cut_reason"] = f"根据长段拆分建议自动拆出的第 {index}/{len(ranges)} 个产品介绍切片。{clip.get('cut_reason', '')}"
            child["duration_advice"] = "已按模型建议从长产品段拆分为短切片"
            expanded.append(child)

    if split_count:
        logger.info("Expanded %s advised sub-clips from long clips in %s", split_count, source)
    return expanded
