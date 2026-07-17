"""Global subtitle audit for product livestream clip boundaries."""

import json
import logging
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .product_identity import (
    canonical_product_name,
    is_suite_product,
    parent_product_name,
    product_family_name,
)

logger = logging.getLogger(__name__)


CONTINUATION_CUES = (
    "\u8fd8\u6709",
    "\u53e6\u5916\u4e00\u6b3e",
    "\u518d\u6765\u4e00\u6b3e",
    "\u4e0b\u4e00\u6b3e",
    "\u6700\u540e\u4e00\u4e2a",
    "\u6700\u540e\u4e00\u6b3e",
    "\u7b2c\u4e00\u6b3e",
    "\u7b2c\u4e8c\u6b3e",
    "\u7b2c\u4e09\u6b3e",
    "\u7b2c\u56db\u6b3e",
    "\u7b2c\u4e94\u6b3e",
    "\u5206\u522b\u662f",
)

PRICE_BOUNDARIES = (
    "\u4ef7\u683c",
    "\u539f\u4ef7",
    "\u73b0\u4ef7",
    "\u5230\u624b",
    "\u4e0b\u5355",
    "\u62cd\u4e0b",
    "\u94fe\u63a5",
    "\u591a\u5c11\u94b1",
    "\u5757\u94b1",
    "\u5143",
)

HARD_STOPS = (
    "\u62bd\u5956",
    "\u7269\u6d41",
    "\u5173\u6ce8\u4e3b\u64ad",
    "\u4e0a\u8f66\u94fe\u63a5",
)

SUITE_START_CUES = (
    "\u4ecb\u7ecd\u4e00\u4e0b",
    "\u7ee7\u7eed\u4ecb\u7ecd",
    "\u518d\u7ee7\u7eed",
    "\u63a5\u4e0b\u6765\u4ecb\u7ecd",
)

SUITE_START_PREFIXES = (
    "\u7ee7\u7eed",
    "\u518d\u7ee7\u7eed",
    "\u63a5\u4e0b\u6765",
    "\u4e0b\u9762",
)

PACKAGING_WORDS = (
    "\u5305\u88c5",
    "\u8bbe\u8ba1",
    "\u989c\u8272",
    "\u914d\u8272",
    "\u63d2\u753b",
    "\u5de5\u827a",
)

FLAVOR_WORDS = (
    "\u53e3\u5473",
    "\u5473\u9053",
    "\u9999\u6c14",
    "\u5165\u53e3",
    "\u9178\u751c",
    "\u679c\u6c41",
    "\u6ce1\u6c34",
    "\u871c\u6843",
    "\u8461\u8404",
    "\u679c\u8336",
    "\u82b1\u8336",
    "\u4e4c\u9f99",
)


def _seconds(value: Any, text_processor: Any) -> Optional[float]:
    try:
        return float(text_processor.time_to_seconds(str(value).replace(",", ".")))
    except Exception:
        return None


def _srt_time(seconds: float) -> str:
    total_ms = max(0, int(round(float(seconds) * 1000)))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{millis:03d}"


def _range(clip: Dict[str, Any], text_processor: Any) -> Optional[Tuple[float, float]]:
    start = _seconds(clip.get("start_time"), text_processor)
    end = _seconds(clip.get("end_time"), text_processor)
    if start is None or end is None or end <= start:
        return None
    return start, end


def _load_global_subtitles(srt_chunks_dir: Path, text_processor: Any) -> List[Dict[str, Any]]:
    subtitles: List[Dict[str, Any]] = []
    seen = set()
    for path in sorted(srt_chunks_dir.glob("chunk_*.json")):
        try:
            raw_items = json.loads(path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Cannot load subtitle checkpoint %s: %s", path, exc)
            continue
        for raw in raw_items:
            start = _seconds(raw.get("start_time"), text_processor)
            end = _seconds(raw.get("end_time"), text_processor)
            text = str(raw.get("text") or "").strip()
            if start is None or end is None or end <= start or not text:
                continue
            key = (round(start, 3), round(end, 3), text)
            if key in seen:
                continue
            seen.add(key)
            subtitles.append({"start": start, "end": end, "text": text})
    subtitles.sort(key=lambda item: (item["start"], item["end"]))
    return subtitles


def _aspect(clip: Dict[str, Any]) -> str:
    explicit = str(clip.get("product_aspect") or "").strip().lower()
    if explicit:
        return explicit
    primary_text = " ".join(
        str(clip.get(key) or "")
        for key in ("product", "product_name", "outline", "selling_point")
    )
    if _contains_any(primary_text, PRICE_BOUNDARIES):
        return "price"
    if _contains_any(primary_text, PACKAGING_WORDS):
        return "packaging"
    if _contains_any(primary_text, FLAVOR_WORDS):
        return "flavor"
    content = str(clip.get("content") or "")
    if _contains_any(content, PACKAGING_WORDS):
        return "packaging"
    if _contains_any(content, FLAVOR_WORDS):
        return "flavor"
    if _contains_any(content, PRICE_BOUNDARIES):
        return "price"
    return "general"


def _contains_any(text: str, words: tuple[str, ...]) -> bool:
    return any(word in text for word in words)


def _is_suite_start_text(text: str) -> bool:
    return _contains_any(text, SUITE_START_CUES) or (
        "\u4ecb\u7ecd" in text
        and _contains_any(text, SUITE_START_PREFIXES)
    )


def _covers_multiple_suite_children(text: str) -> bool:
    cues = (
        "\u8fd8\u6709",
        "\u53e6\u5916\u4e00\u6b3e",
        "\u4e0b\u4e00\u6b3e",
        "\u6700\u540e\u4e00\u4e2a",
        "\u6700\u540e\u4e00\u6b3e",
    )
    return sum(text.count(cue) for cue in cues) >= 2


def _subtitle_content(
    subtitles: List[Dict[str, Any]],
    start: float,
    end: float,
) -> str:
    return " ".join(
        item["text"]
        for item in subtitles
        if item["end"] > start and item["start"] < end - 0.001
    ).strip()


def _continuation_end(
    clip: Dict[str, Any],
    clip_range: Tuple[float, float],
    subtitles: List[Dict[str, Any]],
    max_duration_seconds: float,
) -> float:
    start, end = clip_range
    product = clip.get("product") or clip.get("product_name") or ""
    if not is_suite_product(product):
        return end

    following = [
        item
        for item in subtitles
        if item["start"] >= end - 0.5 and item["start"] <= end + 16.0
    ]
    cue_index = next(
        (
            index
            for index, item in enumerate(following[:5])
            if _contains_any(item["text"], CONTINUATION_CUES)
        ),
        None,
    )
    if cue_index is None:
        return end

    candidate_end = end
    previous_end = end
    aspect = _aspect(clip)
    for offset, item in enumerate(following[cue_index:]):
        # The explicit continuation cue may sit across a chunk checkpoint.
        # Once it is found, following lines must return to normal continuity.
        if offset > 0 and item["start"] - previous_end > 6.0:
            break
        if item["end"] - start > max_duration_seconds:
            break
        text = item["text"]
        if candidate_end > end and _contains_any(text, HARD_STOPS):
            break
        if aspect not in {"price", "gift"} and _contains_any(text, PRICE_BOUNDARIES):
            break
        candidate_end = max(candidate_end, item["end"])
        previous_end = item["end"]
    return candidate_end


def _suite_start(
    clip: Dict[str, Any],
    clip_range: Tuple[float, float],
    subtitles: List[Dict[str, Any]],
) -> float:
    start, _ = clip_range
    product = clip.get("product") or clip.get("product_name") or ""
    if not is_suite_product(product) or _aspect(clip) in {"price", "gift", "packaging"}:
        return start

    family = product_family_name(product)
    family_core = family
    for word in ("\u793c\u76d2", "\u5957\u88c5", "\u5957\u76d2", "\u5957\u9910", "\u7ec4\u5408"):
        family_core = family_core.replace(word, "")
    family_core = family_core.replace("\u7684", "")
    if len(family_core) < 2:
        return start

    candidates = [
        item
        for item in subtitles
        if start - 45.0 <= item["start"] < start
        and family_core in item["text"].replace("\u7684", "")
        and "\u793c\u76d2" in item["text"]
        and _is_suite_start_text(item["text"])
    ]
    if not candidates:
        return start
    return candidates[-1]["start"]


def _trim_price_boundary(
    clip: Dict[str, Any],
    start: float,
    end: float,
    subtitles: List[Dict[str, Any]],
) -> float:
    if _aspect(clip) in {"price", "gift"}:
        return end
    for item in subtitles:
        if item["start"] <= start + 8.0 or item["start"] >= end:
            continue
        if _contains_any(item["text"], PRICE_BOUNDARIES):
            return item["start"]
    return end


def _complete_suite_tail(
    clip: Dict[str, Any],
    start: float,
    end: float,
    subtitles: List[Dict[str, Any]],
    max_duration_seconds: float,
) -> float:
    product = clip.get("product") or clip.get("product_name") or ""
    if not is_suite_product(product) or _aspect(clip) in {"price", "gift"}:
        return end
    completed_end = end
    for item in subtitles:
        if item["start"] < completed_end - 0.001:
            continue
        if item["start"] - completed_end > 1.25:
            break
        if item["end"] - start > max_duration_seconds:
            break
        if _contains_any(item["text"], PRICE_BOUNDARIES + HARD_STOPS):
            break
        completed_end = item["end"]
    return completed_end


def audit_product_timeline(
    clips: List[Dict[str, Any]],
    srt_chunks_dir: Path,
    text_processor: Any,
    max_duration_seconds: float = 95.0,
) -> List[Dict[str, Any]]:
    """Audit model clips against the complete subtitle timeline.

    The model still decides what is useful. This pass supplies deterministic
    product identity and repairs suite clips only when a nearby enumeration cue
    proves that another child product continues beyond the proposed boundary.
    """
    if not clips or not srt_chunks_dir.exists():
        return clips

    subtitles = _load_global_subtitles(srt_chunks_dir, text_processor)
    if not subtitles:
        return clips

    audited: List[Dict[str, Any]] = []
    extensions = 0
    for source in clips:
        clip = deepcopy(source)
        product = clip.get("product") or clip.get("product_name") or ""
        canonical = canonical_product_name(product)
        parent = parent_product_name(product)
        family = product_family_name(product)
        if canonical:
            clip["canonical_product"] = canonical
        if parent:
            clip["parent_product"] = parent
        if family:
            clip["product_family"] = family

        clip_range = _range(clip, text_processor)
        if not clip_range:
            audited.append(clip)
            continue
        old_start, old_end = clip_range
        new_start = _suite_start(clip, clip_range, subtitles)
        new_end = _continuation_end(
            clip,
            clip_range,
            subtitles,
            max_duration_seconds,
        )
        new_end = _trim_price_boundary(clip, new_start, new_end, subtitles)
        new_end = _complete_suite_tail(
            clip,
            new_start,
            new_end,
            subtitles,
            max_duration_seconds,
        )
        if new_end <= new_start:
            new_start, new_end = old_start, old_end
        boundary_changed = (
            new_start < old_start - 0.25
            or abs(new_end - old_end) > 0.25
        )
        if boundary_changed:
            clip["start_time"] = _srt_time(new_start)
            clip["end_time"] = _srt_time(new_end)
            clip["duration"] = round(new_end - new_start, 3)
            clip["duration_seconds"] = clip["duration"]
            clip["content"] = _subtitle_content(subtitles, new_start, new_end)
            clip["product_boundary_audited"] = True
            clip["original_start_time"] = _srt_time(old_start)
            clip["original_end_time"] = _srt_time(old_end)
            extensions += 1
        if parent and (
            boundary_changed
            or _covers_multiple_suite_children(str(clip.get("content") or ""))
        ):
            clip["original_product"] = product
            clip["product"] = parent
            clip["canonical_product"] = parent
            clip["product_family"] = parent
        audited.append(clip)

    if extensions:
        logger.info(
            "Global product timeline audit repaired %s suite boundaries",
            extensions,
        )
    return audited
