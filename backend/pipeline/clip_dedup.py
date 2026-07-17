"""Utilities for removing duplicate clip candidates."""

import logging
import re
from copy import deepcopy
from typing import Any, Dict, List, Optional, Tuple

from .product_identity import product_family_name

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


def _quality_key(clip: Dict[str, Any]) -> Tuple[int, int, float, float, int, int, int]:
    score = float(clip.get("final_score", clip.get("score", 0)) or 0)
    clip_range = _clip_range(clip)
    duration = (clip_range[1] - clip_range[0]) if clip_range else 0
    text_len = len(str(clip.get("content") or "")) + len(str(clip.get("recommend_reason") or ""))
    has_title = 1 if (clip.get("generated_title") or clip.get("title")) else 0
    duration_fit = -abs(duration - 45)
    product_specificity = len(_product_key(clip))
    boundary_audited = 1 if clip.get("product_boundary_audited") else 0
    publishable_duration = 1 if duration >= 15.0 else 0
    return (
        boundary_audited,
        publishable_duration,
        score,
        duration_fit,
        product_specificity,
        text_len,
        has_title,
    )


def _compact_text(value: Any) -> str:
    return re.sub(r"[\s()（）《》【】\-_:/·,，。；;、]+", "", str(value or ""))


def _product_key(clip: Dict[str, Any]) -> str:
    product = clip.get("product") or clip.get("product_name")
    if product:
        return _compact_text(product)[:48]
    text = " ".join(
        str(clip.get(key) or "")
        for key in ("generated_title", "title", "outline")
    )
    return _compact_text(text)[:48]


def _product_family_key(clip: Dict[str, Any]) -> str:
    explicit = clip.get("product_family") or clip.get("parent_product")
    if explicit:
        return _compact_text(explicit)[:48]
    product = clip.get("product") or clip.get("product_name")
    if product:
        return _compact_text(product_family_name(product))[:48]
    return _product_key(clip)


def _aspect_key(clip: Dict[str, Any]) -> str:
    explicit = clip.get("product_aspect")
    if explicit:
        return str(explicit)
    text = " ".join(
        str(clip.get(key) or "")
        for key in ("selling_point", "outline", "generated_title", "title", "content")
    )
    aspect_words = {
        "packaging": ("包装", "礼盒", "套装", "颜色", "设计", "插画", "材质", "工艺"),
        "flavor": ("口味", "味道", "香气", "入口", "酸甜", "清爽", "冷泡", "热泡", "功效"),
        "price": ("价格", "到手", "优惠", "券", "链接", "拍下", "下单", "月销"),
        "gift": ("赠品", "送", "赠", "杯垫", "茶勺", "权益"),
        "audience": ("适合", "人群", "女生", "老人", "小孩", "送礼", "自用"),
    }
    for aspect, words in aspect_words.items():
        if any(word in text for word in words):
            return aspect
    return "general"


def _category_key(clip: Dict[str, Any]) -> str:
    product_text = _product_key(clip)
    text = " ".join(
        str(clip.get(key) or "")
        for key in ("product", "product_name", "generated_title", "title", "outline", "content")
    )
    priority_categories = {
        "tea_pet": ("茶宠", "兔兔茶丛", "小兔子"),
        "tea": ("乌龙茶", "花茶", "果茶", "茶砖", "桂花", "茉莉", "樱花", "朗姆", "礼盒"),
        "jewelry": ("手链", "项链", "耳钉", "戒指", "饰品", "编绳"),
        "jewelry_box": ("首饰盒", "抱抱兔", "胡桃夹子", "魔术师兔兔", "笑茶系列", "瑕疵款", "微瑕"),
    }
    for category, words in priority_categories.items():
        if any(word in product_text for word in words):
            return category

    categories = {
        "jewelry_box": ("首饰盒", "抱抱兔", "胡桃夹子", "魔术师兔兔", "笑茶系列", "瑕疵款", "微瑕"),
        "tea_pet": ("茶宠", "兔兔茶丛", "小兔子"),
        "tea": ("乌龙茶", "花茶", "果茶", "茶砖", "桂花", "茉莉", "樱花", "朗姆", "礼盒"),
        "jewelry": ("手链", "项链", "耳钉", "戒指", "饰品", "编绳"),
    }
    for category, words in categories.items():
        if any(word in text for word in words):
            return category
    return ""


def _product_is_related(left: str, right: str) -> bool:
    if not left or not right:
        return False
    if left in right or right in left:
        return True
    shared = set(left) & set(right)
    return len(shared) >= 3 and len(shared) / max(min(len(left), len(right)), 1) >= 0.45


def _is_heavily_overlapped(left: Dict[str, Any], right: Dict[str, Any]) -> bool:
    left_range = _clip_range(left)
    right_range = _clip_range(right)
    if not left_range or not right_range:
        return False

    left_product = _product_key(left)
    right_product = _product_key(right)
    left_aspect = _aspect_key(left)
    right_aspect = _aspect_key(right)

    left_start, left_end = left_range
    right_start, right_end = right_range
    overlap = min(left_end, right_end) - max(left_start, right_start)
    if overlap <= 0:
        return False

    left_duration = left_end - left_start
    right_duration = right_end - right_start
    shorter_duration = min(left_duration, right_duration)
    longer_duration = max(left_duration, right_duration)
    union = max(left_end, right_end) - min(left_start, right_start)
    if shorter_duration <= 0 or longer_duration <= 0 or union <= 0:
        return False

    shorter_coverage = overlap / shorter_duration
    longer_coverage = overlap / longer_duration
    union_coverage = overlap / union
    same_start = abs(left_start - right_start) <= 3
    same_end = abs(left_end - right_end) <= 3
    left_family = _product_family_key(left)
    right_family = _product_family_key(right)
    same_family = bool(left_family) and left_family == right_family
    same_product = bool(left_product) and left_product == right_product

    if (
        same_product
        and shorter_coverage >= 0.95
        and (same_start or same_end or union_coverage >= 0.65)
    ):
        return True

    # A very short child-product tail fully contained in a complete suite
    # explanation is not independently publishable. Longer child sections
    # remain separate because they can carry useful product-specific detail.
    if (
        same_family
        and shorter_duration < 15.0
        and longer_duration >= 30.0
        and shorter_coverage >= 0.95
    ):
        return True

    if left_product and right_product and left_product != right_product:
        same_category = _category_key(left) and _category_key(left) == _category_key(right)
        related_product = _product_is_related(left_product, right_product)
        if same_category and shorter_coverage >= 0.95 and union_coverage >= 0.25:
            return True
        if same_category and left_aspect != right_aspect and union_coverage < 0.82:
            return False
        return (
            union_coverage >= 0.82
            or (same_start and same_end)
            or (same_category and shorter_coverage >= 0.82 and union_coverage >= 0.25)
            or (related_product and shorter_coverage >= 0.85 and union_coverage >= 0.45)
        )

    if left_aspect != right_aspect and union_coverage < 0.9:
        return False

    # Same-window duplicates can differ by a few seconds after padding or local
    # fallback recovery. Treat a near-contained short clip as duplicate only
    # when one boundary is effectively the same, so adjacent product sections
    # are not collapsed by accident.
    return union_coverage >= 0.65 or (
        shorter_coverage >= 0.8 and (longer_coverage >= 0.45 or same_start or same_end)
    )


def dedupe_clips_by_time(clips: List[Dict[str, Any]], source: str = "clips") -> List[Dict[str, Any]]:
    """Remove exact and heavily overlapping duplicate clip candidates.

    The LLM can occasionally return duplicate candidates with different ids but
    identical or near-contained time ranges. Keeping the highest-quality
    candidate here prevents duplicate exports, collections, and database rows
    downstream while preserving adjacent product explanations.
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

    exact_deduped = passthrough + list(best_by_range.values())
    exact_deduped.sort(key=lambda item: (_time_to_seconds(item.get("start_time")) or 0, _time_to_seconds(item.get("end_time")) or 0))

    deduped: List[Dict[str, Any]] = []
    for clip in exact_deduped:
        if _clip_range(clip) is None:
            deduped.append(clip)
            continue

        duplicate_indexes = [
            index
            for index, existing in enumerate(deduped)
            if _is_heavily_overlapped(clip, existing)
        ]
        if not duplicate_indexes:
            deduped.append(clip)
            continue

        duplicates += len(duplicate_indexes)
        candidates = [clip] + [deduped[index] for index in duplicate_indexes]
        best = max(candidates, key=_quality_key)
        for index in reversed(duplicate_indexes):
            deduped.pop(index)
        deduped.append(best)

    deduped.sort(key=lambda item: (_time_to_seconds(item.get("start_time")) or 0, _time_to_seconds(item.get("end_time")) or 0))

    if duplicates:
        logger.info("Removed %s duplicate clip candidates from %s", duplicates, source)
    return deduped


def merge_cross_chunk_product_clips(
    clips: List[Dict[str, Any]],
    max_gap_seconds: float = 12.0,
    max_duration_seconds: float = 95.0,
) -> List[Dict[str, Any]]:
    """Join one logical product section split only by an analysis boundary.

    Different selling-point aspects remain separate, as do long sections that
    should naturally be edited into multiple short videos.
    """
    if not clips:
        return []

    ordered = sorted(
        (deepcopy(clip) for clip in clips),
        key=lambda item: (
            _time_to_seconds(item.get("start_time")) or 0,
            _time_to_seconds(item.get("end_time")) or 0,
        ),
    )
    merged: List[Dict[str, Any]] = []

    for clip in ordered:
        if not merged:
            merged.append(clip)
            continue

        previous = merged[-1]
        previous_range = _clip_range(previous)
        current_range = _clip_range(clip)
        try:
            previous_chunk = int(previous.get("chunk_index"))
            current_chunk = int(clip.get("chunk_index"))
        except (TypeError, ValueError):
            previous_chunk = current_chunk = -100

        should_merge = False
        if previous_range and current_range and current_chunk == previous_chunk + 1:
            previous_start, previous_end = previous_range
            current_start, current_end = current_range
            gap = current_start - previous_end
            combined_duration = max(previous_end, current_end) - min(previous_start, current_start)
            previous_product = _compact_text(previous.get("product") or previous.get("product_name"))
            current_product = _compact_text(clip.get("product") or clip.get("product_name"))
            previous_aspect = _aspect_key(previous)
            current_aspect = _aspect_key(clip)
            same_logic = (
                previous_aspect == current_aspect
                or previous_aspect == "general"
                or current_aspect == "general"
            )
            should_merge = (
                -max_gap_seconds <= gap <= max_gap_seconds
                and combined_duration <= max_duration_seconds
                and _product_is_related(previous_product, current_product)
                and same_logic
            )

        if not should_merge:
            merged.append(clip)
            continue

        previous_start, previous_end = previous_range
        current_start, current_end = current_range
        previous["start_time"] = _seconds_to_srt_time(min(previous_start, current_start))
        previous["end_time"] = _seconds_to_srt_time(max(previous_end, current_end))
        previous["duration"] = round(max(previous_end, current_end) - min(previous_start, current_start), 3)
        previous["duration_seconds"] = previous["duration"]
        previous["cross_chunk_merged"] = True
        previous["merged_chunk_indexes"] = sorted(
            set(previous.get("merged_chunk_indexes") or [previous_chunk]) | {current_chunk}
        )
        for field in ("content", "summary", "selling_point", "cut_reason"):
            left = str(previous.get(field) or "").strip()
            right = str(clip.get(field) or "").strip()
            if right and right not in left:
                previous[field] = f"{left} {right}".strip()
        logger.info(
            "Merged cross-chunk product section %s across chunks %s and %s",
            previous_product or current_product,
            previous_chunk,
            current_chunk,
        )

    return merged


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
