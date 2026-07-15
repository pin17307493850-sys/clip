"""Product-aware clip recovery for livestream videos."""

import json
import logging
import re
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

logger = logging.getLogger(__name__)


ASPECT_KEYWORDS: Dict[str, Tuple[str, Tuple[str, ...]]] = {
    "packaging": ("包装设计", ("包装", "礼盒", "套装", "盒子", "颜色", "配色", "设计", "插画", "材质", "工艺", "颜值")),
    "flavor": ("口味功能", ("口味", "味道", "香气", "入口", "酸甜", "清爽", "冷泡", "热泡", "功效", "功能", "解腻", "搭配")),
    "price": ("价格权益", ("价格", "到手", "月销", "优惠", "满", "减", "券", "链接", "拍下", "下单", "库存")),
    "gift": ("赠品规则", ("赠品", "送", "赠", "杯垫", "茶勺", "礼品", "权益", "会员")),
    "audience": ("适合人群", ("适合", "人群", "女生", "老人", "小孩", "熬夜", "办公室", "送礼", "自用", "场景")),
    "reason": ("下单理由", ("推荐", "值得", "划算", "必入", "回购", "卖点", "重点", "优势", "闭眼入")),
}

STOP_WORDS = ("下一个", "换一个", "接下来", "再看", "另外", "抽奖", "物流", "关注", "小助理")


def _time_to_seconds(text_processor: Any, value: Any) -> Optional[float]:
    if value is None:
        return None
    try:
        text = str(value).strip().replace(",", ".")
        if not text:
            return None
        return float(text_processor.time_to_seconds(text))
    except Exception:
        return None


def _seconds_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    total_ms = int(round(seconds * 1000))
    hours, remainder = divmod(total_ms, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, millis = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{millis:03d}"


def _text_of(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, dict):
        return " ".join(_text_of(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return " ".join(_text_of(item) for item in value)
    return str(value)


def _compact(text: str) -> str:
    return re.sub(r"[\s()（）《》【】\-_:/·,，。；;、]+", "", text or "")


def _product_aliases(product: str) -> List[str]:
    compact = _compact(product)
    aliases = {compact}
    for suffix in ("产品", "礼盒", "套装", "组合", "茶", "果茶", "乌龙", "花茶", "首饰", "项链", "耳钉", "手链"):
        if compact.endswith(suffix) and len(compact) > len(suffix) + 1:
            aliases.add(compact[: -len(suffix)])
    for sep in ("（", "(", "·", "-", "/", "：", ":"):
        if sep in product:
            head = product.split(sep, 1)[0].strip()
            if len(head) >= 2:
                aliases.add(_compact(head))
    return [alias for alias in aliases if len(alias) >= 2]


def _product_matches(product: str, text: str) -> bool:
    compact_text = _compact(text)
    return any(alias in compact_text for alias in _product_aliases(product))


def _extract_products(items: Iterable[Dict[str, Any]]) -> List[str]:
    products: List[str] = []
    seen = set()
    for item in items:
        product = str(item.get("product") or item.get("product_name") or "").strip()
        if not product:
            title = str(item.get("title") or item.get("outline") or item.get("generated_title") or "").strip()
            if any(word in title for word in ("礼盒", "套装", "茶", "果茶", "乌龙", "花茶", "首饰")):
                product = title[:40]
        key = _compact(product)
        if product and key and key not in seen:
            seen.add(key)
            products.append(product)
    return products


def _item_range(text_processor: Any, item: Dict[str, Any]) -> Optional[Tuple[float, float]]:
    start = _time_to_seconds(text_processor, item.get("start_time"))
    end = _time_to_seconds(text_processor, item.get("end_time"))
    if start is None or end is None or end <= start:
        return None
    return start, end


def _overlap_ratio(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    overlap = min(a[1], b[1]) - max(a[0], b[0])
    if overlap <= 0:
        return 0.0
    shorter = min(a[1] - a[0], b[1] - b[0])
    return overlap / max(shorter, 0.001)


def _has_nearby_product_mention(
    subtitles: List[Dict[str, Any]],
    product: str,
    start: float,
    end: float,
    radius: float = 24.0,
) -> bool:
    return any(
        sub["end"] >= start - radius
        and sub["start"] <= end + radius
        and _product_matches(product, sub["text"])
        for sub in subtitles
    )


def _load_subtitles(srt_chunks_dir: Path, chunk_index: Any, text_processor: Any) -> List[Dict[str, Any]]:
    path = srt_chunks_dir / f"chunk_{chunk_index}.json"
    if not path.exists():
        return []
    try:
        raw_items = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return []

    subtitles: List[Dict[str, Any]] = []
    for raw in raw_items:
        start = _time_to_seconds(text_processor, raw.get("start_time"))
        end = _time_to_seconds(text_processor, raw.get("end_time"))
        text = str(raw.get("text") or "").strip()
        if start is None or end is None or end <= start or not text:
            continue
        subtitles.append({"start": start, "end": end, "text": text})
    return subtitles


def _product_window(text_processor: Any, product: str, chunk_index: Any, outlines: List[Dict[str, Any]], timeline: List[Dict[str, Any]], subtitles: List[Dict[str, Any]]) -> Optional[Tuple[float, float]]:
    ranges: List[Tuple[float, float]] = []
    for item in list(outlines) + list(timeline):
        if item.get("chunk_index") != chunk_index:
            continue
        text = _text_of(item)
        item_product = str(item.get("product") or "").strip()
        if item_product != product and not _product_matches(product, text):
            continue
        item_range = _item_range(text_processor, item)
        if item_range:
            ranges.append(item_range)

    if not ranges:
        for sub in subtitles:
            if _product_matches(product, sub["text"]):
                ranges.append((sub["start"], sub["end"]))

    if not ranges:
        return None

    start = max(0.0, min(item[0] for item in ranges) - 90)
    end = max(item[1] for item in ranges) + 150
    return start, end


def _group_aspect_lines(subtitles: List[Dict[str, Any]], product: str, window: Tuple[float, float], aspect_key: str, keywords: Tuple[str, ...]) -> List[Tuple[float, float, List[str]]]:
    hits: List[int] = []
    for index, sub in enumerate(subtitles):
        if sub["end"] < window[0] or sub["start"] > window[1]:
            continue
        text = sub["text"]
        if any(word in text for word in keywords):
            hits.append(index)

    groups: List[Tuple[float, float, List[str]]] = []
    used = set()
    for hit in hits:
        if hit in used:
            continue
        start_index = hit
        end_index = hit
        used.add(hit)

        while start_index > 0:
            previous = subtitles[start_index - 1]
            current = subtitles[start_index]
            if previous["end"] < window[0] or current["start"] - previous["end"] > 5:
                break
            if any(word in previous["text"] for word in STOP_WORDS):
                break
            start_index -= 1

        while end_index + 1 < len(subtitles):
            current = subtitles[end_index]
            following = subtitles[end_index + 1]
            if following["start"] > window[1] or following["start"] - current["end"] > 7:
                break
            if any(word in following["text"] for word in STOP_WORDS):
                break
            end_index += 1
            used.add(end_index)
            if subtitles[end_index]["end"] - subtitles[start_index]["start"] >= 90:
                break

        lines = [subtitles[index]["text"] for index in range(start_index, end_index + 1)]
        text = " ".join(lines)
        start = subtitles[start_index]["start"]
        end = subtitles[end_index]["end"]
        duration = end - start
        has_product = _product_matches(product, text)
        has_aspect = sum(1 for word in keywords if word in text)
        if duration < 8 or len(text) < 24 or (not has_product and has_aspect < 2):
            continue
        if not has_product and not _has_nearby_product_mention(subtitles, product, start, end):
            continue
        groups.append((start, end, lines))
    return groups


def _aspect_for_text(text: str) -> Tuple[str, str, int]:
    best_key = "general"
    best_label = "产品讲解"
    best_score = 0
    for aspect_key, (aspect_label, keywords) in ASPECT_KEYWORDS.items():
        score = sum(1 for keyword in keywords if keyword in text)
        if score > best_score:
            best_key = aspect_key
            best_label = aspect_label
            best_score = score
    return best_key, best_label, best_score


def _group_product_logic_segments(
    subtitles: List[Dict[str, Any]],
    product: str,
    window: Tuple[float, float],
) -> List[Tuple[float, float, str, str, List[str]]]:
    """Build product explanation segments from consecutive subtitle lines."""
    product_words = tuple(keyword for _, keywords in ASPECT_KEYWORDS.values() for keyword in keywords)
    hits: List[int] = []
    for index, sub in enumerate(subtitles):
        if sub["end"] < window[0] or sub["start"] > window[1]:
            continue
        text = sub["text"]
        if _product_matches(product, text) or any(keyword in text for keyword in product_words):
            hits.append(index)

    groups: List[Tuple[float, float, str, str, List[str]]] = []
    if not hits:
        return groups

    active_start = hits[0]
    active_end = hits[0]

    def flush_group(start_index: int, end_index: int) -> None:
        if end_index < start_index:
            return
        while start_index > 0 and subtitles[start_index]["start"] - subtitles[start_index - 1]["end"] <= 3:
            previous_text = subtitles[start_index - 1]["text"]
            if any(word in previous_text for word in STOP_WORDS):
                break
            start_index -= 1
        while end_index + 1 < len(subtitles) and subtitles[end_index + 1]["start"] - subtitles[end_index]["end"] <= 4:
            next_text = subtitles[end_index + 1]["text"]
            if any(word in next_text for word in STOP_WORDS):
                break
            end_index += 1

        lines = [subtitles[index]["text"] for index in range(start_index, end_index + 1)]
        text = " ".join(lines)
        start = subtitles[start_index]["start"]
        end = subtitles[end_index]["end"]
        duration = end - start
        aspect_key, aspect_label, aspect_score = _aspect_for_text(text)
        has_product = _product_matches(product, text)
        if duration < 10 or duration > 105 or len(text) < 28:
            return
        if not has_product and aspect_score < 2:
            return
        if not has_product and not _has_nearby_product_mention(subtitles, product, start, end):
            return
        groups.append((start, end, aspect_key, aspect_label, lines))

    for hit in hits[1:]:
        previous = subtitles[active_end]
        current = subtitles[hit]
        current_duration = current["end"] - subtitles[active_start]["start"]
        if hit <= active_end + 3 and current["start"] - previous["end"] <= 10 and current_duration <= 100:
            active_end = hit
            continue
        flush_group(active_start, active_end)
        active_start = hit
        active_end = hit

    flush_group(active_start, active_end)
    return groups


def enrich_product_logic_clips(
    timeline_data: List[Dict[str, Any]],
    outlines: List[Dict[str, Any]],
    srt_chunks_dir: Path,
    text_processor: Any,
) -> List[Dict[str, Any]]:
    """Add product/aspect clips that are visible in subtitles but missed by LLM."""
    if not outlines or not srt_chunks_dir.exists():
        return timeline_data

    products = _extract_products(list(timeline_data) + list(outlines))
    if not products:
        return timeline_data

    chunk_indexes = sorted({item.get("chunk_index") for item in list(timeline_data) + list(outlines) if item.get("chunk_index") is not None})
    existing_ranges: List[Tuple[str, str, Tuple[float, float]]] = []
    for item in timeline_data:
        item_range = _item_range(text_processor, item)
        if not item_range:
            continue
        product = str(item.get("product") or "").strip()
        aspect = str(item.get("product_aspect") or item.get("selling_point") or item.get("outline") or "")
        existing_ranges.append((product, aspect, item_range))

    additions: List[Dict[str, Any]] = []
    for chunk_index in chunk_indexes:
        subtitles = _load_subtitles(srt_chunks_dir, chunk_index, text_processor)
        if not subtitles:
            continue
        for product in products:
            window = _product_window(text_processor, product, chunk_index, outlines, timeline_data, subtitles)
            if not window:
                continue
            for aspect_key, (aspect_label, keywords) in ASPECT_KEYWORDS.items():
                groups = _group_aspect_lines(subtitles, product, window, aspect_key, keywords)
                for group_start, group_end, lines in groups[:2]:
                    group_range = (group_start, group_end)
                    if any(
                        (_compact(existing_product) == _compact(product) or _product_matches(product, existing_product))
                        and (aspect_key in existing_aspect or aspect_label in existing_aspect or _overlap_ratio(existing_range, group_range) >= 0.88)
                        and _overlap_ratio(existing_range, group_range) >= 0.55
                        for existing_product, existing_aspect, existing_range in existing_ranges
                    ):
                        continue

                    clip = {
                        "id": f"product_logic_{len(additions) + 1}",
                        "chunk_index": chunk_index,
                        "start_time": _seconds_to_srt_time(group_start),
                        "end_time": _seconds_to_srt_time(group_end),
                        "duration_seconds": round(group_end - group_start, 2),
                        "duration": round(group_end - group_start, 2),
                        "outline": f"{product} {aspect_label}",
                        "product": product,
                        "product_aspect": aspect_key,
                        "selling_point": aspect_label,
                        "content": " ".join(lines),
                        "cut_reason": f"产品逻辑兜底：字幕中出现了{product}的{aspect_label}讲解，作为独立产品介绍切片保留。",
                        "product_logic_guard": True,
                    }
                    additions.append(clip)
                    existing_ranges.append((product, aspect_key, group_range))

            for group_start, group_end, aspect_key, aspect_label, lines in _group_product_logic_segments(subtitles, product, window):
                group_range = (group_start, group_end)
                if any(
                    (_compact(existing_product) == _compact(product) or _product_matches(product, existing_product))
                    and (existing_aspect == aspect_key or _overlap_ratio(existing_range, group_range) >= 0.82)
                    and _overlap_ratio(existing_range, group_range) >= 0.58
                    for existing_product, existing_aspect, existing_range in existing_ranges
                ):
                    continue

                clip = {
                    "id": f"product_logic_segment_{len(additions) + 1}",
                    "chunk_index": chunk_index,
                    "start_time": _seconds_to_srt_time(group_start),
                    "end_time": _seconds_to_srt_time(group_end),
                    "duration_seconds": round(group_end - group_start, 2),
                    "duration": round(group_end - group_start, 2),
                    "outline": f"{product} {aspect_label}",
                    "product": product,
                    "product_aspect": aspect_key,
                    "selling_point": aspect_label,
                    "content": " ".join(lines),
                    "cut_reason": f"产品逻辑兜底：字幕中出现 {product} 的连续产品讲解段落，按 {aspect_label} 独立保留。",
                    "product_logic_guard": True,
                }
                additions.append(clip)
                existing_ranges.append((product, aspect_key, group_range))

    if additions:
        logger.info("Product logic guard recovered %s aspect clips", len(additions))
        return timeline_data + additions
    return timeline_data
