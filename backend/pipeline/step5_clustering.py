"""Step 5: cluster clips into collections.

Product clips need a different strategy from generic commentary clips. For
shopping/live-stream videos, a useful collection is usually "all clips about
this product", even when there is only one high-signal clip. The previous
generic topic clustering could hide product clips under vague themes.
"""
import json
import logging
import re
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, List, Optional

from ..core.shared_config import MAX_CLIPS_PER_COLLECTION, METADATA_DIR, PROMPT_FILES
from ..utils.llm_client import LLMClient
from .clip_dedup import dedupe_clips_by_time

logger = logging.getLogger(__name__)


PRODUCT_SUFFIXES = ("茶", "酒", "果茶", "果酒", "礼盒", "套装", "套餐", "杯", "瓶", "罐", "盒", "包")
PRODUCT_STOPWORDS = ("产品", "价格", "优惠", "口味", "人群", "场景", "直播", "介绍", "切片")


def _text_from_clip(clip: Dict[str, Any]) -> str:
    parts: List[str] = []
    for key in (
        "product",
        "product_name",
        "generated_title",
        "title",
        "title_angle",
        "outline",
        "selling_point",
        "product_value",
        "recommend_reason",
        "content",
    ):
        value = clip.get(key)
        if isinstance(value, list):
            parts.extend(str(item) for item in value if item)
        elif isinstance(value, dict):
            parts.extend(str(item) for item in value.values() if item)
        elif value:
            parts.append(str(value))
    return " ".join(parts)


def _normalize_product_name(name: str) -> str:
    cleaned = re.sub(r"[\s:：,，。；;、()\[\]【】]+", "", str(name or ""))
    cleaned = cleaned.strip("-_")
    if not cleaned:
        return ""
    if "狼木果茶" in cleaned and "朗姆" not in cleaned:
        return "狼木果茶（朗姆果茶）"
    return cleaned[:28]


def _extract_product_name(clip: Dict[str, Any]) -> str:
    for key in ("product", "product_name"):
        product = _normalize_product_name(clip.get(key, ""))
        if product:
            return product

    text = _text_from_clip(clip)
    if not text:
        return ""

    explicit_patterns = [
        r"([\u4e00-\u9fa5A-Za-z0-9]{2,18}(?:果茶|果酒|礼盒|套装|套餐))",
        r"([\u4e00-\u9fa5A-Za-z0-9]{2,18}(?:茶|酒))",
    ]
    for pattern in explicit_patterns:
        for match in re.findall(pattern, text):
            product = _normalize_product_name(match)
            if product and product not in PRODUCT_STOPWORDS:
                return product
    return ""


def _clip_duration_seconds(clip: Dict[str, Any]) -> float:
    def parse(value: Any) -> Optional[float]:
        if value is None:
            return None
        text = str(value).replace(",", ".")
        parts = text.split(":")
        try:
            if len(parts) == 3:
                return int(parts[0]) * 3600 + int(parts[1]) * 60 + float(parts[2])
            if len(parts) == 2:
                return int(parts[0]) * 60 + float(parts[1])
            return float(parts[0])
        except Exception:
            return None

    start = parse(clip.get("start_time"))
    end = parse(clip.get("end_time"))
    if start is None or end is None:
        return 0.0
    return max(0.0, end - start)


class ClusteringEngine:
    """Cluster clips into collections."""

    def __init__(self, metadata_dir: Optional[Path] = None, prompt_files: Dict = None):
        self.llm_client = LLMClient()
        prompt_files_to_use = prompt_files if prompt_files is not None else PROMPT_FILES
        with open(prompt_files_to_use["clustering"], "r", encoding="utf-8") as f:
            self.clustering_prompt = f.read()
        self.metadata_dir = metadata_dir or METADATA_DIR

    def cluster_clips(self, clips_with_titles: List[Dict]) -> List[Dict]:
        logger.info("Clustering %s clips", len(clips_with_titles))
        product_collections = self._create_product_collections(clips_with_titles)
        if product_collections:
            logger.info("Created %s product collections", len(product_collections))
            return product_collections

        try:
            llm_collections = self._cluster_with_llm(clips_with_titles)
            if llm_collections:
                return llm_collections
        except Exception as exc:
            logger.warning("LLM clustering failed; using local fallback: %s", exc)
        return self._create_default_collections(clips_with_titles)

    def _create_product_collections(self, clips: List[Dict]) -> List[Dict]:
        grouped: "OrderedDict[str, List[Dict]]" = OrderedDict()
        product_like_count = 0

        for clip in clips:
            product = _extract_product_name(clip)
            if not product:
                continue
            clip["product"] = product
            product_like_count += 1
            grouped.setdefault(product, []).append(clip)

        if product_like_count < max(1, len(clips) // 5):
            return []

        collections: List[Dict[str, Any]] = []
        collection_id = 1
        for product, product_clips in grouped.items():
            ordered = sorted(
                product_clips,
                key=lambda item: (
                    -float(item.get("final_score", 0) or 0),
                    -_clip_duration_seconds(item),
                    str(item.get("start_time", "")),
                ),
            )
            selected = ordered[: max(MAX_CLIPS_PER_COLLECTION, 1)]
            highlights = []
            for clip in selected[:3]:
                point = clip.get("selling_point") or clip.get("title_angle") or clip.get("generated_title")
                if point:
                    highlights.append(str(point))
            summary_tail = "；".join(highlights) if highlights else "覆盖卖点、口味、价格、适用人群等产品介绍片段"
            collections.append(
                {
                    "id": str(collection_id),
                    "collection_title": f"{product} 产品介绍",
                    "collection_summary": summary_tail[:160],
                    "clip_ids": [str(clip["id"]) for clip in selected],
                    "collection_type": "product",
                    "product": product,
                }
            )
            collection_id += 1

        all_product_ids = []
        for product_clips in grouped.values():
            for clip in product_clips:
                clip_id = str(clip["id"])
                if clip_id not in all_product_ids:
                    all_product_ids.append(clip_id)

        if len(collections) > 1:
            collections.insert(
                0,
                {
                    "id": "0",
                    "collection_title": "全部产品介绍切片",
                    "collection_summary": "按原始时间线汇总所有产品介绍，便于检查是否漏掉某个产品。",
                    "clip_ids": all_product_ids[: max(MAX_CLIPS_PER_COLLECTION * 3, MAX_CLIPS_PER_COLLECTION)],
                    "collection_type": "product_overview",
                },
            )

        return collections

    def _cluster_with_llm(self, clips_with_titles: List[Dict]) -> List[Dict]:
        clips_for_clustering = [
            {
                "id": str(clip["id"]),
                "title": clip.get("generated_title") or clip.get("outline") or clip.get("title") or "",
                "summary": clip.get("recommend_reason", ""),
                "score": clip.get("final_score", 0),
            }
            for clip in clips_with_titles
        ]
        prompt = self.clustering_prompt + "\n\n以下是视频切片列表：\n"
        for index, clip in enumerate(clips_for_clustering, start=1):
            prompt += f"{index}. 标题：{clip['title']}\n   摘要：{clip['summary']}\n   评分：{float(clip['score'] or 0):.2f}\n\n"

        response = self.llm_client.call_with_retry(prompt)
        collections_data = self.llm_client.parse_json_response(response)
        return self._validate_collections(collections_data, clips_with_titles)

    def _validate_collections(self, collections_data: List[Dict], clips_with_titles: List[Dict]) -> List[Dict]:
        if not isinstance(collections_data, list):
            return []
        title_to_id = {}
        for clip in clips_with_titles:
            for value in (clip.get("generated_title"), clip.get("title"), clip.get("outline")):
                if value:
                    title_to_id[str(value)] = str(clip["id"])

        validated = []
        for index, collection in enumerate(collections_data, start=1):
            if not isinstance(collection, dict):
                continue
            clip_refs = collection.get("clips") or collection.get("clip_ids") or []
            valid_ids = []
            for ref in clip_refs:
                clip_id = str(ref)
                if any(str(clip.get("id")) == clip_id for clip in clips_with_titles):
                    valid_ids.append(clip_id)
                elif clip_id in title_to_id:
                    valid_ids.append(title_to_id[clip_id])
            valid_ids = list(dict.fromkeys(valid_ids))[:MAX_CLIPS_PER_COLLECTION]
            if len(valid_ids) < 2:
                continue
            validated.append(
                {
                    "id": str(index),
                    "collection_title": collection.get("collection_title", f"合集 {index}"),
                    "collection_summary": collection.get("collection_summary", ""),
                    "clip_ids": valid_ids,
                }
            )
        return validated

    def _create_default_collections(self, clips_with_titles: List[Dict]) -> List[Dict]:
        sorted_clips = sorted(clips_with_titles, key=lambda clip: float(clip.get("final_score", 0) or 0), reverse=True)
        if not sorted_clips:
            return []
        return [
            {
                "id": "1",
                "collection_title": "精选高分片段",
                "collection_summary": "评分最高的精彩片段合集。",
                "clip_ids": [str(clip["id"]) for clip in sorted_clips[:MAX_CLIPS_PER_COLLECTION]],
            }
        ]

    def save_collections(self, collections_data: List[Dict], output_path: Optional[Path] = None) -> Path:
        if output_path is None:
            output_path = self.metadata_dir / "collections.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(collections_data, f, ensure_ascii=False, indent=2)
        logger.info("Saved collections to: %s", output_path)
        return output_path

    def load_collections(self, input_path: Path) -> List[Dict]:
        with open(input_path, "r", encoding="utf-8") as f:
            return json.load(f)


def run_step5_clustering(
    clips_with_titles_path: Path,
    output_path: Optional[Path] = None,
    metadata_dir: Optional[str] = None,
    prompt_files: Dict = None,
) -> List[Dict]:
    with open(clips_with_titles_path, "r", encoding="utf-8") as f:
        clips_with_titles = json.load(f)
    clips_with_titles = dedupe_clips_by_time(clips_with_titles, "step5_clustering_input")

    metadata_path = Path(metadata_dir) if metadata_dir is not None else METADATA_DIR
    clusterer = ClusteringEngine(metadata_dir=metadata_path, prompt_files=prompt_files)
    collections_data = clusterer.cluster_clips(clips_with_titles)

    if output_path is None:
        output_path = metadata_path / "step5_collections.json"
    clusterer.save_collections(collections_data, output_path)
    return collections_data
