"""
Step 4: generate titles for scored clips.

Titles are generated in small resumable batches. If the LLM is slow or fails,
the pipeline falls back to a readable local title so video export can continue.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.shared_config import METADATA_DIR, PROMPT_FILES
from ..utils.llm_client import LLMClient
from ..utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, str, int], None]]


def _compact_text(value: Any) -> str:
    if isinstance(value, list):
        value = " ".join(str(item) for item in value if item)
    elif isinstance(value, dict):
        value = " ".join(str(item) for item in value.values() if item)
    return " ".join(str(value or "").replace("\n", " ").split())


def _product_name_from_clip(clip: Dict[str, Any]) -> str:
    for key in ("product", "product_name"):
        product = _compact_text(clip.get(key)).strip(" -_")
        if product:
            if "狼木果茶" in product and "朗姆" not in product:
                return "狼木果茶（朗姆果茶）"
            return product[:24]
    text = _compact_text(
        " ".join(
            _compact_text(clip.get(key))
            for key in ("generated_title", "outline", "content", "title_angle", "recommend_reason")
        )
    )
    if "狼木果茶" in text or "朗姆果茶" in text:
        return "狼木果茶（朗姆果茶）"
    return ""


class TitleGenerator:
    """Generate or fallback titles for scored clips."""

    def __init__(
        self,
        metadata_dir: Optional[Path] = None,
        prompt_files: Dict = None,
        progress_callback: ProgressCallback = None,
    ):
        self.llm_client = LLMClient()
        self.text_processor = TextProcessor()
        self.progress_callback = progress_callback

        prompt_files_to_use = prompt_files if prompt_files is not None else PROMPT_FILES
        with open(prompt_files_to_use["title"], "r", encoding="utf-8") as f:
            self.title_prompt = f.read()

        self.metadata_dir = metadata_dir or METADATA_DIR
        self.llm_raw_output_dir = self.metadata_dir / "step4_llm_raw_output"
        self.partial_path = self.metadata_dir / "step4_titles.partial.json"

    def generate_titles(self, high_score_clips: List[Dict]) -> List[Dict]:
        if not high_score_clips:
            return []

        self.llm_raw_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Generating titles for %s clips in small batches", len(high_score_clips))

        clips_by_chunk = defaultdict(list)
        for clip in high_score_clips:
            clips_by_chunk[clip.get("chunk_index", 0)].append(clip)

        total_batches = sum((len(items) + 4) // 5 for items in clips_by_chunk.values())
        completed_batches = 0
        titled: List[Dict] = []

        for chunk_index, chunk_clips in clips_by_chunk.items():
            chunk_clips = sorted(chunk_clips, key=lambda x: int(x.get("id", 0)))
            for batch_index, start in enumerate(range(0, len(chunk_clips), 5), start=1):
                batch = chunk_clips[start : start + 5]
                completed_batches += 1
                subpercent = 20 + int((completed_batches / max(total_batches, 1)) * 20)
                self._emit(
                    "HIGHLIGHT",
                    f"正在生成切片标题 {completed_batches}/{total_batches} 批",
                    subpercent,
                )
                titled_batch = self._generate_batch(chunk_index, batch_index, batch)
                titled.extend(titled_batch)
                self._save_partial(titled)

        titled.sort(key=lambda x: int(x.get("id", 0)))
        logger.info("Finished title generation for %s clips", len(titled))
        return titled

    def _generate_batch(self, chunk_index: Any, batch_index: int, clips: List[Dict]) -> List[Dict]:
        cache_path = self.llm_raw_output_dir / f"chunk_{chunk_index}_batch_{batch_index}.txt"
        try:
            if cache_path.exists():
                raw_response = cache_path.read_text(encoding="utf-8")
                logger.info("Reusing Step4 LLM cache: %s", cache_path)
            else:
                input_for_llm = [
                    {
                        "id": str(clip.get("id")),
                        "title": clip.get("outline"),
                        "content": clip.get("content"),
                        "recommend_reason": clip.get("recommend_reason"),
                    }
                    for clip in clips
                ]
                raw_response = self.llm_client.call_with_retry(
                    self.title_prompt,
                    input_for_llm,
                    max_retries=1,
                )
                cache_path.write_text(raw_response or "", encoding="utf-8")

            titles_map = self.llm_client.parse_json_response(raw_response)
            if not isinstance(titles_map, dict):
                raise ValueError(f"title response is not a dict: {type(titles_map)}")

            for clip in clips:
                clip_id = str(clip.get("id"))
                generated_title = titles_map.get(clip_id) or titles_map.get(clip.get("id"))
                clip["generated_title"] = (
                    generated_title.strip()
                    if isinstance(generated_title, str) and generated_title.strip()
                    else self._fallback_title(clip)
                )
                clip["generated_title"] = self._product_first_title(clip, clip["generated_title"])
                if not clip.get("title") or str(clip.get("title")).startswith("片段_"):
                    clip["title"] = clip["generated_title"]
            return clips
        except Exception as exc:
            logger.warning(
                "LLM title generation failed for chunk=%s batch=%s; using fallback titles: %s",
                chunk_index,
                batch_index,
                exc,
            )
            for clip in clips:
                clip["generated_title"] = self._product_first_title(clip, self._fallback_title(clip))
                if not clip.get("title") or str(clip.get("title")).startswith("片段_"):
                    clip["title"] = clip["generated_title"]
                clip["title_source"] = "fallback"
            return clips

    def _fallback_title(self, clip: Dict) -> str:
        product = _product_name_from_clip(clip)
        point = _compact_text(clip.get("selling_point") or clip.get("title_angle") or clip.get("product_value"))
        if product and point:
            return f"{product}：{point}"[:42]
        if product:
            return f"{product} 产品介绍"[:42]

        outline = clip.get("outline")
        if isinstance(outline, dict):
            title = outline.get("title") or outline.get("summary") or ""
        else:
            title = str(outline or "")
        title = " ".join(title.replace("\n", " ").split())
        if not title:
            title = str(clip.get("content") or "").strip().replace("\n", " ")
        if not title:
            title = f"产品切片 {clip.get('id', '')}".strip()
        return title[:42]

    def _product_first_title(self, clip: Dict, title: str) -> str:
        product = _product_name_from_clip(clip)
        clean_title = _compact_text(title)
        if not product:
            return clean_title[:42]
        if product in clean_title:
            return clean_title[:42]
        if "狼木果茶" in clean_title and product.startswith("狼木果茶"):
            return clean_title.replace("狼木果茶", product, 1)[:42]
        return f"{product}：{clean_title}"[:42]

    def _save_partial(self, clips_with_titles: List[Dict]) -> None:
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.partial_path, "w", encoding="utf-8") as f:
            json.dump(clips_with_titles, f, ensure_ascii=False, indent=2)

    def _emit(self, stage: str, message: str, subpercent: int) -> None:
        if self.progress_callback:
            self.progress_callback(stage, message, subpercent)

    def save_clips_with_titles(self, clips_with_titles: List[Dict], output_path: Path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(clips_with_titles, f, ensure_ascii=False, indent=2)
        logger.info("Saved titled clips to: %s", output_path)


def normalize_clip_titles(clips_with_titles: List[Dict]) -> List[Dict]:
    """Apply product-first titles to cached title checkpoints."""
    normalizer = TitleGenerator.__new__(TitleGenerator)
    changed = []
    for clip in clips_with_titles:
        title = clip.get("generated_title") or clip.get("title") or ""
        normalized = normalizer._product_first_title(clip, title) if title else normalizer._fallback_title(clip)
        clip["generated_title"] = normalized
        if not clip.get("title") or str(clip.get("title")).startswith("片段_"):
            clip["title"] = normalized
        changed.append(clip)
    return changed


def run_step4_title(
    high_score_clips_path: Path,
    output_path: Optional[Path] = None,
    metadata_dir: Optional[str] = None,
    prompt_files: Dict = None,
    progress_callback: ProgressCallback = None,
) -> List[Dict]:
    with open(high_score_clips_path, "r", encoding="utf-8") as f:
        high_score_clips = json.load(f)

    if metadata_dir is None:
        metadata_dir = METADATA_DIR

    title_generator = TitleGenerator(
        metadata_dir=Path(metadata_dir),
        prompt_files=prompt_files,
        progress_callback=progress_callback,
    )
    clips_with_titles = normalize_clip_titles(title_generator.generate_titles(high_score_clips))

    if output_path is None:
        output_path = Path(metadata_dir) / "step4_titles.json"

    title_generator.save_clips_with_titles(clips_with_titles, output_path)
    return clips_with_titles
