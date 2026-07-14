"""
Step 3: score candidate clips.

This step used to send a large chunk of candidates to the LLM in one request.
If that single request stalled, the whole project looked frozen. The scorer now
works in small batches, persists partial results, and falls back to a local
heuristic score when the LLM cannot return usable JSON.
"""
import json
import logging
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..core.shared_config import METADATA_DIR, MIN_SCORE_THRESHOLD, PROMPT_FILES
from ..utils.llm_client import LLMClient
from ..utils.text_processor import TextProcessor

logger = logging.getLogger(__name__)


ProgressCallback = Optional[Callable[[str, str, int], None]]


class ClipScorer:
    """Score timeline candidates with resumable small batches."""

    def __init__(
        self,
        metadata_dir: Optional[Path] = None,
        prompt_files: Dict = None,
        progress_callback: ProgressCallback = None,
    ):
        self.llm_client = LLMClient()
        self.text_processor = TextProcessor()
        self.metadata_dir = metadata_dir or METADATA_DIR
        self.progress_callback = progress_callback
        self.raw_output_dir = self.metadata_dir / "step3_llm_raw_output"
        self.partial_path = self.metadata_dir / "step3_all_scored.partial.json"

        prompt_files_to_use = prompt_files if prompt_files is not None else PROMPT_FILES
        with open(prompt_files_to_use["recommendation"], "r", encoding="utf-8") as f:
            self.recommendation_prompt = f.read()

    def score_clips(self, timeline_data: List[Dict]) -> List[Dict]:
        if not timeline_data:
            logger.warning("Timeline data is empty; no clips to score")
            return []

        self.raw_output_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Scoring %s timeline candidates in resumable batches", len(timeline_data))

        timeline_by_chunk = defaultdict(list)
        for item in timeline_data:
            timeline_by_chunk[item.get("chunk_index", 0)].append(item)

        total_batches = sum((len(items) + 2) // 3 for items in timeline_by_chunk.values())
        completed_batches = 0
        all_scored_clips: List[Dict] = []

        for chunk_index, chunk_items in timeline_by_chunk.items():
            chunk_items = sorted(chunk_items, key=lambda x: int(x.get("id", 0)))
            for batch_index, start in enumerate(range(0, len(chunk_items), 3), start=1):
                batch = chunk_items[start : start + 3]
                completed_batches += 1
                subpercent = 60 + int((completed_batches / max(total_batches, 1)) * 30)
                self._emit(
                    "ANALYZE",
                    f"正在给候选片段评分 {completed_batches}/{total_batches} 批",
                    subpercent,
                )

                scored_batch = self._score_batch(chunk_index, batch_index, batch)
                all_scored_clips.extend(scored_batch)
                self._save_partial(all_scored_clips)

        all_scored_clips.sort(key=lambda x: int(x.get("id", 0)))
        logger.info("Finished scoring %s clips", len(all_scored_clips))
        return all_scored_clips

    def _score_batch(self, chunk_index: Any, batch_index: int, clips: List[Dict]) -> List[Dict]:
        cache_path = self.raw_output_dir / f"chunk_{chunk_index}_batch_{batch_index}.txt"
        try:
            if cache_path.exists():
                raw_response = cache_path.read_text(encoding="utf-8")
                logger.info("Reusing Step3 LLM cache: %s", cache_path)
            else:
                input_for_llm = [
                    {
                        "id": clip.get("id"),
                        "outline": clip.get("outline"),
                        "content": clip.get("content"),
                        "start_time": clip.get("start_time"),
                        "end_time": clip.get("end_time"),
                    }
                    for clip in clips
                ]
                raw_response = self.llm_client.call_with_retry(
                    self.recommendation_prompt,
                    input_for_llm,
                    max_retries=2,
                )
                cache_path.write_text(raw_response or "", encoding="utf-8")

            parsed_list = self.llm_client.parse_json_response(raw_response)
            if not isinstance(parsed_list, list) or len(parsed_list) != len(clips):
                raise ValueError(
                    f"score response count mismatch: input={len(clips)} output={len(parsed_list) if isinstance(parsed_list, list) else type(parsed_list)}"
                )

            return [self._merge_llm_score(clip, result) for clip, result in zip(clips, parsed_list)]
        except Exception as exc:
            logger.warning(
                "LLM scoring failed for chunk=%s batch=%s; using heuristic scores: %s",
                chunk_index,
                batch_index,
                exc,
            )
            return [self._fallback_score(clip, f"模型评分失败，已使用本地规则保底评分: {exc}") for clip in clips]

    def _merge_llm_score(self, clip: Dict, result: Dict) -> Dict:
        score = result.get("final_score", result.get("score"))
        reason = result.get("recommend_reason", result.get("recommendation_reason"))
        keep = result.get("keep", True)

        if score is None:
            return self._fallback_score(clip, "模型未返回分数，已使用本地规则评分")

        try:
            numeric_score = float(score)
        except Exception:
            return self._fallback_score(clip, "模型分数格式异常，已使用本地规则评分")

        if numeric_score > 1:
            numeric_score = numeric_score / 100
        clip["final_score"] = round(numeric_score if keep else 0.0, 2)
        clip["recommend_reason"] = reason or "模型已完成评分"
        for field in ("keep", "title_angle", "product_value", "duration_advice"):
            if field in result:
                clip[field] = result[field]
        return clip

    def _fallback_score(self, clip: Dict, reason: str) -> Dict:
        content = str(clip.get("content") or "")
        outline = str(clip.get("outline") or "")
        text = f"{outline} {content}"
        duration = self._duration_seconds(clip)

        score = 0.68
        if duration >= 15:
            score += 0.05
        if duration >= 35:
            score += 0.04
        if any(word in text for word in ("产品", "价格", "优惠", "买", "口味", "功效", "适合", "赠", "直播间")):
            score += 0.08
        if any(word in text for word in ("茉莉", "桂花", "樱花", "乌龙", "茶", "罐", "杯垫", "勺")):
            score += 0.07
        if duration > 120:
            score -= 0.04

        clip["final_score"] = round(max(0.55, min(score, 0.92)), 2)
        clip["recommend_reason"] = reason
        clip["keep"] = clip["final_score"] >= 0.65
        clip["duration_advice"] = "本地规则评分，可继续导出后人工复查"
        return clip

    def _duration_seconds(self, clip: Dict) -> float:
        def parse_time(value: Any) -> Optional[float]:
            if not value:
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

        start = parse_time(clip.get("start_time"))
        end = parse_time(clip.get("end_time"))
        if start is None or end is None:
            return 0
        return max(0, end - start)

    def _save_partial(self, scored_clips: List[Dict]) -> None:
        self.partial_path.parent.mkdir(parents=True, exist_ok=True)
        with open(self.partial_path, "w", encoding="utf-8") as f:
            json.dump(scored_clips, f, ensure_ascii=False, indent=2)

    def _emit(self, stage: str, message: str, subpercent: int) -> None:
        if self.progress_callback:
            self.progress_callback(stage, message, subpercent)

    def save_scores(self, scored_clips: List[Dict], output_path: Path):
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(scored_clips, f, ensure_ascii=False, indent=2)
        logger.info("Saved scores to: %s", output_path)


def run_step3_scoring(
    timeline_path: Path,
    metadata_dir: Path = None,
    output_path: Optional[Path] = None,
    prompt_files: Dict = None,
    progress_callback: ProgressCallback = None,
) -> List[Dict]:
    with open(timeline_path, "r", encoding="utf-8") as f:
        timeline_data = json.load(f)

    if metadata_dir is None:
        metadata_dir = METADATA_DIR

    scorer = ClipScorer(metadata_dir=metadata_dir, prompt_files=prompt_files, progress_callback=progress_callback)
    scored_clips = scorer.score_clips(timeline_data)
    high_score_clips = [clip for clip in scored_clips if clip.get("final_score", 0) >= MIN_SCORE_THRESHOLD]

    all_scored_path = metadata_dir / "step3_all_scored.json"
    scorer.save_scores(scored_clips, all_scored_path)

    if output_path is None:
        output_path = metadata_dir / "step3_high_score_clips.json"
    scorer.save_scores(high_score_clips, output_path)

    return high_score_clips
