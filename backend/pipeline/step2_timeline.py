"""
Step 2: 时间线提取 - 为大纲中的每个话题定位具体时间区间
"""
import json
import logging
import re
from typing import List, Dict, Any, Optional
from pathlib import Path
from collections import defaultdict

# 导入依赖
from ..utils.llm_client import LLMClient
from ..utils.text_processor import TextProcessor
from ..core.shared_config import PROMPT_FILES, METADATA_DIR
from .clip_dedup import dedupe_clips_by_time, merge_cross_chunk_product_clips
from .product_clip_logic import enrich_product_logic_clips
from .parallel_llm import get_llm_concurrency, run_parallel_ordered

logger = logging.getLogger(__name__)

class TimelineExtractor:
    """从大纲和SRT字幕中提取精确时间线"""
    
    def __init__(self, metadata_dir: Path = None, prompt_files: Dict = None, progress_callback=None):
        self.llm_client = LLMClient()
        self.text_processor = TextProcessor()
        self.progress_callback = progress_callback
        
        # 使用传入的metadata_dir或默认值
        if metadata_dir is None:
            metadata_dir = METADATA_DIR
        self.metadata_dir = metadata_dir
        
        # 加载提示词
        prompt_files_to_use = prompt_files if prompt_files is not None else PROMPT_FILES
        with open(prompt_files_to_use['timeline'], 'r', encoding='utf-8') as f:
            self.timeline_prompt = f.read()
            
        # SRT块的目录
        self.srt_chunks_dir = self.metadata_dir / "step1_srt_chunks"
        self.timeline_chunks_dir = self.metadata_dir / "step2_timeline_chunks"
        self.llm_raw_output_dir = self.metadata_dir / "step2_llm_raw_output"

    def extract_timeline(self, outlines: List[Dict]) -> List[Dict]:
        """
        提取话题时间区间。
        新版特性：
        - 基于预先分块的SRT
        - 按块批量处理
        - 缓存原始LLM响应，避免重复调用
        - 保存每个块的处理结果作为中间文件，增强健壮性
        """
        logger.info("开始提取话题时间区间...")
        
        if not outlines:
            logger.warning("大纲数据为空，无法提取时间线。")
            return []

        if not self.srt_chunks_dir.exists():
            logger.error(f"SRT块目录不存在: {self.srt_chunks_dir}。请先运行Step 1。")
            return []

        # 1. 创建本步骤需要的目录
        self.timeline_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.llm_raw_output_dir.mkdir(parents=True, exist_ok=True)

        # 2. 按 chunk_index 对所有大纲进行分组
        outlines_by_chunk = defaultdict(list)
        for outline in outlines:
            chunk_index = outline.get('chunk_index')
            if chunk_index is not None:
                outlines_by_chunk[chunk_index].append(outline)
            else:
                logger.warning(f"  > 话题 '{outline.get('title', '未知')}' 缺少 chunk_index，将被跳过。")

        all_timeline_data = []
        # 3. 遍历每个块，批量处理，并将结果存为独立的JSON文件
        for chunk_index, chunk_outlines in outlines_by_chunk.items():
            logger.info(f"处理块 {chunk_index}，其中包含 {len(chunk_outlines)} 个话题...")
            
            # 每次都重新处理，不使用缓存
            chunk_output_path = self.timeline_chunks_dir / f"chunk_{chunk_index}.json"

            try:
                # 首先加载对应的SRT块文件，无论是否使用缓存都需要这些信息
                srt_chunk_path = self.srt_chunks_dir / f"chunk_{chunk_index}.json"
                if not srt_chunk_path.exists():
                    logger.warning(f"  > 找不到对应的SRT块文件: {srt_chunk_path}，跳过整个块。")
                    continue
                
                with open(srt_chunk_path, 'r', encoding='utf-8') as f:
                    srt_chunk_data = json.load(f)

                if not srt_chunk_data:
                    logger.warning(f"  > SRT块文件为空: {srt_chunk_path}，跳过整个块。")
                    continue

                # 获取时间范围信息
                chunk_start_time = srt_chunk_data[0]['start_time']
                chunk_end_time = srt_chunk_data[-1]['end_time']

                if len(chunk_outlines) > 4:
                    chunk_items = self._extract_large_chunk_in_batches(
                        chunk_index,
                        chunk_outlines,
                        srt_chunk_data,
                        chunk_start_time,
                        chunk_end_time,
                    )
                    if chunk_items:
                        with open(chunk_output_path, 'w', encoding='utf-8') as f:
                            json.dump(chunk_items, f, ensure_ascii=False, indent=2)
                        logger.info(f"  > Chunk {chunk_index} parsed {len(chunk_items)} timeline items in batches")
                    else:
                        logger.warning(f"  > Chunk {chunk_index} did not produce timeline items in batches")
                    continue

                raw_response = ""
                llm_cache_path = self.llm_raw_output_dir / f"chunk_{chunk_index}.txt"

                if llm_cache_path.exists():
                    logger.info(f"  > 找到块 {chunk_index} 的LLM原始响应缓存，直接读取。")
                    with open(llm_cache_path, 'r', encoding='utf-8') as f:
                        raw_response = f.read()
                else:
                    logger.info(f"  > 未找到LLM缓存，开始调用API...")
                    
                    # 构建用于LLM的SRT文本
                    srt_text_for_prompt = ""
                    for sub in srt_chunk_data:
                        srt_text_for_prompt += f"{sub['index']}\\n{sub['start_time']} --> {sub['end_time']}\\n{sub['text']}\\n\\n"
                    
                    # 为LLM准备一个"干净"的输入，只包含它需要的信息
                    llm_input_outlines = [
                        {"title": o.get("title"), "subtopics": o.get("subtopics")}
                        for o in chunk_outlines
                    ]

                    input_data = {
                        "outline": llm_input_outlines,  # 使用干净的数据
                        "srt_text": srt_text_for_prompt
                    }
                    
                    # 调用LLM获取原始响应，带重试机制
                    parsed_items = None
                    max_parse_retries = 2
                    
                    for retry_count in range(max_parse_retries + 1):
                        try:
                            raw_response = self.llm_client.call_with_retry(self.timeline_prompt, input_data)
                            
                            if not raw_response:
                                logger.warning(f"  > 块 {chunk_index} LLM响应为空，跳过")
                                break
                            
                            # 保存原始响应到缓存
                            cache_file = self.llm_raw_output_dir / f"chunk_{chunk_index}_attempt_{retry_count}.txt"
                            with open(cache_file, 'w', encoding='utf-8') as f:
                                f.write(raw_response)
                            
                            # 解析LLM的原始响应
                            parsed_items = self._parse_and_validate_response(
                                raw_response, 
                                chunk_start_time, 
                                chunk_end_time,
                                chunk_index
                            )
                            
                            if parsed_items:
                                parsed_items = self._apply_product_completion_guard(
                                    parsed_items,
                                    chunk_outlines,
                                    srt_chunk_data,
                                )
                                parsed_items = self._add_product_prelude_clips(
                                    parsed_items,
                                    chunk_outlines,
                                    srt_chunk_data,
                                )
                                # 保存解析后的结果
                                with open(chunk_output_path, 'w', encoding='utf-8') as f:
                                    json.dump(parsed_items, f, ensure_ascii=False, indent=2)
                                
                                logger.info(f"  > 块 {chunk_index} 成功解析 {len(parsed_items)} 个时间段")
                                break  # 成功解析，跳出重试循环
                            else:
                                if retry_count < max_parse_retries:
                                    logger.warning(f"  > 块 {chunk_index} 解析失败，尝试重试 ({retry_count + 1}/{max_parse_retries + 1})")
                                    # 在重试时强化提示词，强调JSON格式
                                    input_data['additional_instruction'] = "\n\n【重要】输出要求：\n1. 必须以[开始，以]结束\n2. 使用英文双引号，不要使用中文引号\n3. 字符串中的引号必须转义为\\\"\n4. 不要添加任何解释文字或代码块标记\n5. 确保JSON格式完全正确"
                                else:
                                    logger.error(f"  > 块 {chunk_index} 经过 {max_parse_retries + 1} 次尝试仍然解析失败")
                                    # 保存最后一次的原始响应以便调试
                                    self._save_debug_response(raw_response, chunk_index, "final_parse_failure")
                                    
                        except Exception as parse_error:
                            logger.error(f"  > 块 {chunk_index} 第 {retry_count + 1} 次尝试解析过程中发生异常: {parse_error}")
                            if retry_count == max_parse_retries:
                                # 保存原始响应以便调试
                                self._save_debug_response(raw_response if 'raw_response' in locals() else "No response", chunk_index, "parse_exception")
                            continue
                    
                    if not parsed_items:
                         logger.warning(f"  > 块 {chunk_index} 最终解析失败，跳过")
                         continue

            except Exception as e:
                logger.error(f"  > 处理块 {chunk_index} 时出错: {str(e)}")
                continue
        
        # 4. 从所有中间文件中拼接最终结果
        logger.info("所有块处理完毕，开始从中间文件拼接最终结果...")
        all_timeline_data = []
        chunk_files = sorted(self.timeline_chunks_dir.glob("*.json"))
        for chunk_file in chunk_files:
            with open(chunk_file, 'r', encoding='utf-8') as f:
                chunk_data = json.load(f)
                all_timeline_data.extend(chunk_data)

        logger.info(f"成功从 {len(chunk_files)} 个块文件中加载了 {len(all_timeline_data)} 个话题。")

        all_timeline_data = enrich_product_logic_clips(
            all_timeline_data,
            outlines,
            self.srt_chunks_dir,
            self.text_processor,
        )
        all_timeline_data = merge_cross_chunk_product_clips(all_timeline_data)
        all_timeline_data = dedupe_clips_by_time(all_timeline_data, "step2_timeline")
        
        # 最终排序：在返回所有结果前，按开始时间进行全局排序
        if all_timeline_data:
            logger.info("按开始时间对所有话题进行最终排序...")
            try:
                # 使用 text_processor 将时间字符串转换为秒数以便正确排序
                all_timeline_data.sort(key=lambda x: self.text_processor.time_to_seconds(x['start_time']))
                logger.info("排序完成。")
                
                # 为所有片段按时间顺序分配固定的ID
                logger.info("为所有片段按时间顺序分配固定ID...")
                for i, timeline_item in enumerate(all_timeline_data):
                    timeline_item['id'] = str(i + 1)
                logger.info(f"已为 {len(all_timeline_data)} 个片段分配了固定ID（1-{len(all_timeline_data)}）")
                
            except Exception as e:
                logger.error(f"对最终结果排序时出错: {e}。返回未排序的结果。")

        return all_timeline_data
        
    def _extract_large_chunk_in_batches(
        self,
        chunk_index: int,
        chunk_outlines: List[Dict],
        srt_chunk_data: List[Dict],
        chunk_start_time: str,
        chunk_end_time: str,
    ) -> List[Dict]:
        batch_size = 1
        max_parse_retries = 2
        total_batches = (len(chunk_outlines) + batch_size - 1) // batch_size
        tasks = []
        for batch_start in range(0, len(chunk_outlines), batch_size):
            batch_index = batch_start // batch_size
            batch_outlines = chunk_outlines[batch_start:batch_start + batch_size]
            tasks.append((batch_index, batch_outlines))

        def process_batch(task) -> List[Dict]:
            batch_index, batch_outlines = task
            batch_number = batch_index + 1
            llm_cache_path = self.llm_raw_output_dir / f"chunk_{chunk_index}_batch_{batch_index}.txt"
            raw_response = ""

            logger.info(
                f"  > Processing chunk {chunk_index} batch {batch_number}, "
                f"{len(batch_outlines)} outlines"
            )
            if llm_cache_path.exists():
                with open(llm_cache_path, 'r', encoding='utf-8') as f:
                    raw_response = f.read()

                parsed_items = self._parse_and_validate_response(
                    raw_response,
                    chunk_start_time,
                    chunk_end_time,
                    chunk_index
                )
                if parsed_items:
                    parsed_items = self._apply_product_completion_guard(
                        parsed_items,
                        batch_outlines,
                        srt_chunk_data,
                    )
                    parsed_items = self._add_product_prelude_clips(
                        parsed_items,
                        batch_outlines,
                        srt_chunk_data,
                    )
                    return parsed_items
                return self._fallback_timeline_from_outlines(
                    batch_outlines,
                    srt_chunk_data,
                    chunk_index,
                    chunk_start_time,
                    chunk_end_time,
                )

            llm_input_outlines = [
                {
                    "title": outline.get("title"),
                    "subtopics": outline.get("subtopics"),
                    "product": outline.get("product"),
                    "summary": outline.get("summary"),
                    "reason": outline.get("reason"),
                    "start_time_hint": outline.get("start_time"),
                    "end_time_hint": outline.get("end_time"),
                }
                for outline in batch_outlines
            ]

            srt_text_for_prompt = self._build_srt_text_for_outline_batch(
                srt_chunk_data,
                batch_outlines,
            )

            input_data = {
                "outline": llm_input_outlines,
                "srt_text": srt_text_for_prompt
            }

            parsed_items = None
            for retry_count in range(max_parse_retries + 1):
                try:
                    raw_response = self.llm_client.call_with_retry(self.timeline_prompt, input_data)

                    if not raw_response:
                        logger.warning(f"  > Chunk {chunk_index} batch {batch_number} LLM response is empty.")
                        break

                    attempt_cache_path = self.llm_raw_output_dir / (
                        f"chunk_{chunk_index}_batch_{batch_index}_attempt_{retry_count}.txt"
                    )
                    with open(attempt_cache_path, 'w', encoding='utf-8') as f:
                        f.write(raw_response)
                    with open(llm_cache_path, 'w', encoding='utf-8') as f:
                        f.write(raw_response)

                    parsed_items = self._parse_and_validate_response(
                        raw_response,
                        chunk_start_time,
                        chunk_end_time,
                        chunk_index
                    )

                    if parsed_items:
                        parsed_items = self._apply_product_completion_guard(
                            parsed_items,
                            batch_outlines,
                            srt_chunk_data,
                        )
                        parsed_items = self._add_product_prelude_clips(
                            parsed_items,
                            batch_outlines,
                            srt_chunk_data,
                        )
                        logger.info(
                            f"  > Chunk {chunk_index} batch {batch_number} parsed "
                            f"{len(parsed_items)} timeline items"
                        )
                        return parsed_items

                    if retry_count < max_parse_retries:
                        logger.warning(
                            f"  > Chunk {chunk_index} batch {batch_number} parse failed, "
                            f"retrying ({retry_count + 1}/{max_parse_retries + 1})"
                        )
                        input_data['additional_instruction'] = (
                            "\n\nIMPORTANT output requirements:\n"
                            "1. Start with [ and end with ]\n"
                            "2. Use English double quotes\n"
                            "3. Escape quotes inside strings as \\\"\n"
                            "4. Do not add explanations or markdown code fences\n"
                            "5. Return valid JSON only"
                        )
                    else:
                        logger.error(
                            f"  > Chunk {chunk_index} batch {batch_number} failed after "
                            f"{max_parse_retries + 1} attempts"
                        )
                        self._save_debug_response(
                            raw_response,
                            chunk_index,
                            f"batch_{batch_index}_final_parse_failure"
                        )

                except Exception as parse_error:
                    logger.error(
                        f"  > Chunk {chunk_index} batch {batch_number} attempt "
                        f"{retry_count + 1} failed: {parse_error}"
                    )
                    if retry_count == max_parse_retries:
                        self._save_debug_response(
                            raw_response if 'raw_response' in locals() else "No response",
                            chunk_index,
                            f"batch_{batch_index}_parse_exception"
                        )
                    continue
            return self._fallback_timeline_from_outlines(
                batch_outlines,
                srt_chunk_data,
                chunk_index,
                chunk_start_time,
                chunk_end_time,
            )

        logger.info(
            "Step2 chunk %s using bounded LLM concurrency=%s for %s batches",
            chunk_index,
            get_llm_concurrency(),
            total_batches,
        )

        def on_completed(completed: int, total: int) -> None:
            self._emit_timeline_progress(
                completed,
                total,
                f"并行定位时间线（并发{min(get_llm_concurrency(), total)}）",
            )

        batch_results = run_parallel_ordered(tasks, process_batch, on_completed)
        chunk_items = [item for batch in batch_results for item in batch]
        self._save_partial_chunk(chunk_index, chunk_items)
        return chunk_items

    def _fallback_timeline_from_outlines(
        self,
        outlines: List[Dict],
        srt_chunk_data: List[Dict],
        chunk_index: int,
        chunk_start_time: str,
        chunk_end_time: str,
    ) -> List[Dict]:
        """Keep valid Step1 ranges when a Step2 response cannot be parsed."""
        def srt_time(value: float) -> str:
            value = max(0.0, float(value))
            whole = int(value)
            milliseconds = int(round((value - whole) * 1000))
            if milliseconds == 1000:
                whole += 1
                milliseconds = 0
            hours, remainder = divmod(whole, 3600)
            minutes, seconds = divmod(remainder, 60)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

        try:
            chunk_start_seconds = self.text_processor.time_to_seconds(chunk_start_time)
            chunk_end_seconds = self.text_processor.time_to_seconds(chunk_end_time)
        except Exception:
            return []

        fallback_items: List[Dict] = []
        for outline in outlines:
            start_value = outline.get("start_time")
            end_value = outline.get("end_time")
            if not start_value or not end_value:
                continue
            try:
                start_time = self._convert_time_format(str(start_value))
                end_time = self._convert_time_format(str(end_value))
                start_seconds = max(
                    chunk_start_seconds,
                    self.text_processor.time_to_seconds(start_time),
                )
                end_seconds = min(
                    chunk_end_seconds,
                    self.text_processor.time_to_seconds(end_time),
                )
            except Exception:
                continue
            if end_seconds <= start_seconds:
                continue

            start_time = srt_time(start_seconds)
            end_time = srt_time(end_seconds)

            content_lines = []
            for subtitle in srt_chunk_data:
                try:
                    subtitle_start = self.text_processor.time_to_seconds(subtitle["start_time"])
                    subtitle_end = self.text_processor.time_to_seconds(subtitle["end_time"])
                except Exception:
                    continue
                if subtitle_start <= end_seconds and subtitle_end >= start_seconds:
                    text = str(subtitle.get("text") or "").strip()
                    if text:
                        content_lines.append(text)

            fallback_item = {
                "outline": outline.get("title") or outline.get("topic") or outline.get("product") or "产品介绍",
                "product": outline.get("product"),
                "summary": outline.get("summary"),
                "reason": outline.get("reason"),
                "start_time": start_time,
                "end_time": end_time,
                "content": " ".join(content_lines),
                "chunk_index": chunk_index,
                "timeline_source": "outline_fallback",
                "cut_reason": "时间线模型响应解析失败，使用带时间戳大纲保留该产品候选。",
            }
            fallback_items.append(fallback_item)
            logger.warning(
                "Step2 fallback kept outline range: %s (%s -> %s)",
                fallback_item["outline"],
                start_time,
                end_time,
            )

        return fallback_items

    def _apply_product_completion_guard(
        self,
        parsed_items: List[Dict],
        outlines: List[Dict],
        srt_chunk_data: List[Dict],
    ) -> List[Dict]:
        """Keep product explanations from being cut too early.

        Live product streams often describe the product first, then continue
        with specs, usage, price, gifts, and purchase guidance. The model may
        correctly explain that the full range should be kept, but still emit a
        JSON end time that stops at the first selling point. Extend product clips
        toward the outline hint plus a small closing grace window when later
        subtitles still look like the same product explanation.
        """
        if not parsed_items or not outlines or not srt_chunk_data:
            return parsed_items

        suite_words = ("礼盒", "套装", "套餐", "组合")
        product_words = (
            "产品", "茶", "果茶", "乌龙", "花茶", "首饰", "项链", "手链", "耳钉",
            "戒指", "茶宠", "兔兔", "礼盒", "套装", "套餐", "组合", "杯垫",
        )
        closing_words = (
            "价格", "链接", "拍下", "带回家", "赠", "送", "杯垫", "茶勺",
            "黄冰糖", "粉丝团", "会员", "188", "309", "四罐", "小圆罐",
            "门店", "直播间", "到手", "下单", "拍", "买", "换", "送礼",
            "适合", "老人", "小孩", "女生", "冷泡", "热泡", "口味", "味道",
            "香", "入口", "配料", "原料", "工艺", "包装", "设计", "材质",
        )
        hard_stop_words = (
            "下一个", "另外", "接下来", "再看", "我们换", "上车吧", "小助理",
        )
        suite_continuation_words = (
            "还有", "最后一个", "最后一款", "第一款", "第二款", "第三款",
            "第四款", "其中", "里面", "内含", "包含",
        )
        suite_section_stop_words = (
            "块钱", "元钱", "原价", "现价", "直播价", "到手价",
            "几号链接", "上链接", "赠送", "额外送",
        )

        def text_of(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, (list, tuple)):
                return " ".join(text_of(item) for item in value)
            if isinstance(value, dict):
                return " ".join(text_of(item) for item in value.values())
            return str(value)

        def seconds(value: Any) -> Optional[float]:
            if not value:
                return None
            try:
                return self.text_processor.time_to_seconds(self._convert_time_format(str(value)))
            except Exception:
                return None

        def srt_time(value: float) -> str:
            value = max(0.0, float(value))
            whole = int(value)
            milliseconds = int(round((value - whole) * 1000))
            if milliseconds >= 1000:
                whole += 1
                milliseconds -= 1000
            hours = whole // 3600
            minutes = (whole % 3600) // 60
            secs = whole % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

        outline_by_product: Dict[str, Dict] = {}
        product_outlines: List[Dict] = []
        for outline in outlines:
            combined = text_of(outline)
            if not any(word in combined for word in product_words):
                continue
            product_outlines.append(outline)
            product = str(outline.get("product") or "").strip()
            if product:
                outline_by_product[product] = outline

        if not product_outlines:
            return parsed_items

        subtitles = []
        for sub in srt_chunk_data:
            start = seconds(sub.get("start_time"))
            end = seconds(sub.get("end_time"))
            if start is None or end is None:
                continue
            subtitles.append((start, end, str(sub.get("text") or "").strip()))
        subtitles.sort(key=lambda entry: (entry[0], entry[1]))

        adjusted_items = []
        for item in parsed_items:
            item_text = text_of(item)
            product = str(item.get("product") or "").strip()
            is_suite = any(word in item_text for word in suite_words)
            if not product and not any(word in item_text for word in product_words):
                adjusted_items.append(item)
                continue

            outline = outline_by_product.get(product)
            if outline is None:
                outline = next((candidate for candidate in product_outlines if product and product in text_of(candidate)), None)
            if outline is None:
                outline = product_outlines[0] if len(product_outlines) == 1 else None
            if outline is None:
                adjusted_items.append(item)
                continue

            start_sec = seconds(item.get("start_time"))
            end_sec = seconds(item.get("end_time"))
            hint_end = seconds(outline.get("end_time"))
            if start_sec is None or end_sec is None or hint_end is None:
                adjusted_items.append(item)
                continue

            # A short grace window catches final price/link sentences that often
            # land just after the outline hint in live streams. Suites can be
            # longer because they naturally contain child products.
            max_duration = 150 if is_suite else 110
            grace = 12 if is_suite else 6
            target_limit = min(hint_end + grace, start_sec + max_duration)
            target_end = end_sec

            # A suite/gift box may introduce its child products one after
            # another. Do not stop at the end hint when the subtitles explicitly
            # say that another item in the same suite follows.
            if is_suite:
                continuation_limit = min(end_sec + 45, start_sec + max_duration)
                saw_continuation = False
                previous_end = end_sec
                for sub_start, sub_end, sub_text in subtitles:
                    if sub_end <= end_sec:
                        continue
                    if sub_start > continuation_limit or sub_start - previous_end > 5:
                        break
                    has_continuation = any(word in sub_text for word in suite_continuation_words)
                    if any(word in sub_text for word in hard_stop_words) and not has_continuation:
                        break
                    if saw_continuation and any(word in sub_text for word in suite_section_stop_words):
                        break
                    if has_continuation:
                        saw_continuation = True
                    if saw_continuation:
                        target_end = max(target_end, sub_end)
                    previous_end = sub_end

                if saw_continuation:
                    target_limit = max(target_limit, target_end)

            for sub_start, sub_end, sub_text in subtitles:
                if sub_start < end_sec or sub_start > hint_end + 45:
                    continue
                if any(word in sub_text for word in hard_stop_words) and sub_start > hint_end + grace:
                    break
                if any(word in sub_text for word in closing_words):
                    target_end = max(target_end, min(sub_end, target_limit))

            if target_end <= end_sec + 3:
                adjusted_items.append(item)
                continue

            content_lines = [
                sub_text
                for sub_start, sub_end, sub_text in subtitles
                if sub_text and sub_end >= start_sec and sub_start < target_end
            ]
            adjusted = dict(item)
            adjusted["end_time"] = srt_time(target_end)
            adjusted["duration"] = round(target_end - start_sec, 2)
            adjusted["duration_seconds"] = round(target_end - start_sec, 2)
            if content_lines:
                adjusted["content"] = " ".join(content_lines)
            adjusted["product_completion_guard"] = True
            adjusted["cut_reason"] = (
                f"{adjusted.get('cut_reason', '')} "
                "产品完整度兜底：该片段后续仍包含规格、价格、赠品、场景或购买信息，已延展到更完整的成交闭环。"
            ).strip()
            logger.info(
                "Product completion guard extended %s from %s to %s",
                product or adjusted.get("outline"),
                item.get("end_time"),
                adjusted["end_time"],
            )
            adjusted_items.append(adjusted)

        return adjusted_items

    def _add_product_prelude_clips(
        self,
        parsed_items: List[Dict],
        outlines: List[Dict],
        srt_chunk_data: List[Dict],
    ) -> List[Dict]:
        """Recover early product-intro segments that the model may skip.

        Product livestreams often introduce packaging, colors, design, origin,
        or craft before the later price/gift pitch. If the model only selects
        the later pitch, keep the earlier product explanation as a separate
        clip instead of merging through noisy interaction.
        """
        if not parsed_items or not outlines or not srt_chunk_data:
            return parsed_items

        def text_of(value: Any) -> str:
            if value is None:
                return ""
            if isinstance(value, (list, tuple)):
                return " ".join(text_of(item) for item in value)
            if isinstance(value, dict):
                return " ".join(text_of(item) for item in value.values())
            return str(value)

        def seconds(value: Any) -> Optional[float]:
            if not value:
                return None
            try:
                return self.text_processor.time_to_seconds(self._convert_time_format(str(value)))
            except Exception:
                return None

        def srt_time(value: float) -> str:
            value = max(0.0, float(value))
            whole = int(value)
            milliseconds = int(round((value - whole) * 1000))
            if milliseconds >= 1000:
                whole += 1
                milliseconds -= 1000
            hours = whole // 3600
            minutes = (whole % 3600) // 60
            secs = whole % 60
            return f"{hours:02d}:{minutes:02d}:{secs:02d},{milliseconds:03d}"

        def product_aliases(product: str) -> List[str]:
            compact = re.sub(r"[\\s（）()·・\\-_/]+", "", product or "")
            compact = re.sub(r"[（(].*?[）)]", "", compact)
            aliases = {compact}
            aliases.add(compact.replace("的", ""))
            for suffix in ("礼盒", "套装", "组合", "茶", "花茶", "乌龙茶", "果茶", "耳钉", "项链", "手链", "茶宠", "首饰盒"):
                if compact.endswith(suffix) and len(compact) > len(suffix) + 1:
                    root = compact[: -len(suffix)]
                    aliases.add(root)
                    aliases.add(f"{root}的{suffix}")
            for sep in ("（", "(", "·", "-", "_", "/"):
                if sep in product:
                    head = product.split(sep, 1)[0].strip()
                    if len(head) >= 2:
                        aliases.add(head)
            return [alias for alias in aliases if len(alias) >= 2]

        def product_matches(product: str, text: str) -> bool:
            compact_text = re.sub(r"[\\s（）()·・\\-_/]+", "", text or "")
            return any(alias and alias in compact_text for alias in product_aliases(product))

        setup_words = (
            "介绍", "包装", "颜色", "设计", "原创", "插画", "礼盒", "套装", "材质", "工艺",
            "进口", "冻干", "原料", "配料", "成分", "里面", "外面", "口味", "味道", "香",
            "适合", "风味", "限定", "高级", "好看", "特点", "卖点", "核心", "重点", "产品",
        )
        interaction_words = (
            "吗", "可以的", "要不", "备注", "拍下", "上车", "送你", "给你送", "小助理",
            "链接", "库存", "还有没有", "能不能", "换", "退", "订单",
        )

        subtitles = []
        for sub in srt_chunk_data:
            start = seconds(sub.get("start_time"))
            end = seconds(sub.get("end_time"))
            text = str(sub.get("text") or "").strip()
            if start is None or end is None or not text:
                continue
            subtitles.append((start, end, text))

        if not subtitles:
            return parsed_items

        outline_by_product: Dict[str, Dict] = {}
        for outline in outlines:
            product = str(outline.get("product") or "").strip()
            if product:
                outline_by_product[product] = outline

        existing_ranges = []
        for item in parsed_items:
            start = seconds(item.get("start_time"))
            end = seconds(item.get("end_time"))
            product = str(item.get("product") or "").strip()
            if start is not None and end is not None:
                existing_ranges.append((product, start, end))

        additions: List[Dict] = []
        for item in parsed_items:
            product = str(item.get("product") or "").strip()
            if not product:
                continue

            item_start = seconds(item.get("start_time"))
            if item_start is None or item_start < 20:
                continue

            outline = outline_by_product.get(product)
            outline_start = seconds(outline.get("start_time")) if outline else None
            window_end = item_start - 8
            window_start = max(0.0, min(value for value in (item_start, outline_start or item_start) if value is not None) - 180)
            candidates = [
                (start, end, text)
                for start, end, text in subtitles
                if start >= window_start and end <= window_end
            ]
            if not candidates:
                continue

            best_segment = None
            current = None
            for start, end, text in candidates:
                product_hit = product_matches(product, text)
                setup_hit = any(word in text for word in setup_words)
                if product_hit and current is None:
                    current = [start, end, [text], 1]
                    continue
                if current is None:
                    continue

                gap = start - current[1]
                has_setup = setup_hit
                has_interaction = any(word in text for word in interaction_words)
                duration = current[1] - current[0]

                if gap <= 6 and (has_setup or product_hit or (not has_interaction and duration < 45)):
                    current[1] = end
                    current[2].append(text)
                    if product_hit or has_setup:
                        current[3] += 1
                    continue

                if has_interaction and duration >= 15:
                    pass
                elif gap <= 10 and duration < 18:
                    current[1] = end
                    current[2].append(text)
                    continue

                if current and current[1] - current[0] >= 15 and current[3] >= 2:
                    best_segment = current
                current = [start, end, [text], 1] if product_hit else None

            if current and current[1] - current[0] >= 15 and current[3] >= 2:
                best_segment = current

            if not best_segment:
                continue

            start, end, lines, evidence_count = best_segment
            if end >= item_start - 5 or end - start < 15 or len("".join(lines)) < 40 or evidence_count < 2:
                continue
            overlaps_existing = any(
                product == existing_product and start < existing_end and end > existing_start
                for existing_product, existing_start, existing_end in existing_ranges
            )
            if overlaps_existing:
                continue

            prelude = dict(item)
            prelude["id"] = f"{item.get('id', '0')}-0"
            prelude["start_time"] = srt_time(start)
            prelude["end_time"] = srt_time(end)
            prelude["duration"] = round(end - start, 2)
            prelude["duration_seconds"] = round(end - start, 2)
            prelude["content"] = " ".join(lines)
            prelude["outline"] = f"{product} 包装/设计/工艺介绍"
            prelude["selling_point"] = "产品前置介绍：包装、颜色、设计、工艺或基础卖点"
            prelude["recommend_reason"] = "字幕中存在连续的产品基础介绍，适合作为独立产品介绍切片。"
            prelude["cut_reason"] = "产品前置介绍兜底：主切片前存在连续包装、颜色、设计、工艺等产品说明，已独立保留。"
            prelude["product_prelude_guard"] = True
            additions.append(prelude)
            existing_ranges.append((product, start, end))
            logger.info(
                "Product prelude guard added %s from %s to %s",
                product,
                prelude["start_time"],
                prelude["end_time"],
            )

        if not additions:
            return parsed_items

        combined = parsed_items + additions
        combined.sort(key=lambda clip: seconds(clip.get("start_time")) or 0)
        return combined

    def _save_partial_chunk(self, chunk_index: int, chunk_items: List[Dict]) -> None:
        if not chunk_items:
            return
        try:
            partial_path = self.timeline_chunks_dir / f"chunk_{chunk_index}.partial.json"
            with open(partial_path, "w", encoding="utf-8") as f:
                json.dump(chunk_items, f, ensure_ascii=False, indent=2)
        except Exception as partial_error:
            logger.debug(f"保存时间线分批断点失败: {partial_error}")

    def _emit_timeline_progress(self, batch_number: int, total_batches: int, action: str) -> None:
        if not self.progress_callback or total_batches <= 0:
            return
        try:
            completed_fraction = min(batch_number, total_batches) / total_batches
            subpercent = 10 + completed_fraction * 35
            self.progress_callback(
                "ANALYZE",
                f"{action} {batch_number}/{total_batches} 批",
                subpercent=subpercent,
            )
        except Exception as progress_error:
            logger.debug(f"更新时间线批次进度失败: {progress_error}")

    def _build_srt_text_for_outline_batch(self, srt_chunk_data: List[Dict], outlines: List[Dict]) -> str:
        hint_ranges = []
        for outline in outlines:
            start_hint = outline.get("start_time")
            end_hint = outline.get("end_time")
            if not start_hint or not end_hint:
                continue
            try:
                start_seconds = self.text_processor.time_to_seconds(self._convert_time_format(start_hint))
                end_seconds = self.text_processor.time_to_seconds(self._convert_time_format(end_hint))
                if end_seconds > start_seconds:
                    hint_ranges.append((start_seconds, end_seconds))
            except Exception:
                continue

        selected_subtitles = srt_chunk_data
        if hint_ranges:
            # Step 1 time hints can drift in live streams because product talk is interrupted
            # by replies and benefits. Keep a wider window so later product details are not cut.
            window_padding_seconds = 360
            window_start = max(0, min(start for start, _ in hint_ranges) - window_padding_seconds)
            window_end = max(end for _, end in hint_ranges) + window_padding_seconds
            selected_subtitles = []
            for sub in srt_chunk_data:
                try:
                    sub_start = self.text_processor.time_to_seconds(self._convert_time_format(sub["start_time"]))
                    sub_end = self.text_processor.time_to_seconds(self._convert_time_format(sub["end_time"]))
                except Exception:
                    continue
                if sub_start <= window_end and sub_end >= window_start:
                    selected_subtitles.append(sub)

            if not selected_subtitles:
                selected_subtitles = srt_chunk_data

            logger.info(
                f"  > Batch subtitle window: {window_start:.1f}s-{window_end:.1f}s, "
                f"{len(selected_subtitles)}/{len(srt_chunk_data)} subtitle lines"
            )

        srt_text_for_prompt = ""
        for sub in selected_subtitles:
            srt_text_for_prompt += f"{sub['index']}\n{sub['start_time']} --> {sub['end_time']}\n{sub['text']}\n\n"
        return srt_text_for_prompt

    def _parse_and_validate_response(self, response: str, chunk_start: str, chunk_end: str, chunk_index: int) -> List[Dict]:
        """增强的解析LLM的批量响应、验证并调整时间"""
        validated_items = []
        
        # 保存原始响应用于调试
        self._save_debug_response(response, chunk_index, "original_response")
        
        try:
            # 尝试解析JSON
            parsed_response = self.llm_client.parse_json_response(response)

            parsed_response = self._normalize_timeline_response(parsed_response)

            if not isinstance(parsed_response, list):
                logger.warning(f"  > 块 {chunk_index} LLM返回的不是一个列表")
                self._save_debug_response(f"类型: {type(parsed_response)}, 内容: {parsed_response}", chunk_index, "not_list")
                return []

            # 验证JSON结构
            if not self.llm_client._validate_json_structure(parsed_response):
                logger.error(f"  > 块 {chunk_index} JSON结构验证失败")
                self._save_debug_response(str(parsed_response), chunk_index, "invalid_structure")
                return []
            
            for timeline_item in parsed_response:
                if 'outline' not in timeline_item or 'start_time' not in timeline_item or 'end_time' not in timeline_item:
                    logger.warning(f"  > 从LLM返回的某个JSON对象格式不正确: {timeline_item}")
                    continue
                
                # 将 chunk_index 添加回对象中，以便后续步骤使用
                timeline_item['chunk_index'] = chunk_index
                
                # 验证和调整时间范围
                try:
                    # 验证时间格式
                    if not self._validate_time_format(timeline_item['start_time']):
                        logger.warning(f"  > 话题 '{timeline_item['outline']}' 开始时间格式不正确: {timeline_item['start_time']}")
                        continue
                    
                    if not self._validate_time_format(timeline_item['end_time']):
                        logger.warning(f"  > 话题 '{timeline_item['outline']}' 结束时间格式不正确: {timeline_item['end_time']}")
                        continue
                    
                    start_time = self._convert_time_format(timeline_item['start_time'])
                    end_time = self._convert_time_format(timeline_item['end_time'])
                    
                    start_sec = self.text_processor.time_to_seconds(start_time)
                    end_sec = self.text_processor.time_to_seconds(end_time)
                    chunk_start_sec = self.text_processor.time_to_seconds(chunk_start)
                    chunk_end_sec = self.text_processor.time_to_seconds(chunk_end)
                    
                    if start_sec < chunk_start_sec:
                        logger.warning(f"  > 调整话题 '{timeline_item['outline']}' 的开始时间从 {start_time} 到 {chunk_start}")
                        timeline_item['start_time'] = chunk_start
                    
                    if end_sec > chunk_end_sec:
                        logger.warning(f"  > 调整话题 '{timeline_item['outline']}' 的结束时间从 {end_time} 到 {chunk_end}")
                        timeline_item['end_time'] = chunk_end
                    
                    logger.info(f"  > 定位成功: {timeline_item['outline']} ({timeline_item['start_time']} -> {timeline_item['end_time']})")
                    validated_items.append(timeline_item)
                except Exception as e:
                    logger.error(f"  > 验证单个时间戳时出错: {e} - 项目: {timeline_item}")
                    continue
            
            return validated_items

        except Exception as e:
            logger.error(f"  > 块 {chunk_index} 解析LLM响应时出错: {e}")
            # 保存详细的错误信息
            error_info = {
                "error": str(e),
                "error_type": type(e).__name__,
                "response_length": len(response),
                "response_preview": response[:200],
                "chunk_index": chunk_index,
                "chunk_start": chunk_start,
                "chunk_end": chunk_end
            }
            import json
            self._save_debug_response(json.dumps(error_info, indent=2, ensure_ascii=False), chunk_index, "parse_error")
            return []

    def _normalize_timeline_response(self, parsed_response: Any) -> Any:
        """兼容产品切片提示词返回的字段名，转换为旧管线需要的 outline/content 结构。"""
        if isinstance(parsed_response, dict):
            for key in ("timeline", "items", "clips", "outline", "outlines"):
                value = parsed_response.get(key)
                if isinstance(value, list):
                    parsed_response = value
                    break

        if not isinstance(parsed_response, list):
            return parsed_response

        normalized_items = []
        for index, item in enumerate(parsed_response, start=1):
            if not isinstance(item, dict):
                normalized_items.append(item)
                continue

            normalized = dict(item)
            if not normalized.get("outline"):
                outline = (
                    normalized.get("title")
                    or normalized.get("selling_point")
                    or normalized.get("product")
                    or normalized.get("content")
                    or f"产品切片 {index}"
                )
                normalized["outline"] = str(outline).strip()

            if not normalized.get("content"):
                content_parts = []
                for field in ("content", "selling_point", "cut_reason", "product"):
                    value = item.get(field)
                    if isinstance(value, str) and value.strip():
                        content_parts.append(value.strip())
                normalized["content"] = "\n".join(content_parts)

            if "duration" not in normalized and "duration_seconds" in normalized:
                normalized["duration"] = normalized["duration_seconds"]

            normalized_items.append(normalized)

        return normalized_items

    def _validate_time_format(self, time_str: str) -> bool:
        """
        验证时间格式是否正确 (HH:MM:SS,mmm)
        """
        pattern = r'^\d{2}:\d{2}:\d{2},\d{3}$'
        return bool(re.match(pattern, time_str))
    
    def _convert_time_format(self, time_str: str) -> str:
        """
        转换时间格式：SRT格式 -> FFmpeg格式
        """
        if not time_str or time_str == "end":
            return time_str
        return time_str.replace(',', '.')

    def _save_debug_response(self, response: str, chunk_index: int, error_type: str) -> None:
        """保存调试响应到文件"""
        try:
            debug_dir = self.metadata_dir / "debug_responses"
            debug_dir.mkdir(parents=True, exist_ok=True)
            debug_file = debug_dir / f"chunk_{chunk_index}_{error_type}.txt"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(response)
            logger.info(f"调试响应已保存到: {debug_file}")
        except Exception as e:
            logger.error(f"保存调试响应失败: {e}")

    def save_timeline(self, timeline_data: List[Dict], output_path: Optional[Path] = None) -> Path:
        """
        保存时间区间数据
        """
        if output_path is None:
            output_path = METADATA_DIR / "step2_timeline.json"
        
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(timeline_data, f, ensure_ascii=False, indent=2)
            
        logger.info(f"时间数据已保存到: {output_path}")
        return output_path

    def load_timeline(self, input_path: Path) -> List[Dict]:
        """
        从文件加载时间数据
        """
        with open(input_path, 'r', encoding='utf-8') as f:
            return json.load(f)

def run_step2_timeline(
    outline_path: Path,
    metadata_dir: Path = None,
    output_path: Optional[Path] = None,
    prompt_files: Dict = None,
    progress_callback=None,
) -> List[Dict]:
    """
    运行Step 2: 时间点提取
    """
    if metadata_dir is None:
        metadata_dir = METADATA_DIR
        
    extractor = TimelineExtractor(metadata_dir, prompt_files, progress_callback=progress_callback)
    
    # 加载大纲
    with open(outline_path, 'r', encoding='utf-8') as f:
        outlines = json.load(f)
        
    timeline_data = extractor.extract_timeline(outlines)
    
    # 保存结果
    if output_path is None:
        output_path = metadata_dir / "step2_timeline.json"
        
    extractor.save_timeline(timeline_data, output_path)
    
    return timeline_data
