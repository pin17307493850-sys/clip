"""
Step 6: 视频生成 - 根据聚类结果生成最终视频切片
"""
import json
import logging
import re
import shutil
import subprocess
from typing import Callable, List, Dict, Any, Optional
from pathlib import Path

# 导入依赖
from ..utils.video_processor import VideoProcessor
from ..utils.text_processor import TextProcessor
from ..utils.ffmpeg_utils import get_ffmpeg_path
from ..core.shared_config import METADATA_DIR, CLIPS_DIR, COLLECTIONS_DIR

logger = logging.getLogger(__name__)

ProgressCallback = Optional[Callable[[str, str, int], None]]
CLIP_PADDING_SECONDS = 2.0


def _seconds_to_srt_time(seconds: float) -> str:
    seconds = max(0.0, float(seconds))
    millis_total = int(round(seconds * 1000))
    hours, rem = divmod(millis_total, 3600000)
    minutes, rem = divmod(rem, 60000)
    whole_seconds, millis = divmod(rem, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d},{millis:03d}"


def _ffmpeg_subtitle_path(path: Path) -> str:
    value = path.resolve().as_posix()
    value = value.replace(":", r"\:")
    value = value.replace("'", r"\'")
    return value


def _ffmpeg_force_style(style: str) -> str:
    return style.replace(",", r"\,")


def _time_to_seconds(value: Any) -> float:
    if isinstance(value, (int, float)):
        return max(0.0, float(value))
    return max(0.0, TextProcessor.time_to_seconds(str(value)))


def _write_shifted_srt(source_srt: Path, output_srt: Path, start: float, end: float, fallback_text: str = "") -> bool:
    subtitles = TextProcessor.parse_srt(source_srt)
    duration = max(0.1, end - start)
    selected = []
    for sub in subtitles:
        try:
            sub_start = TextProcessor.time_to_seconds(sub["start_time"])
            sub_end = TextProcessor.time_to_seconds(sub["end_time"])
        except Exception:
            continue
        if sub_end <= start or sub_start >= end:
            continue
        shifted_start = max(0.0, sub_start - start)
        shifted_end = min(duration, sub_end - start)
        text = (sub.get("text") or "").strip()
        if text and shifted_end > shifted_start:
            selected.append((shifted_start, shifted_end, text))

    if not selected and fallback_text:
        selected.append((0.0, duration, fallback_text.strip()))
    if not selected:
        return False

    output_srt.parent.mkdir(parents=True, exist_ok=True)
    with open(output_srt, "w", encoding="utf-8") as f:
        for index, (sub_start, sub_end, text) in enumerate(selected, start=1):
            f.write(f"{index}\n")
            f.write(f"{_seconds_to_srt_time(sub_start)} --> {_seconds_to_srt_time(sub_end)}\n")
            f.write(f"{text}\n\n")
    return True


def _burn_subtitles_into_clip(clip_path: Path, source_srt: Path, clip: Dict[str, Any], work_dir: Path) -> bool:
    try:
        start = _time_to_seconds(clip.get("export_start_time", clip["start_time"]))
        end = _time_to_seconds(clip.get("export_end_time", clip["end_time"]))
    except Exception as exc:
        logger.warning("Subtitle burn skipped, clip time parse failed %s: %s", clip.get("id"), exc)
        return False

    subtitle_path = work_dir / f"{clip_path.stem}_burn.srt"
    fallback_content = clip.get("content", [])
    if isinstance(fallback_content, list):
        fallback_text = " ".join(str(item) for item in fallback_content if item)
    else:
        fallback_text = str(fallback_content or "")
    fallback_text = fallback_text or clip.get("summary", "") or clip.get("generated_title", "")
    if not _write_shifted_srt(source_srt, subtitle_path, start, end, fallback_text):
        logger.warning("Subtitle burn skipped, no subtitle text in clip range: %s", clip.get("id"))
        return False

    temp_output = clip_path.with_name(f"{clip_path.stem}_with_subtitles.tmp.mp4")
    style = _ffmpeg_force_style(
        "FontName=Microsoft YaHei,FontSize=18,"
        "PrimaryColour=&H00FFFFFF,OutlineColour=&H80000000,"
        "BorderStyle=1,Outline=2,Shadow=1,Alignment=2,MarginV=42"
    )
    subtitle_filter = f"subtitles=filename='{_ffmpeg_subtitle_path(subtitle_path)}':force_style='{style}'"
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-i", str(clip_path),
        "-vf", subtitle_filter,
        "-c:v", "libx264",
        "-preset", "veryfast",
        "-crf", "23",
        "-c:a", "copy",
        "-movflags", "+faststart",
        str(temp_output),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="ignore")
    if result.returncode != 0 or not temp_output.exists():
        logger.error("Subtitle burn failed %s: %s", clip_path, result.stderr)
        temp_output.unlink(missing_ok=True)
        return False

    shutil.move(str(temp_output), str(clip_path))
    logger.info("Subtitle burned into clip: %s", clip_path)
    return True

class VideoGenerator:
    """视频生成器"""
    
    def __init__(self, clips_dir: Optional[str] = None, collections_dir: Optional[str] = None, metadata_dir: Optional[str] = None):
        # 强制使用项目内专属目录，不使用全局目录作为后备
        if not clips_dir:
            raise ValueError("clips_dir 参数是必需的，不能使用全局路径")
        if not collections_dir:
            raise ValueError("collections_dir 参数是必需的，不能使用全局路径")
        
        self.clips_dir = Path(clips_dir)
        self.collections_dir = Path(collections_dir)
        self.metadata_dir = Path(metadata_dir) if metadata_dir else METADATA_DIR
        
        # 确保目录存在
        self.clips_dir.mkdir(parents=True, exist_ok=True)
        self.collections_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建VideoProcessor实例，强制使用项目内路径
        self.video_processor = VideoProcessor(clips_dir=str(self.clips_dir), collections_dir=str(self.collections_dir))
    
    def generate_clips(self, clips_with_titles: List[Dict], input_video: Path) -> List[Path]:
        """
        生成切片视频
        
        Args:
            clips_with_titles: 带标题的片段数据
            input_video: 输入视频路径
            
        Returns:
            生成的切片视频路径列表
        """
        logger.info("开始生成切片视频...")
        
        video_info = self.video_processor.get_video_info(input_video)
        video_duration = float(video_info.get("duration") or 0)

        # 准备切片数据
        clips_data = []
        for clip in clips_with_titles:
            start_seconds = _time_to_seconds(clip["start_time"])
            end_seconds = _time_to_seconds(clip["end_time"])
            export_start = max(0.0, start_seconds - CLIP_PADDING_SECONDS)
            export_end = end_seconds + CLIP_PADDING_SECONDS
            if video_duration > 0:
                export_end = min(video_duration, export_end)
            if export_end <= export_start:
                export_start, export_end = start_seconds, end_seconds

            clip["export_start_time"] = _seconds_to_srt_time(export_start)
            clip["export_end_time"] = _seconds_to_srt_time(export_end)
            clip["export_padding_seconds"] = CLIP_PADDING_SECONDS

            clips_data.append({
                'id': clip['id'],
                'title': clip.get('generated_title', f"片段_{clip['id']}"),
                'start_time': clip["export_start_time"],
                'end_time': clip["export_end_time"],
                'original_start_time': clip['start_time'],
                'original_end_time': clip['end_time'],
                'force': True,
            })
        
        # 批量生成切片
        successful_clips = self.video_processor.batch_extract_clips(input_video, clips_data)
        
        logger.info(f"切片视频生成完成，共{len(successful_clips)}个切片")
        return successful_clips
    
    def generate_collections(self, collections_data: List[Dict]) -> List[Dict]:
        """
        生成合集视频
        
        Args:
            collections_data: 合集数据
            
        Returns:
            生成的合集信息列表，包含视频路径和缩略图路径
        """
        logger.info("开始生成合集视频...")
        
        # 生成合集视频和缩略图
        successful_collections = self.video_processor.create_collections_from_metadata(collections_data)
        
        logger.info(f"合集视频生成完成，共{len(successful_collections)}个合集")
        return successful_collections
    
    def save_clip_metadata(self, clips_with_titles: List[Dict], output_path: Optional[Path] = None) -> Path:
        """
        保存最终的切片元数据到clips_metadata.json
        
        Args:
            clips_with_titles: 带标题的片段数据（来自step4）
            output_path: 输出路径，默认为clips_metadata.json
            
        Returns:
            保存的文件路径
            
        Note:
            此方法保存的是最终的切片元数据，包含视频生成后的完整信息。
            与step4的step4_titles.json不同，这里保存的是用于前端展示的最终数据。
        """
        if output_path is None:
            output_path = self.metadata_dir / "clips_metadata.json"
        
        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(clips_with_titles, f, ensure_ascii=False, indent=2)
        
        logger.info(f"切片元数据已保存到: {output_path}")
        return output_path
    
    def save_collection_metadata(self, collections_data: List[Dict], output_path: Optional[Path] = None) -> Path:
        """
        保存合集元数据
        
        Args:
            collections_data: 合集数据
            output_path: 输出路径
            
        Returns:
            保存的文件路径
        """
        if output_path is None:
            output_path = self.metadata_dir / "collections_metadata.json"
        
        # 确保目录存在
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # 保存数据
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(collections_data, f, ensure_ascii=False, indent=2)
        
        logger.info(f"合集元数据已保存到: {output_path}")
        return output_path

def run_step6_video(clips_with_titles_path: Path, collections_path: Path, 
                   input_video: Path, output_dir: Optional[Path] = None, 
                   clips_dir: Optional[str] = None, collections_dir: Optional[str] = None, 
                   metadata_dir: Optional[str] = None, burn_subtitles: bool = False,
                   generate_collections: bool = True,
                   progress_callback: ProgressCallback = None) -> Dict:
    """
    运行Step 6: 视频切割
    
    Args:
        clips_with_titles_path: 带标题的片段文件路径
        collections_path: 合集文件路径
        input_video: 输入视频路径
        output_dir: 输出目录
        
    Returns:
        生成结果信息
    """
    # 加载数据
    with open(clips_with_titles_path, 'r', encoding='utf-8') as f:
        clips_with_titles = json.load(f)
    
    collections_data = []
    if collections_path and collections_path.exists() and collections_path.stat().st_size > 0:
        try:
            with open(collections_path, 'r', encoding='utf-8') as f:
                collections_data = json.load(f)
        except Exception as exc:
            logger.warning("Collection metadata is invalid, continuing without collections: %s", exc)
    
    # 创建视频生成器
    generator = VideoGenerator(clips_dir=clips_dir, collections_dir=collections_dir, metadata_dir=metadata_dir)
    
    # 生成切片视频
    if progress_callback:
        progress_callback("EXPORT", f"正在导出切片视频 0/{len(clips_with_titles)}", 20)
    successful_clips = generator.generate_clips(clips_with_titles, input_video)
    if progress_callback:
        progress_callback("EXPORT", f"切片视频导出完成 {len(successful_clips)}/{len(clips_with_titles)}", 45)

    if burn_subtitles and metadata_dir:
        source_srt = Path(metadata_dir) / "input.srt"
        if source_srt.exists():
            clip_paths_by_id = {path.name.split("_", 1)[0]: path for path in successful_clips}
            subtitle_work_dir = Path(metadata_dir) / "burned_clip_subtitles"
            burned_count = 0
            total_to_burn = len(clips_with_titles)
            for index, clip in enumerate(clips_with_titles, start=1):
                if progress_callback:
                    subpercent = 45 + int((index - 1) / max(total_to_burn, 1) * 35)
                    progress_callback("EXPORT", f"正在烧录字幕 {index}/{total_to_burn}", subpercent)
                clip_path = clip_paths_by_id.get(str(clip.get("id")))
                if clip_path and _burn_subtitles_into_clip(clip_path, source_srt, clip, subtitle_work_dir):
                    burned_count += 1
            logger.info("Burned subtitles into %s/%s clips", burned_count, len(successful_clips))
            if progress_callback:
                progress_callback("EXPORT", f"字幕烧录完成 {burned_count}/{len(successful_clips)}", 80)
        else:
            logger.warning("Subtitle burn enabled, but subtitle file was not found: %s", source_srt)
    
    # 生成合集视频
    successful_collections = []
    if generate_collections and collections_data:
        if progress_callback:
            progress_callback("EXPORT", "正在导出合集视频", 85)
        successful_collections = generator.generate_collections(collections_data)
        if progress_callback:
            progress_callback("EXPORT", f"合集视频导出完成 {len(successful_collections)} 个", 95)
    
    # 保存元数据到项目目录
    # 注意：clips_metadata.json在这里保存，包含最终的切片元数据（包含视频路径等信息）
    # 这与step4的step4_titles.json不同，step4只保存带标题的片段数据
    if metadata_dir:
        project_metadata_dir = Path(metadata_dir)
        generator.save_clip_metadata(clips_with_titles, project_metadata_dir / "clips_metadata.json")
        generator.save_collection_metadata(collections_data, project_metadata_dir / "collections_metadata.json")
    else:
        generator.save_clip_metadata(clips_with_titles)
        generator.save_collection_metadata(collections_data)
    
    # 返回结果信息
    result = {
        'clips_generated': len(successful_clips),
        'collections_generated': len(successful_collections),
        'clip_paths': [str(path) for path in successful_clips],
        'collection_paths': [collection['video_path'] for collection in successful_collections],
        'collection_thumbnails': [collection['thumbnail_path'] for collection in successful_collections if collection['thumbnail_path']],
        'collections_info': successful_collections  # 包含完整的合集信息
    }
    
    logger.info(f"视频生成完成: {result['clips_generated']}个切片, {result['collections_generated']}个合集")
    
    # 保存结果到输出文件
    if output_dir is not None:
        output_path = output_dir / "step6_video_output.json"
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        logger.info(f"步骤6结果已保存到: {output_path}")
    
    return result
