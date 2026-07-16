"""
简化的流水线适配器 - 集成新的进度系统
"""

import logging
import time
import json
from typing import Dict, Any, Optional, Callable
from pathlib import Path

from backend.services.simple_progress import emit_progress, clear_progress
from backend.pipeline.step1_outline import run_step1_outline
from backend.pipeline.step2_timeline import run_step2_timeline
from backend.pipeline.step3_scoring import run_step3_scoring
from backend.pipeline.step4_title import normalize_clip_titles, run_step4_title
from backend.pipeline.step5_clustering import run_step5_clustering
from backend.pipeline.step6_video import run_step6_video

logger = logging.getLogger(__name__)


class SimplePipelineAdapter:
    """简化的流水线适配器，使用固定阶段进度系统"""
    
    def __init__(self, project_id: str, task_id: str):
        self.project_id = project_id
        self.task_id = task_id

    def _load_checkpoint(self, path: Path, label: str, require_non_empty: bool = True):
        if not path.exists():
            return None
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            if require_non_empty and isinstance(data, list) and not data:
                logger.info("断点文件为空，将重新生成: %s", path)
                return None
            logger.info("复用%s断点: %s", label, path)
            return data
        except Exception as checkpoint_error:
            logger.warning("读取%s断点失败，将重新生成: %s", label, checkpoint_error)
            return None
        
    async def _generate_subtitle_automatically(self, video_path: str, metadata_dir: Path) -> Path:
        """
        自动生成字幕文件
        
        Args:
            video_path: 视频文件路径
            metadata_dir: 元数据目录
            
        Returns:
            生成的SRT文件路径，如果失败返回None
        """
        try:
            logger.info(f"开始为视频 {video_path} 自动生成字幕")
            
            # 更新进度
            from backend.services.simple_progress import emit_progress
            emit_progress(self.project_id, "SUBTITLE", "正在使用AI生成字幕...", subpercent=25)
            
            # 使用Whisper本地模型生成字幕
            try:
                from backend.utils.speech_recognizer import generate_subtitle_for_video
                from pathlib import Path
                
                video_file_path = Path(video_path)
                if not video_file_path.exists():
                    logger.error(f"视频文件不存在: {video_path}")
                    return None
                
                try:
                    from backend.core.desktop_config import get_desktop_config
                    speech_config = get_desktop_config().speech_recognition
                    whisper_model = speech_config.whisper_config.model_name or "base"
                    whisper_language = speech_config.whisper_config.language or "auto"
                    whisper_device = getattr(speech_config.whisper_config, "device", "auto")
                    whisper_compute_type = getattr(speech_config.whisper_config, "compute_type", "auto")
                except Exception as config_error:
                    logger.warning(f"读取语音识别配置失败，使用默认Whisper配置: {config_error}")
                    whisper_model = "base"
                    whisper_language = "auto"
                    whisper_device = "auto"
                    whisper_compute_type = "auto"

                logger.info(
                    "尝试使用Whisper本地模型生成字幕: model=%s, language=%s, device=%s, compute_type=%s",
                    whisper_model,
                    whisper_language,
                    whisper_device,
                    whisper_compute_type,
                )
                output_path = metadata_dir / f"{video_file_path.stem}.srt"
                last_progress_emit = {"ts": 0.0, "percent": -1}

                def subtitle_progress(
                    current_seconds: float,
                    total_seconds: float,
                    segment_count: int,
                    phase: str = "",
                ) -> None:
                    phase_messages = {
                        "waiting": ("等待字幕识别队列，批量导入时会串行处理", 5),
                        "cached": ("已复用历史字幕，跳过AI识别", 100),
                        "extracting_audio": ("正在从视频中提取音频", 12),
                        "loading_model": f"正在加载Whisper模型 {whisper_model}",
                        "transcribing": "正在识别字幕",
                        "writing_srt": ("正在写入字幕文件", 96),
                        "done": ("AI字幕生成完成", 100),
                    }
                    phase_message = phase_messages.get(phase)
                    if isinstance(phase_message, tuple):
                        emit_progress(self.project_id, "SUBTITLE", phase_message[0], subpercent=phase_message[1])
                        return
                    if isinstance(phase_message, str) and total_seconds <= 0:
                        emit_progress(self.project_id, "SUBTITLE", phase_message, subpercent=20)
                        return
                    if total_seconds <= 0:
                        return
                    now = time.monotonic()
                    percent = max(25, min(95, 25 + (current_seconds / total_seconds) * 70))
                    rounded_percent = int(percent)
                    if (
                        rounded_percent <= last_progress_emit["percent"]
                        and now - last_progress_emit["ts"] < 8
                    ):
                        return
                    last_progress_emit["ts"] = now
                    last_progress_emit["percent"] = rounded_percent
                    emit_progress(
                        self.project_id,
                        "SUBTITLE",
                        f"AI字幕识别中 {current_seconds / 60:.1f}/{total_seconds / 60:.1f} 分钟，{segment_count} 段",
                        subpercent=percent,
                    )

                srt_path = generate_subtitle_for_video(
                    video_file_path,
                    output_path=output_path,
                    method="whisper_local",
                    model=whisper_model,
                    language=whisper_language,
                    device=whisper_device,
                    compute_type=whisper_compute_type,
                    progress_callback=subtitle_progress,
                )
                
                if srt_path and srt_path.exists():
                    logger.info(f"Whisper生成字幕成功: {srt_path}")
                    emit_progress(self.project_id, "SUBTITLE", "AI字幕生成完成", subpercent=40)
                    return srt_path
                else:
                    logger.warning("Whisper生成字幕失败")
                    
            except Exception as e:
                logger.warning(f"Whisper生成字幕失败: {e}")
            
            logger.error("Whisper字幕生成失败")
            return None
            
        except Exception as e:
            logger.error(f"自动生成字幕过程中发生错误: {e}")
            return None
        
    async def process_project_sync(self, input_video_path: str, input_srt_path: str) -> Dict[str, Any]:
        """
        同步处理项目 - 使用简化的进度系统
        
        Args:
            input_video_path: 输入视频路径
            input_srt_path: 输入SRT路径
            
        Returns:
            处理结果
        """
        logger.info(f"开始处理项目: {self.project_id}")
        
        try:
            # 清除之前的进度数据
            clear_progress(self.project_id)
            
            # 创建必要的目录结构 - 使用正确的路径
            from backend.core.path_utils import get_project_directory
            project_dir = get_project_directory(self.project_id)
            metadata_dir = project_dir / "metadata"
            output_dir = project_dir / "output"
            metadata_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(parents=True, exist_ok=True)
            # 项目内专属输出子目录
            clips_output_dir = output_dir / "clips"
            collections_output_dir = output_dir / "collections"
            clips_output_dir.mkdir(parents=True, exist_ok=True)
            collections_output_dir.mkdir(parents=True, exist_ok=True)

            video_category = "default"
            try:
                from backend.core.database import SessionLocal
                from backend.models.project import Project
                from backend.core.shared_config import get_prompt_files

                db = SessionLocal()
                try:
                    project = db.query(Project).filter(Project.id == self.project_id).first()
                    if project and project.processing_config:
                        video_category = project.processing_config.get("video_category", "default")
                finally:
                    db.close()

                prompt_files = get_prompt_files(video_category)
                logger.info("使用视频分类提示词: %s", video_category)
            except Exception as prompt_error:
                from backend.core.shared_config import get_prompt_files

                logger.warning("读取视频分类提示词失败，使用默认提示词: %s", prompt_error)
                prompt_files = get_prompt_files()
            
            # 阶段1: 素材准备
            emit_progress(self.project_id, "INGEST", "素材准备完成")
            
            # 阶段2: 字幕处理
            emit_progress(self.project_id, "SUBTITLE", "开始字幕处理")
            
            # Step 1: 大纲提取
            logger.info("执行Step 1: 大纲提取")
            outlines_path = metadata_dir / "step1_outline.json"
            outlines = self._load_checkpoint(outlines_path, "大纲")
            if outlines is not None:
                emit_progress(self.project_id, "SUBTITLE", "已复用大纲断点，跳过大纲分析", subpercent=50)
            elif input_srt_path and Path(input_srt_path).exists():
                logger.info(f"使用现有SRT文件: {input_srt_path}")
                outlines = run_step1_outline(Path(input_srt_path), metadata_dir=metadata_dir, prompt_files=prompt_files)
            else:
                logger.warning("没有SRT文件，尝试自动生成字幕")
                # 尝试自动生成字幕
                srt_path = await self._generate_subtitle_automatically(input_video_path, metadata_dir)
                if srt_path and srt_path.exists():
                    logger.info(f"自动生成字幕成功: {srt_path}")
                    outlines = run_step1_outline(srt_path, metadata_dir=metadata_dir, prompt_files=prompt_files)
                else:
                    error_msg = (
                        "字幕生成失败，无法继续生成切片。请在设置 -> 语音识别中安装 Whisper 运行时并下载模型，"
                        "或在创建项目时上传 .srt 字幕文件。"
                    )
                    logger.error(error_msg)
                    emit_progress(self.project_id, "DONE", f"处理失败: {error_msg}")
                    return {
                        "status": "failed",
                        "project_id": self.project_id,
                        "task_id": self.task_id,
                        "message": error_msg,
                    }
            emit_progress(self.project_id, "SUBTITLE", "字幕处理完成", subpercent=50)
            
            # 阶段3: 内容分析
            emit_progress(self.project_id, "ANALYZE", "开始内容分析")
            
            # Step 2: 时间线提取
            logger.info("执行Step 2: 时间线提取")
            if outlines:  # 只有当有大纲时才执行后续步骤
                emit_progress(self.project_id, "ANALYZE", "正在定位时间线", subpercent=10)
                timeline_path = metadata_dir / "step2_timeline.json"
                timeline_data = self._load_checkpoint(timeline_path, "时间线")
                if timeline_data is not None:
                    emit_progress(self.project_id, "ANALYZE", "已复用时间线断点", subpercent=50)
                else:
                    timeline_data = run_step2_timeline(
                        metadata_dir / "step1_outline.json",
                        metadata_dir=metadata_dir,
                        prompt_files=prompt_files,
                        progress_callback=lambda stage, message, subpercent=10: emit_progress(
                            self.project_id,
                            stage,
                            message,
                            subpercent=subpercent,
                        ),
                    )
                emit_progress(self.project_id, "ANALYZE", "时间线提取完成", subpercent=50)
                
                # Step 3: 内容评分
                logger.info("执行Step 3: 内容评分")
                emit_progress(self.project_id, "ANALYZE", "正在给候选片段评分", subpercent=60)
                scored_path = metadata_dir / "step3_high_score_clips.json"
                scored_clips = self._load_checkpoint(scored_path, "评分")
                if scored_clips is not None:
                    emit_progress(self.project_id, "ANALYZE", "已复用评分断点", subpercent=95)
                else:
                    scored_clips = run_step3_scoring(
                        metadata_dir / "step2_timeline.json",
                        metadata_dir=metadata_dir,
                        prompt_files=prompt_files,
                        progress_callback=lambda stage, message, subpercent=60: emit_progress(
                            self.project_id,
                            stage,
                            message,
                            subpercent=subpercent,
                        ),
                    )
                emit_progress(self.project_id, "ANALYZE", "内容分析完成", subpercent=100)
            else:
                logger.warning("没有大纲数据，跳过时间线提取和内容评分")
                # 创建空的时间线和评分文件
                timeline_file = metadata_dir / "step2_timeline.json"
                scored_file = metadata_dir / "step3_high_score_clips.json"
                with open(timeline_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                with open(scored_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                # 初始化空变量
                timeline_data = []
                scored_clips = []
                emit_progress(self.project_id, "ANALYZE", "内容分析完成", subpercent=100)
            
            # 阶段4: 片段定位
            emit_progress(self.project_id, "HIGHLIGHT", "开始片段定位")
            
            # Step 4: 标题生成
            logger.info("执行Step 4: 标题生成")
            if outlines:  # 只有当有大纲时才执行后续步骤
                emit_progress(self.project_id, "HIGHLIGHT", "正在生成切片标题", subpercent=20)
                titles_path = metadata_dir / "step4_titles.json"
                titled_clips = self._load_checkpoint(titles_path, "标题")
                if titled_clips is not None:
                    titled_clips = normalize_clip_titles(titled_clips)
                    with open(titles_path, "w", encoding="utf-8") as f:
                        json.dump(titled_clips, f, ensure_ascii=False, indent=2)
                    emit_progress(self.project_id, "HIGHLIGHT", "已复用标题断点", subpercent=40)
                else:
                    titled_clips = run_step4_title(
                        metadata_dir / "step3_high_score_clips.json",
                        metadata_dir=str(metadata_dir),
                        prompt_files=prompt_files,
                        progress_callback=lambda stage, message, subpercent=20: emit_progress(
                            self.project_id,
                            stage,
                            message,
                            subpercent=subpercent,
                        ),
                    )
                emit_progress(self.project_id, "HIGHLIGHT", "标题生成完成", subpercent=40)

                emit_progress(self.project_id, "EXPORT", "正在先导出已分析切片", subpercent=20)
                empty_collections_path = metadata_dir / "step5_collections.preview.json"
                with open(empty_collections_path, "w", encoding="utf-8") as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                preview_video_result = run_step6_video(
                    metadata_dir / "step4_titles.json",
                    empty_collections_path,
                    input_video_path,
                    output_dir=output_dir,
                    clips_dir=str(clips_output_dir),
                    collections_dir=str(collections_output_dir),
                    metadata_dir=str(metadata_dir),
                    burn_subtitles=False,
                    generate_collections=False,
                    progress_callback=lambda stage, message, subpercent=20: emit_progress(
                        self.project_id,
                        stage,
                        message,
                        subpercent=subpercent,
                    ),
                )
                logger.info("已先导出单条切片: %s", preview_video_result)
                
                # Step 5: 主题聚类
                logger.info("执行Step 5: 主题聚类")
                emit_progress(self.project_id, "HIGHLIGHT", "正在整理合集", subpercent=70)
                collections_path = metadata_dir / "step5_collections.json"
                collections = None
                if video_category not in ("product_intro_short", "live_product"):
                    collections = self._load_checkpoint(collections_path, "合集")
                if collections is not None:
                    emit_progress(self.project_id, "HIGHLIGHT", "已复用合集断点", subpercent=95)
                else:
                    collections = run_step5_clustering(
                        metadata_dir / "step4_titles.json",
                        metadata_dir=str(metadata_dir),
                        prompt_files=prompt_files,
                    )
                emit_progress(self.project_id, "HIGHLIGHT", "片段定位完成", subpercent=100)
                
                # 阶段5: 视频导出
                emit_progress(self.project_id, "EXPORT", "开始视频导出")
                
                # Step 6: 视频切割
                logger.info("执行Step 6: 视频切割")
                emit_progress(self.project_id, "EXPORT", "正在导出切片视频", subpercent=20)
                video_result = run_step6_video(
                    metadata_dir / "step4_titles.json",
                    metadata_dir / "step5_collections.json",
                    input_video_path,
                    output_dir=output_dir,
                    clips_dir=str(clips_output_dir),
                    collections_dir=str(collections_output_dir),
                    metadata_dir=str(metadata_dir),
                    burn_subtitles=False,
                    progress_callback=lambda stage, message, subpercent=20: emit_progress(
                        self.project_id,
                        stage,
                        message,
                        subpercent=subpercent,
                    ),
                )
            else:
                logger.warning("没有大纲数据，跳过标题生成、主题聚类和视频切割")
                # 创建空的标题和合集文件
                titles_file = metadata_dir / "step4_titles.json"
                collections_file = metadata_dir / "step5_collections.json"
                with open(titles_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                with open(collections_file, 'w', encoding='utf-8') as f:
                    json.dump([], f, ensure_ascii=False, indent=2)
                # 初始化空变量
                titled_clips = []
                collections = []
                emit_progress(self.project_id, "HIGHLIGHT", "片段定位完成", subpercent=100)
                emit_progress(self.project_id, "EXPORT", "开始视频导出")
                video_result = {"status": "skipped", "message": "没有内容可处理"}
            emit_progress(self.project_id, "EXPORT", "视频导出完成", subpercent=100)
            
            # 阶段6: 处理完成
            emit_progress(self.project_id, "DONE", "处理完成")
            
            # 自动同步数据到数据库
            try:
                from backend.services.data_sync_service import DataSyncService
                from backend.core.database import SessionLocal
                
                db = SessionLocal()
                try:
                    sync_service = DataSyncService(db)
                    sync_result = sync_service.sync_project_from_filesystem(self.project_id, project_dir)
                    if sync_result.get("success"):
                        logger.info(f"项目 {self.project_id} 数据同步成功: {sync_result}")
                    else:
                        logger.error(f"项目 {self.project_id} 数据同步失败: {sync_result}")
                finally:
                    db.close()
            except Exception as e:
                logger.error(f"数据同步失败: {e}")
            
            logger.info(f"项目处理完成: {self.project_id}")
            return {
                "status": "succeeded",
                "project_id": self.project_id,
                "task_id": self.task_id,
                "result": {
                    "outlines": outlines,
                    "timeline": timeline_data,
                    "scored_clips": scored_clips,
                    "titled_clips": titled_clips,
                    "collections": collections,
                    "video_result": video_result
                }
            }
            
        except Exception as e:
            error_msg = f"流水线处理失败: {str(e)}"
            logger.error(error_msg)
            
            # 发送失败状态
            emit_progress(self.project_id, "DONE", f"处理失败: {error_msg}")
            
            return {
                "status": "failed",
                "project_id": self.project_id,
                "task_id": self.task_id,
                "error": error_msg
            }


def create_simple_pipeline_adapter(project_id: str, task_id: str) -> SimplePipelineAdapter:
    """创建简化的流水线适配器实例"""
    return SimplePipelineAdapter(project_id, task_id)
