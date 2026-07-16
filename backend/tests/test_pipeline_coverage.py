from backend.pipeline.clip_dedup import merge_cross_chunk_product_clips
from backend.pipeline.step2_timeline import TimelineExtractor
from backend.utils.text_processor import TextProcessor


def test_cross_chunk_same_product_section_is_merged():
    clips = [
        {
            "chunk_index": 0,
            "product": "爱丽丝之梦礼盒",
            "product_aspect": "packaging",
            "start_time": "00:04:30,000",
            "end_time": "00:05:01,000",
            "content": "介绍礼盒外观",
        },
        {
            "chunk_index": 1,
            "product": "爱丽丝之梦礼盒",
            "product_aspect": "packaging",
            "start_time": "00:05:02,000",
            "end_time": "00:05:36,000",
            "content": "继续介绍包装颜色",
        },
    ]

    merged = merge_cross_chunk_product_clips(clips)

    assert len(merged) == 1
    assert merged[0]["start_time"] == "00:04:30,000"
    assert merged[0]["end_time"] == "00:05:36,000"
    assert "包装颜色" in merged[0]["content"]


def test_cross_chunk_different_product_aspects_remain_separate():
    clips = [
        {
            "chunk_index": 0,
            "product": "爱丽丝之梦礼盒",
            "product_aspect": "packaging",
            "start_time": "00:04:30,000",
            "end_time": "00:05:01,000",
        },
        {
            "chunk_index": 1,
            "product": "爱丽丝之梦礼盒",
            "product_aspect": "price",
            "start_time": "00:05:02,000",
            "end_time": "00:05:36,000",
        },
    ]

    assert len(merge_cross_chunk_product_clips(clips)) == 2


def test_timeline_parse_fallback_keeps_timed_outline_and_subtitles():
    processor = TextProcessor()
    extractor = TimelineExtractor.__new__(TimelineExtractor)
    extractor.text_processor = processor
    subtitles = [
        {
            "index": 1,
            "start_time": "00:20:44,090",
            "end_time": "00:20:49,000",
            "text": "继续介绍这款产品的颜色和包装。",
        },
        {
            "index": 2,
            "start_time": "00:20:49,100",
            "end_time": "00:21:15,000",
            "text": "再说明适合人群和购买建议。",
        },
    ]

    fallback = extractor._fallback_timeline_from_outlines(
        [
            {
                "title": "产品对比与购买建议",
                "product": "直播产品",
                "start_time": "00:20:44,090",
                "end_time": "00:21:15,000",
            }
        ],
        subtitles,
        4,
        "00:20:34,190",
        "00:21:32,710",
    )

    assert len(fallback) == 1
    assert fallback[0]["start_time"] == "00:20:44,090"
    assert fallback[0]["end_time"] == "00:21:15,000"
    assert fallback[0]["timeline_source"] == "outline_fallback"
    assert "购买建议" in fallback[0]["content"]
