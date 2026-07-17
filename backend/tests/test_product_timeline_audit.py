import json

from backend.pipeline.product_identity import (
    canonical_product_name,
    parent_product_name,
    product_family_name,
)
from backend.pipeline.clip_dedup import dedupe_clips_by_time
from backend.pipeline.product_timeline_audit import audit_product_timeline
from backend.pipeline.step5_clustering import ClusteringEngine
from backend.utils.text_processor import TextProcessor


GIFT_BOX = "\u7231\u4e3d\u4e1d\u4e4b\u68a6\u793c\u76d2"
ROSE_TEA = "\u73ab\u7470\u82b1\u8336"


def _subtitle(start, end, text):
    return {"start_time": start, "end_time": end, "text": text}


def test_product_identity_keeps_child_and_links_explicit_parent():
    child = f"{ROSE_TEA}\uff08{GIFT_BOX}\uff09"

    assert canonical_product_name(f"{GIFT_BOX}\uff08\u8336\u53f6\uff09") == GIFT_BOX
    assert canonical_product_name(child) == ROSE_TEA
    assert parent_product_name(child) == GIFT_BOX
    assert product_family_name(child) == GIFT_BOX
    assert canonical_product_name(f"{GIFT_BOX}-\u871c\u6843\u751c\u5fc3") == "\u871c\u6843\u751c\u5fc3"
    assert parent_product_name(f"{GIFT_BOX}-\u871c\u6843\u751c\u5fc3") == GIFT_BOX
    assert canonical_product_name(f"{GIFT_BOX}\uff08\u542b\u6240\u6709\u914d\u7f6e\uff09") == GIFT_BOX


def test_global_audit_extends_suite_to_last_child_and_stops_before_price(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "chunk_0.json").write_text(
        json.dumps(
            [
                _subtitle("00:13:41,850", "00:14:27,070", "\u524d\u9762\u4e09\u6b3e\u8336\u7684\u4ecb\u7ecd"),
                _subtitle("00:14:27,070", "00:14:36,270", "\u559d\u8d77\u6765\u9178\u751c\uff0c\u50cf\u9178\u6885\u6c64"),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (chunks / "chunk_1.json").write_text(
        json.dumps(
            [
                _subtitle("00:14:37,210", "00:14:40,870", "\u7136\u540e\u6700\u540e\u4e00\u4e2a\u662f\u73ab\u7470\u82b1\u8336"),
                _subtitle("00:14:40,870", "00:14:46,530", "\u5bf9\u5973\u751f\u53cb\u597d\uff0c\u53ef\u4ee5\u52a0\u5728\u8336\u91cc\u6216\u76f4\u63a5\u6ce1\u6c34"),
                _subtitle("00:14:46,530", "00:14:54,810", "\u5c0f\u5706\u7f50\u4e00\u7f50\u516d\u5341\u5757\u94b1\uff0c\u56db\u7f50\u4e8c\u767e\u56db"),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    clips = [
        {
            "id": "1",
            "product": GIFT_BOX,
            "product_aspect": "flavor",
            "start_time": "00:13:41,850",
            "end_time": "00:14:27,070",
        }
    ]

    audited = audit_product_timeline(clips, chunks, TextProcessor())

    assert audited[0]["end_time"] == "00:14:46,530"
    assert ROSE_TEA in audited[0]["content"]
    assert "\u516d\u5341\u5757\u94b1" not in audited[0]["content"]
    assert audited[0]["product_family"] == GIFT_BOX


def test_global_audit_does_not_extend_non_suite_product(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "chunk_0.json").write_text(
        json.dumps(
            [
                _subtitle("00:01:00,000", "00:01:08,000", "\u73ab\u7470\u82b1\u8336\u9999\u6c14\u5f88\u597d"),
                _subtitle("00:01:08,000", "00:01:16,000", "\u6700\u540e\u4e00\u6b3e\u662f\u6842\u82b1\u4e4c\u9f99"),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    clips = [
        {
            "id": "1",
            "product": ROSE_TEA,
            "product_aspect": "flavor",
            "start_time": "00:01:00,000",
            "end_time": "00:01:08,000",
        }
    ]

    audited = audit_product_timeline(clips, chunks, TextProcessor())

    assert audited[0]["end_time"] == "00:01:08,000"


def test_global_audit_rewinds_composite_child_to_suite_intro(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "chunk_0.json").write_text(
        json.dumps(
            [
                _subtitle("00:13:41,850", "00:13:45,690", "\u7ee7\u7eed\u7ed9\u5b9d\u5b9d\u4ecb\u7ecd\u4e00\u4e0b\u6211\u4eec\u7684\u7231\u4e3d\u4e1d\u4e4b\u68a6\u7684\u793c\u76d2"),
                _subtitle("00:13:45,690", "00:14:07,650", "\u7b2c\u4e00\u6b3e\u8461\u8404\u5c0f\u591c\u66f2\uff0c\u5976\u6cb9\u8461\u8404\u9999"),
                _subtitle("00:14:07,650", "00:14:17,030", "\u8fd8\u6709\u6211\u4eec\u7684\u871c\u6843\u751c\u5fc3\uff0c\u4eba\u95f4\u6c34\u871c\u6843"),
                _subtitle("00:14:17,030", "00:14:37,210", "\u72fc\u59c6\u679c\u8336\u662f\u95e8\u5e97\u7206\u6b3e\uff0c\u5165\u53e3\u9178\u751c"),
                _subtitle("00:14:37,210", "00:14:46,530", "\u6700\u540e\u4e00\u6b3e\u662f\u73ab\u7470\u82b1\u8336\uff0c\u53ef\u4ee5\u76f4\u63a5\u6ce1\u6c34"),
                _subtitle("00:14:46,530", "00:14:54,810", "\u5c0f\u5706\u7f50\u4e00\u7f50\u516d\u5341\u5757\u94b1"),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    clips = [
        {
            "id": "1",
            "product": f"{GIFT_BOX}-\u871c\u6843\u751c\u5fc3",
            "start_time": "00:14:07,650",
            "end_time": "00:14:48,270",
            "outline": "\u4eba\u95f4\u6c34\u871c\u6843\u3001\u51b0\u9547\u679c\u6c41\u611f",
            "content": "\u871c\u6843\u751c\u5fc3\uff0c\u540e\u9762\u8fd8\u6df7\u5165\u4e86\u516d\u5341\u5757\u94b1\u7684\u4ef7\u683c\u53e5",
        }
    ]

    audited = audit_product_timeline(clips, chunks, TextProcessor())

    assert audited[0]["start_time"] == "00:13:41,850"
    assert audited[0]["end_time"] == "00:14:46,530"
    assert "\u8461\u8404\u5c0f\u591c\u66f2" in audited[0]["content"]
    assert ROSE_TEA in audited[0]["content"]
    assert audited[0]["parent_product"] == GIFT_BOX
    assert audited[0]["product"] == GIFT_BOX
    assert audited[0]["original_product"] == f"{GIFT_BOX}-\u871c\u6843\u751c\u5fc3"


def test_suite_intro_allows_words_between_continue_and_introduce(tmp_path):
    chunks = tmp_path / "chunks"
    chunks.mkdir()
    (chunks / "chunk_0.json").write_text(
        json.dumps(
            [
                _subtitle(
                    "00:13:41,850",
                    "00:13:45,690",
                    "\u7136\u540e\u518d\u7ee7\u7eed\u7ed9\u5b9d\u5b9d\u4ecb\u7ecd\u4e00\u4e0b\u6211\u4eec\u7684\u7231\u4e3d\u4e1d\u4e4b\u68a6\u7684\u793c\u76d2",
                ),
                _subtitle(
                    "00:13:45,690",
                    "00:13:49,710",
                    "\u8fd9\u4e2a\u793c\u76d2\u91cc\u90fd\u662f\u6765\u81ea\u5168\u4e16\u754c\u5404\u5730\u7684\u8336",
                ),
            ],
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    clips = [
        {
            "start_time": "00:13:46,710",
            "end_time": "00:13:49,710",
            "product": f"{GIFT_BOX}\uff08\u542b\u6240\u6709\u914d\u7f6e\uff09",
            "outline": "\u793c\u76d2\u4e2d\u7684\u56db\u6b3e\u8336",
        }
    ]

    audited = audit_product_timeline(clips, chunks, TextProcessor())

    assert audited[0]["start_time"] == "00:13:41,850"


def test_short_contained_child_is_deduped_against_complete_suite():
    clips = [
        {
            "id": "suite",
            "start_time": "00:13:41,850",
            "end_time": "00:14:46,530",
            "product": f"{GIFT_BOX}\uff08\u542b\u6240\u6709\u914d\u7f6e\uff09",
            "score": 92,
            "product_boundary_audited": True,
        },
        {
            "id": "child",
            "start_time": "00:14:37,210",
            "end_time": "00:14:46,530",
            "product": f"{GIFT_BOX}-{ROSE_TEA}",
            "score": 94,
            "product_boundary_audited": True,
        },
    ]

    deduped = dedupe_clips_by_time(clips)

    assert [clip["id"] for clip in deduped] == ["suite"]


def test_same_product_contained_variants_dedupe_across_aspects():
    clips = [
        {
            "id": "complete",
            "start_time": "00:20:36,130",
            "end_time": "00:21:18,130",
            "product": "\u6a31\u82b1\u8336",
            "product_aspect": "general",
            "score": 85,
        },
        {
            "id": "nested",
            "start_time": "00:20:43,090",
            "end_time": "00:21:15,330",
            "product": "\u6a31\u82b1\u8336",
            "product_aspect": "flavor",
            "score": 80,
        },
    ]

    deduped = dedupe_clips_by_time(clips)

    assert [clip["id"] for clip in deduped] == ["complete"]


def test_product_collections_merge_generic_suite_variants_and_avoid_contained_child():
    engine = ClusteringEngine.__new__(ClusteringEngine)
    clips = [
        {
            "id": "1",
            "product": f"{GIFT_BOX}\uff08\u8336\u53f6\uff09",
            "start_time": "00:10:00,000",
            "end_time": "00:11:20,000",
            "final_score": 95,
        },
        {
            "id": "2",
            "product": f"{ROSE_TEA}\uff08{GIFT_BOX}\uff09",
            "start_time": "00:10:35,000",
            "end_time": "00:10:55,000",
            "final_score": 93,
        },
        {
            "id": "3",
            "product": f"{GIFT_BOX}\uff08\u4ef7\u683c\u6743\u76ca\uff09",
            "start_time": "00:11:25,000",
            "end_time": "00:11:55,000",
            "final_score": 90,
        },
    ]

    collections = engine._create_product_collections(clips)
    parent = next(item for item in collections if item.get("product") == GIFT_BOX)
    child = next(item for item in collections if item.get("product") == ROSE_TEA)

    assert parent["clip_ids"] == ["1", "3"]
    assert child["clip_ids"] == ["2"]
