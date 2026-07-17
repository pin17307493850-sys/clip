"""Shared product identity helpers for livestream clip processing."""

import re
from typing import Any, Optional


SUITE_WORDS = (
    "\u793c\u76d2",
    "\u5957\u88c5",
    "\u5957\u76d2",
    "\u5957\u9910",
    "\u7ec4\u5408",
)

GENERIC_QUALIFIERS = (
    "\u4ea7\u54c1",
    "\u4ea7\u54c1\u4ecb\u7ecd",
    "\u8336\u53f6",
    "\u56db\u6b3e\u8336\u54c1",
    "\u5185\u542b\u4ea7\u54c1",
    "\u5305\u88c5\u8bbe\u8ba1",
    "\u4ef7\u683c\u6743\u76ca",
    "\u94fe\u63a5\u914d\u7f6e",
    "\u542b\u6240\u6709\u914d\u7f6e",
    "\u6240\u6709\u914d\u7f6e",
)


def compact_product_name(value: Any) -> str:
    return re.sub(
        r"[\s:：,，。；;、\[\]【】<>《》/_\-]+",
        "",
        str(value or ""),
    ).strip()


def _parenthetical_parts(value: Any) -> tuple[str, Optional[str]]:
    text = str(value or "").strip()
    match = re.match(r"^(.*?)\s*[\(\uff08]([^\)\uff09]+)[\)\uff09]\s*$", text)
    if not match:
        return text, None
    return match.group(1).strip(), match.group(2).strip()


def _suite_composite_parts(value: Any) -> tuple[str, Optional[str]]:
    text = str(value or "").strip()
    for separator in ("-", "\u2014", ":", "\uff1a", "/"):
        if separator not in text:
            continue
        left, right = (part.strip() for part in text.split(separator, 1))
        if left and right and is_suite_product(left):
            return left, right
    return text, None


def is_suite_product(value: Any) -> bool:
    compact = compact_product_name(value)
    return any(word in compact for word in SUITE_WORDS)


def parent_product_name(value: Any) -> Optional[str]:
    """Return an explicit parent suite from names such as child(parent suite)."""
    _, qualifier = _parenthetical_parts(value)
    if qualifier and is_suite_product(qualifier):
        return canonical_product_name(qualifier)
    suite, child = _suite_composite_parts(value)
    if child:
        return canonical_product_name(suite)
    return None


def canonical_product_name(value: Any) -> str:
    """Normalize cosmetic product-name variants without collapsing child items."""
    _, child = _suite_composite_parts(value)
    if child:
        return canonical_product_name(child)
    head, qualifier = _parenthetical_parts(value)
    head = compact_product_name(head)
    qualifier_compact = compact_product_name(qualifier)
    if not head:
        return ""

    if qualifier_compact in GENERIC_QUALIFIERS:
        return head[:48]
    if qualifier and is_suite_product(qualifier):
        return head[:48]
    if qualifier_compact and qualifier_compact != head:
        return compact_product_name(f"{head}({qualifier_compact})")[:48]
    return head[:48]


def product_family_name(value: Any) -> str:
    """Return the suite family when explicitly present, otherwise the product."""
    return parent_product_name(value) or canonical_product_name(value)
