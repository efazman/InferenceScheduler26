"""Heuristic workload-category assignment (version "heuristic-v1").

Deliberately simple and transparent: ordered keyword/regex rules, first match wins. Labels are
approximate and exist only to diversify the 500-prompt subset, not as ground truth.

To replace this later (e.g. with an LLM or embedding classifier), keep the signature
``assign_category(prompt) -> str`` returning one of CATEGORIES, and bump CATEGORY_SOURCE.
"""

from __future__ import annotations

import re

CATEGORY_SOURCE = "heuristic-v1"

CATEGORIES = (
    "knowledge_factual",
    "explanation_reasoning",
    "coding_debugging",
    "summarization_transformation",
    "writing_structured",
    "troubleshooting_advice",
)
FALLBACK = "knowledge_factual"


def _rx(*patterns: str) -> re.Pattern:
    return re.compile("|".join(f"(?:{p})" for p in patterns), re.IGNORECASE)


# Order matters: earlier rules win. Coding comes first because "fix this error" in code is
# debugging, not general troubleshooting; transformation comes before writing because
# "rewrite this" should not count as fresh writing.
RULES: list[tuple[str, re.Pattern]] = [
    ("coding_debugging", _rx(
        r"```", r"\bdef \w+\(", r"\bfunction\s*\w*\s*\(", r"#include\b", r"\bimport \w+",
        r"\b(?:python|javascript|typescript|java|c\+\+|c#|golang|rust|kotlin|swift|php|ruby|sql|bash|"
        r"html|css|regex|json|yaml|react|django|flask|pandas|numpy|api|sdk|compiler|stack ?trace|"
        r"traceback|segfault|runtime error|syntax error|exception|unit test|algorithm|leetcode)\b",
        r"\b(?:write|implement|debug|refactor)\b.{0,40}\b(?:code|function|class|script|program|query)\b",
    )),
    ("summarization_transformation", _rx(
        r"\b(?:summari[sz]e|summary|tl;?dr|condense|shorten|paraphrase|rephrase|reword|rewrite|"
        r"translate|translation|proofread|correct the grammar|fix the grammar|convert (?:this|the|it)|"
        r"extract (?:the|all)|bullet points? from)\b",
    )),
    ("writing_structured", _rx(
        r"\b(?:write|draft|compose|generate|create|make)\b.{0,40}\b(?:story|poem|essay|email|letter|"
        r"article|blog|post|script|speech|song|lyrics|outline|list|table|description|bio|resume|"
        r"cover letter|ad|slogan|tweet|review|report|plan|dialogue|names?|json|template)\b",
        r"^\s*(?:write|draft|compose|generate)\b",
    )),
    ("troubleshooting_advice", _rx(
        r"\b(?:error|bug|issue|problem|not working|doesn'?t work|won'?t|broken|crash(?:es|ed)?|fix|"
        r"troubleshoot|stuck|fails?|failing)\b",
        r"\b(?:how (?:can|do|should) i|should i|what should i|advice|tips?|recommend|help me)\b",
    )),
    ("explanation_reasoning", _rx(
        r"\b(?:explain|why|compare|comparison|difference between|differ|pros and cons|trade-?offs?|"
        r"step by step|reason(?:ing)?|analy[sz]e|implications?|what would happen|how does|how do)\b",
        r"\b(?:if|suppose)\b.{0,80}\bthen\b",
    )),
]


def assign_category(prompt: str) -> str:
    for category, pattern in RULES:
        if pattern.search(prompt):
            return category
    return FALLBACK
