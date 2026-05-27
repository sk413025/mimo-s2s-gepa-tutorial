from __future__ import annotations

import json
from typing import Any

ALLOWED_EDIT_OPS = {"append_after_heading", "insert_after_text", "replace_text", "append_to_end"}


def strip_code_fence(text: str) -> str:
    stripped = text.strip()
    if not stripped.startswith("```"):
        return stripped
    lines = stripped.splitlines()
    if lines and lines[0].startswith("```"):
        lines = lines[1:]
    if lines and lines[-1].startswith("```"):
        lines = lines[:-1]
    return "\n".join(lines).strip()


def parse_edits_json(text: str) -> list[dict[str, Any]]:
    cleaned = strip_code_fence(text)
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        start = cleaned.find("[")
        end = cleaned.rfind("]")
        if start == -1 or end == -1 or end <= start:
            raise
        parsed = json.loads(cleaned[start : end + 1])
    if not isinstance(parsed, list):
        raise ValueError("Patch proposer must return a JSON array of edits")
    edits: list[dict[str, Any]] = []
    for index, edit in enumerate(parsed, start=1):
        if not isinstance(edit, dict):
            raise ValueError(f"edit {index} is not a JSON object")
        op = str(edit.get("op") or "")
        if op not in ALLOWED_EDIT_OPS:
            raise ValueError(f"edit {index} has unsupported op: {op}")
        edits.append(edit)
    return edits


def validate_edits(edits: list[dict[str, Any]], max_edits: int = 3, max_text_chars: int = 1200) -> list[str]:
    errors: list[str] = []
    if not edits:
        errors.append("patch must contain at least one edit")
    if len(edits) > max_edits:
        errors.append(f"patch may contain at most {max_edits} edits")

    for index, edit in enumerate(edits, start=1):
        op = str(edit.get("op") or "")
        text = str(edit.get("text") or edit.get("new_text") or "")
        if len(text) > max_text_chars:
            errors.append(f"edit {index} text is too long")
        if op == "append_after_heading" and not edit.get("heading"):
            errors.append(f"edit {index} append_after_heading requires heading")
        if op == "insert_after_text" and not edit.get("anchor"):
            errors.append(f"edit {index} insert_after_text requires anchor")
        if op == "replace_text" and (not edit.get("old_text") or not edit.get("new_text")):
            errors.append(f"edit {index} replace_text requires old_text and new_text")
        if op in {"append_after_heading", "insert_after_text", "append_to_end"} and not text.strip():
            errors.append(f"edit {index} requires non-empty text")
    return errors


def _insert_after_line(lines: list[str], line_index: int, text: str) -> list[str]:
    insert_lines = text.strip("\n").splitlines()
    return lines[: line_index + 1] + insert_lines + lines[line_index + 1 :]


def append_after_heading(skill_md: str, heading: str, text: str) -> str:
    lines = skill_md.splitlines()
    wanted = heading.strip()
    for index, line in enumerate(lines):
        if line.strip() == wanted:
            return "\n".join(_insert_after_line(lines, index, text)).strip() + "\n"
    raise ValueError(f"heading not found: {heading}")


def insert_after_text(skill_md: str, anchor: str, text: str) -> str:
    if anchor not in skill_md:
        raise ValueError(f"anchor not found: {anchor}")
    return skill_md.replace(anchor, anchor.rstrip("\n") + "\n" + text.strip("\n"), 1)


def replace_text(skill_md: str, old_text: str, new_text: str) -> str:
    if old_text not in skill_md:
        raise ValueError(f"old_text not found: {old_text[:120]}")
    return skill_md.replace(old_text, new_text, 1)


def apply_skill_edits(skill_md: str, edits: list[dict[str, Any]]) -> str:
    result = skill_md
    for edit in edits:
        op = str(edit["op"])
        if op == "append_after_heading":
            result = append_after_heading(result, str(edit["heading"]), str(edit["text"]))
        elif op == "insert_after_text":
            result = insert_after_text(result, str(edit["anchor"]), str(edit["text"]))
        elif op == "replace_text":
            result = replace_text(result, str(edit["old_text"]), str(edit["new_text"]))
        elif op == "append_to_end":
            result = result.rstrip() + "\n\n" + str(edit["text"]).strip() + "\n"
        else:
            raise ValueError(f"unsupported edit op: {op}")
    return result.strip() + "\n"
