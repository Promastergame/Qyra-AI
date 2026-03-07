"""
tools/parser.py — Detect and parse tool calls in model output.

Format:
    <|tool_start|>tool_name
    {"arg": "value"}
    <|tool_end|>
"""

import json
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class ToolCall:
    """Parsed tool call."""
    tool_name: str
    args_json: str
    args: dict
    raw_text: str
    start_pos: int
    end_pos: int


class ToolParser:
    """Parse tool calls from model-generated text."""

    TOOL_START = "<|tool_start|>"
    TOOL_END = "<|tool_end|>"
    TOOL_RESULT = "<|tool_result|>"

    @classmethod
    def has_tool_call(cls, text: str) -> bool:
        return cls.TOOL_START in text and cls.TOOL_END in text

    @classmethod
    def parse(cls, text: str) -> Optional[ToolCall]:
        start_idx = text.find(cls.TOOL_START)
        if start_idx == -1:
            return None

        end_idx = text.find(cls.TOOL_END, start_idx)
        if end_idx == -1:
            return None

        content_start = start_idx + len(cls.TOOL_START)
        raw = text[content_start:end_idx].strip()  # type: ignore

        if not raw:
            return None

        tool_name, args_json, args = cls._parse_content(raw)
        if tool_name is None:
            return None

        return ToolCall(
            tool_name=tool_name,
            args_json=args_json,
            args=args,
            raw_text=text[start_idx:end_idx + len(cls.TOOL_END)],  # type: ignore
            start_pos=start_idx,
            end_pos=end_idx + len(cls.TOOL_END),
        )

    @classmethod
    def parse_all(cls, text: str) -> List[ToolCall]:
        calls = []
        search_start = 0

        while True:
            start_idx = text.find(cls.TOOL_START, search_start)
            if start_idx == -1:
                break

            end_idx = text.find(cls.TOOL_END, start_idx)
            if end_idx == -1:
                break

            content_start = start_idx + len(cls.TOOL_START)
            raw = text[content_start:end_idx].strip()  # type: ignore

            tool_name, args_json, args = cls._parse_content(raw)
            if tool_name is not None:
                calls.append(ToolCall(
                    tool_name=tool_name,
                    args_json=args_json,
                    args=args,
                    raw_text=text[start_idx:end_idx + len(cls.TOOL_END)],  # type: ignore
                    start_pos=start_idx,
                    end_pos=end_idx + len(cls.TOOL_END),
                ))

            search_start = end_idx + len(cls.TOOL_END)

        return calls

    @classmethod
    def _parse_content(cls, raw: str):
        json_start = raw.find("{")

        if json_start == -1:
            tool_name = raw.strip().split()[0] if raw.strip() else None
            return tool_name, "{}", {}

        tool_name = raw[:json_start].strip()  # type: ignore
        if not tool_name:
            return None, "", {}

        tool_name = tool_name.split()[0].strip()
        json_str = raw[json_start:].strip()  # type: ignore

        brace_depth = 0
        json_end = None
        for i, ch in enumerate(json_str):
            if ch == "{":
                brace_depth += 1
            elif ch == "}":
                brace_depth -= 1
                if brace_depth == 0:
                    json_end = i + 1
                    break

        if json_end is not None:
            json_str = json_str[:json_end]

        try:
            args = json.loads(json_str)
        except json.JSONDecodeError:
            fixed = json_str.replace("'", '"')
            try:
                args = json.loads(fixed)
                json_str = fixed
            except json.JSONDecodeError:
                args = {"raw": json_str}

        return tool_name, json_str, args

    @classmethod
    def extract_text_segments(cls, full_response: str) -> dict:
        result = {"before": "", "tool_calls": [], "after": ""}

        tool_call = cls.parse(full_response)
        if tool_call is None:
            result["after"] = full_response
            return result

        result["before"] = full_response[:tool_call.start_pos].strip()
        result["tool_calls"] = cls.parse_all(full_response)

        last_result_end = full_response.rfind(cls.TOOL_END)
        if last_result_end != -1:
            after_text = full_response[last_result_end + len(cls.TOOL_END):].strip()  # type: ignore
            after_text = after_text.replace("<|end|>", "").strip()
            result["after"] = after_text

        return result
