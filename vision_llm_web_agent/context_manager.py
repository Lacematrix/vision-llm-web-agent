"""Context window management for LLM calls."""

from __future__ import annotations

import json
from typing import Any, Callable, Dict, List

from .prompts import build_context_summary_history_message


class ContextWindowManager:
    """Manages conversation history size with windowing and summary compression."""

    def __init__(
        self,
        max_history_chars: int = 32000,
        keep_last_messages: int = 12,
        max_summary_chars: int = 1200,
    ) -> None:
        self.max_history_chars = max_history_chars
        self.keep_last_messages = keep_last_messages
        self.max_summary_chars = max_summary_chars

    @staticmethod
    def _message_to_text(message: Dict[str, Any]) -> str:
        role = message.get("role", "user")
        content = message.get("content", "")

        if isinstance(content, list):
            parts: List[str] = []
            for item in content:
                if not isinstance(item, dict):
                    parts.append(str(item))
                    continue
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                elif item.get("type") == "image_url":
                    parts.append("[image]")
                else:
                    parts.append(str(item))
            content_text = "\n".join(parts)
        elif isinstance(content, dict):
            content_text = json.dumps(content, ensure_ascii=False)
        else:
            content_text = str(content)

        return f"{role}: {content_text}"

    def estimate_history_chars(self, history: List[Dict[str, Any]]) -> int:
        return sum(len(self._message_to_text(msg)) for msg in history)

    def compress_history(
        self,
        history: List[Dict[str, Any]],
        summarizer: Callable[[str, int], str],
    ) -> List[Dict[str, Any]]:
        """Compress old history into one summary message when over budget."""
        if not history:
            return history

        if self.estimate_history_chars(history) <= self.max_history_chars:
            return history

        if len(history) <= self.keep_last_messages:
            # Hard truncate fallback when history is too short to summarize/split.
            return history[-max(1, self.keep_last_messages // 2):]

        head = history[:-self.keep_last_messages]
        tail = history[-self.keep_last_messages:]

        head_text = "\n\n".join(self._message_to_text(msg) for msg in head)
        summary = summarizer(head_text, self.max_summary_chars).strip()
        if not summary:
            summary = "Summary unavailable due to model response issue."

        summary_message = {
            "role": "assistant",
            "content": build_context_summary_history_message(summary),
        }

        compressed = [summary_message] + tail

        # Safety fallback: if still too large, progressively keep fewer tail messages.
        while (
            self.estimate_history_chars(compressed) > self.max_history_chars
            and len(compressed) > 2
        ):
            compressed = [summary_message] + compressed[2:]

        return compressed
