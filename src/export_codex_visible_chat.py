"""Export user-visible Codex messages from a local rollout JSONL file.

The source rollout is read-only. Internal reasoning, tool traffic, system/developer
messages, environment snapshots, and token accounting are intentionally excluded.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo


SKIP_PREFIXES = (
    "<environment_context>",
    "<recommended_plugins>",
    "<permissions instructions>",
    "<app-context>",
    "<skills_instructions>",
)


def visible_text(payload: dict) -> str:
    pieces: list[str] = []
    for item in payload.get("content", []):
        if item.get("type") in {"input_text", "output_text", "text", "Text"}:
            value = item.get("text")
            if isinstance(value, str):
                pieces.append(value)
    return "\n".join(pieces).strip()


def local_time(timestamp: str) -> str:
    value = datetime.fromisoformat(timestamp.replace("Z", "+00:00"))
    return value.astimezone(ZoneInfo("Asia/Shanghai")).strftime("%Y-%m-%d %H:%M:%S")


def export(sources: list[Path], destination: Path) -> tuple[int, int, str]:
    digest = hashlib.sha256()
    messages: list[tuple[str, str, str, str]] = []
    seen_ids: set[str] = set()

    for source in sources:
        with source.open("rb") as raw:
            for binary_line in raw:
                digest.update(binary_line)
                try:
                    record = json.loads(binary_line)
                except (UnicodeDecodeError, json.JSONDecodeError):
                    continue
                if record.get("type") != "response_item":
                    continue
                payload = record.get("payload", {})
                if payload.get("type") != "message":
                    continue
                role = payload.get("role")
                if role not in {"user", "assistant"}:
                    continue
                message_id = str(payload.get("id", ""))
                if message_id and message_id in seen_ids:
                    continue
                text = visible_text(payload)
                if not text or text.lstrip().startswith(SKIP_PREFIXES):
                    continue
                if message_id:
                    seen_ids.add(message_id)
                phase = str(payload.get("phase", ""))
                messages.append((local_time(record["timestamp"]), role, phase, text))

    messages.sort(key=lambda item: item[0])

    user_count = sum(role == "user" for _, role, _, _ in messages)
    assistant_count = sum(role == "assistant" for _, role, _, _ in messages)
    lines = [
        "# Codex 对话可读备份：傅里叶仿真",
        "",
        f"- 原任务 ID：`01a046de-7c0d-7811-b7af-7f0e541ea434`",
        f"- 导出时间：`{datetime.now(ZoneInfo('Asia/Shanghai')).strftime('%Y-%m-%d %H:%M:%S')}`",
        f"- 用户消息：`{user_count}` 条",
        f"- 助手可见消息：`{assistant_count}` 条",
        f"- 合并源日志：`{len(sources)}` 个",
        f"- 源日志 SHA-256：`{digest.hexdigest()}`",
        "- 范围：用户可见消息、助手 commentary 和 final；不含内部推理、工具流量及环境配置。",
        "",
        "---",
        "",
    ]
    for timestamp, role, phase, text in messages:
        if role == "user":
            heading = f"## {timestamp}｜用户"
        else:
            label = "过程更新" if phase == "commentary" else "答复"
            heading = f"## {timestamp}｜助手·{label}"
        lines.extend((heading, "", text, "", "---", ""))
    destination.write_text("\n".join(lines), encoding="utf-8")
    return user_count, assistant_count, digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("destination", type=Path)
    parser.add_argument("sources", type=Path, nargs="+")
    args = parser.parse_args()
    user_count, assistant_count, digest = export(args.sources, args.destination)
    print(json.dumps({
        "destination": str(args.destination.resolve()),
        "user_messages": user_count,
        "assistant_messages": assistant_count,
        "source_sha256": digest,
    }, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
