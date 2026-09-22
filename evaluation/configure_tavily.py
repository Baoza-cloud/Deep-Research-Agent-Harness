"""Securely store a Tavily key in evaluation/.env using hidden input."""

from __future__ import annotations

import getpass
import os
from pathlib import Path


ENV_PATH = Path(__file__).resolve().parent / ".env"


def main() -> int:
    key = getpass.getpass("Tavily API Key（输入不可见）: ").strip()
    if not key:
        print("未写入：Key 为空。")
        return 2
    if not key.startswith("tvly-"):
        print("未写入：Tavily Key 通常以 tvly- 开头，请检查复制内容。")
        return 2

    existing: list[str] = []
    if ENV_PATH.exists():
        existing = [
            line
            for line in ENV_PATH.read_text(encoding="utf-8").splitlines()
            if not line.lstrip().startswith("TAVILY_API_KEY=")
        ]
    existing.append(f"TAVILY_API_KEY={key}")
    ENV_PATH.write_text("\n".join(existing) + "\n", encoding="utf-8")
    os.chmod(ENV_PATH, 0o600)
    print(f"已安全写入 {ENV_PATH}；文件权限为 600，且 .gitignore 会忽略 .env。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
