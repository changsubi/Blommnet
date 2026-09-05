"""BloomNet 버전·git revision (06 §2.1.1, 레벨 L-1).

레벨 규약 (06 §2.1 규칙 2 + 정정 A-23):
    이 파일은 `bloomnet.*` 를 **하나도 import 하지 않는다**. 따라서 어느 레벨의 파일이든
    이 파일을 import 할 수 있다. (`constants.py` 와 함께 유일한 예외)
"""

from __future__ import annotations

import subprocess
from pathlib import Path
from typing import Final

__all__ = ["__version__", "GIT_REVISION_UNKNOWN", "git_revision", "short_revision"]

__version__: Final[str] = "0.1.0"

GIT_REVISION_UNKNOWN: Final[str] = "unknown"

_REPO_ROOT: Final[Path] = Path(__file__).resolve().parent.parent


def git_revision(*, repo_root: "Path | str | None" = None) -> str:
    """현재 소스 트리의 git commit SHA(40자)를 반환한다.

    git 저장소가 아니거나 git 실행이 불가능하면 ``"unknown"`` 을 반환한다
    (매니페스트 재현성 기록용이므로 예외를 올리지 않는다).
    dirty 여부는 접미사 ``"-dirty"`` 로 표시한다.
    """
    root = Path(repo_root) if repo_root is not None else _REPO_ROOT
    try:
        sha = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if sha.returncode != 0:
            return GIT_REVISION_UNKNOWN
        rev = sha.stdout.strip()
        if not rev:
            return GIT_REVISION_UNKNOWN
        dirty = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=str(root),
            capture_output=True,
            text=True,
            timeout=5.0,
            check=False,
        )
        if dirty.returncode == 0 and dirty.stdout.strip():
            rev += "-dirty"
        return rev
    except (OSError, subprocess.SubprocessError):
        return GIT_REVISION_UNKNOWN


def short_revision(n: int = 7, *, repo_root: "Path | str | None" = None) -> str:
    """`run_name = f"{mode}_{git_rev[:7]}_{timestamp}"` (06 §4.2) 용 축약 revision."""
    return git_revision(repo_root=repo_root)[:n]
