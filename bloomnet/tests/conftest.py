"""pytest 공통 설정 — 06 §5.0 "공통 규약" `[동결]`.

autouse fixture 가 매 테스트마다
  (1) ``torch.cuda.is_available() == False`` 를 assert  (헌법 C-5.2 — GPU 절대 금지),
  (2) ``torch.manual_seed(0)``,
  (3) ``torch.set_num_threads(4)``
를 수행한다.

기존 테스트 파일들은 각자 같은 취지의 autouse fixture 를 이미 갖고 있다(작성 시점에
conftest 소유자가 정해지지 않았기 때문). 중복은 무해하며 — conftest fixture 가 먼저,
모듈 fixture 가 나중에 돈다 — 모듈 쪽이 자기 seed 를 다시 잡으면 그쪽이 이긴다.

마커
----
``data`` / ``slow`` 는 ``pyproject.toml`` 에 등록되어 있다.
CI 기본 게이트는 ``-m "not data and not slow"`` 다.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Iterator, List

import pytest
import torch

#: 06 §5.0 — 미설치 시 export 게이트(T25, 04 §9.4)가 검증되지 않는다.
DEPLOY_PACKAGES = ("onnx", "onnxscript", "onnxruntime", "onnxsim", "polygraphy")
DEV_PACKAGES = ("pytest", "pytest_cov", "ruff")

#: 테스트 임시 파일 루트. ``/tmp`` 이 아니라 저장소 안에 만든다
#: (헌법: ``<repo_root>`` 밖 쓰기 금지).
TMP_ROOT = Path(__file__).resolve().parents[2] / ".pytest_tmp"


@pytest.fixture(autouse=True)
def _cpu_only_deterministic() -> Iterator[None]:
    """헌법 C-5.2 GPU 차단 + 결정론 + 스레드 상한."""
    assert not torch.cuda.is_available(), (
        "헌법 C-5.2 위반: GPU 가 보인다. CUDA_VISIBLE_DEVICES=\"\" 로 실행하라."
    )
    torch.manual_seed(0)
    torch.set_num_threads(4)
    yield


def pytest_terminal_summary(terminalreporter, exitstatus, config) -> None:  # noqa: ANN001
    """★ 06 §5.0 격리 규칙 2 — 미설치 의존성을 **명시 출력**한다.

    조용한 skip 을 "통과" 로 오독하지 않게 하기 위한 것이다. 릴리스 태그에는 이
    상태를 남기지 않는다.
    """
    missing_deploy: List[str] = [
        p for p in DEPLOY_PACKAGES if importlib.util.find_spec(p) is None
    ]
    missing_dev: List[str] = [p for p in DEV_PACKAGES if importlib.util.find_spec(p) is None]
    if not (missing_deploy or missing_dev):
        return
    terminalreporter.write_sep("=", "의존성 리포트 (06 §5.0)")
    if missing_deploy:
        terminalreporter.write_line(
            f"{', '.join(missing_deploy)} 미설치 — export 게이트(T25 slow, 04 §9.4) **미검증** "
            "(requirements-deploy.txt)"
        )
    if missing_dev:
        terminalreporter.write_line(f"{', '.join(missing_dev)} 미설치 (requirements-dev.txt)")


@pytest.fixture()
def kw_tmp(request: pytest.FixtureRequest) -> Iterator[Path]:
    """저장소 안(``k_water/.pytest_tmp``)의 임시 디렉터리. 테스트 종료 시 삭제한다."""
    d = TMP_ROOT / f"{request.node.name[:60].replace('/', '_')}"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True, exist_ok=True)
    try:
        yield d
    finally:
        shutil.rmtree(d, ignore_errors=True)
        try:
            TMP_ROOT.rmdir()  # 비었을 때만 (다른 테스트가 쓰고 있으면 실패하고 그대로 둔다)
        except OSError:
            pass
