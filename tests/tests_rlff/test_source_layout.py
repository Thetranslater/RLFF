from __future__ import annotations

from pathlib import Path

import rlff.proxy as proxy
import rlff.rewards as rewards
import rlff.rollout as rollout
import rlff.runtime as runtime


def test_split_package_facades_resolve_every_public_symbol() -> None:
    for module in (rewards, proxy, runtime, rollout):
        for name in module.__all__:
            assert getattr(module, name) is not None, f"{module.__name__}.{name} is unavailable"


def test_large_legacy_modules_are_replaced_by_packages() -> None:
    source_root = Path(__file__).resolve().parents[2] / "src" / "rlff"
    for name in ("rewards", "proxy", "runtime", "rollout"):
        assert (source_root / name / "__init__.py").is_file()
        assert not (source_root / f"{name}.py").exists()
