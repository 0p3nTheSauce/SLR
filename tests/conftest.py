from pathlib import Path

import pytest

from src.models.og_mvit import WEIGHTS_PATH_16x4, WEIGHTS_PATH_32x3

WEIGHT_PATHS: tuple[Path, ...] = (WEIGHTS_PATH_16x4, WEIGHTS_PATH_32x3)


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "requires_weights: needs real pretrained checkpoint files on disk; "
        "skipped when running on a machine without them (see CLAUDE.md's two-machine setup).",
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: list[pytest.Item]
) -> None:
    if all(p.exists() for p in WEIGHT_PATHS):
        return
    skip_no_weights = pytest.mark.skip(
        reason="pretrained weight files not found on this machine"
    )
    for item in items:
        if "requires_weights" in item.keywords:
            item.add_marker(skip_no_weights)
