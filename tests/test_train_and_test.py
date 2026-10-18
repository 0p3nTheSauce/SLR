from pathlib import Path

import pytest

from src.configs import load_config
from src.run_types import CONFIGS_PATH
from src.train_and_test import build_admin, get_parser

CONFIG = CONFIGS_PATH / "asl100/MViTv2_S_e/exp000.toml"


def _admin_args(*extra: str, config: Path = CONFIG) -> list[str]:
    return ["MViTv2_S_e", "asl100", str(config), *extra]


def test_default_run_name_is_config_stem_and_seed(tmp_path: Path) -> None:
    admin = build_admin(get_parser().parse_args(_admin_args("--seed", "3")), runs_path=tmp_path)

    assert admin.exp_no == "exp000_s3"
    assert admin.seed == 3
    assert Path(admin.save_path) == tmp_path / "asl100/MViTv2_S_e/exp000_s3/checkpoints"
    assert Path(admin.save_path).is_dir()
    assert Path(admin.config_path).is_absolute()


def test_run_name_overrides_directory(tmp_path: Path) -> None:
    admin = build_admin(get_parser().parse_args(_admin_args("-n", "mine")), runs_path=tmp_path)

    assert Path(admin.save_path).parent.name == "mine"


def test_admin_loads_with_config(tmp_path: Path) -> None:
    config = load_config(build_admin(get_parser().parse_args(_admin_args()), runs_path=tmp_path))

    assert config.admin.model == "MViTv2_S_e"


def test_refuses_existing_checkpoints(tmp_path: Path) -> None:
    args = get_parser().parse_args(_admin_args())
    save_path = Path(build_admin(args, runs_path=tmp_path).save_path)
    (save_path / "best.pth").touch()

    with pytest.raises(FileExistsError):
        build_admin(args, runs_path=tmp_path)


def test_missing_config(tmp_path: Path) -> None:
    args = get_parser().parse_args(_admin_args(config=tmp_path / "nope.toml"))

    with pytest.raises(FileNotFoundError):
        build_admin(args, runs_path=tmp_path)
