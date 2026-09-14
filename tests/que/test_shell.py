from pathlib import Path

import pytest

from src.que.shell import get_filters_drop_keys, unpack_filters


class TestUnpackFilters:
    def test_flattens_single_level(self) -> None:
        is_pos = lambda x: x > 0
        key_sets, criterions = unpack_filters({"a": is_pos})
        assert key_sets == [["a"]]
        assert criterions == [is_pos]

    def test_flattens_nested_dict(self) -> None:
        is_a = lambda x: x == "a"
        is_b = lambda x: x == "b"
        key_sets, criterions = unpack_filters({"outer": {"inner": is_a, "other": is_b}})
        assert key_sets == [["outer", "inner"], ["outer", "other"]]
        assert criterions == [is_a, is_b]

    def test_rejects_non_dict_non_callable_leaf(self) -> None:
        with pytest.raises(TypeError):
            unpack_filters({"a": 5})  # type: ignore[dict-item]

    def test_empty_filters(self) -> None:
        assert unpack_filters({}) == ([], [])


class TestGetFiltersDropKeys:
    def test_loads_filters_and_drop_keys(self, tmp_path: Path) -> None:
        filters_file = tmp_path / "filters.py"
        filters_file.write_text(
            "filters = {'admin': {'model': lambda x: x == 'S3D'}}\n"
            "drop_keys = [['results', 'check_name']]\n"
        )

        filters, drop_keys = get_filters_drop_keys(filters_file)

        assert drop_keys == [["results", "check_name"]]
        assert filters["admin"]["model"]("S3D") is True
        assert filters["admin"]["model"]("MViTv2_S") is False

    def test_missing_required_attribute_raises(self, tmp_path: Path) -> None:
        filters_file = tmp_path / "filters.py"
        filters_file.write_text("filters = {}\n")  # missing drop_keys

        with pytest.raises(AttributeError):
            get_filters_drop_keys(filters_file)
