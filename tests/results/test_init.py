from pydantic import BaseModel

from src.results import match, same_augs


class _Aug(BaseModel):
    type: str
    frame_size: int


class TestMatch:
    def test_dict_subset_matches(self) -> None:
        obj = {"a": 1, "b": 2, "c": 3}
        assert match(obj, {"a": 1, "b": 2})

    def test_dict_mismatch_fails(self) -> None:
        obj = {"a": 1, "b": 2}
        assert not match(obj, {"a": 1, "b": 3})

    def test_both_none(self) -> None:
        assert match(None, None)

    def test_one_none_fails(self) -> None:
        assert not match(None, {"a": 1})
        assert not match({"a": 1}, None)

    def test_pydantic_model_dumped_before_comparison(self) -> None:
        obj = _Aug(type="Centre_crop", frame_size=224)
        assert match(obj, {"type": "Centre_crop"})
        assert not match(obj, {"type": "Random_crop"})


class TestSameAugs:
    def test_equal_lists_of_dicts(self) -> None:
        augs = [{"type": "Centre_crop", "frame_size": 224}]
        assert same_augs(augs, list(augs))

    def test_different_lengths(self) -> None:
        assert not same_augs([{"type": "a"}], [{"type": "a"}, {"type": "b"}])

    def test_different_order_fails(self) -> None:
        a = [{"type": "a"}, {"type": "b"}]
        b = [{"type": "b"}, {"type": "a"}]
        assert not same_augs(a, b)

    def test_pydantic_models_compared_by_value(self) -> None:
        a = [_Aug(type="Centre_crop", frame_size=224)]
        b = [_Aug(type="Centre_crop", frame_size=224)]
        assert same_augs(a, b)

    def test_empty_lists_are_equal(self) -> None:
        assert same_augs([], [])
