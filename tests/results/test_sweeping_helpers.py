from pathlib import Path

import pandas as pd
import pytest

from src.results.sweeping.helpers import TunedParam, load_tuned_params, suggest_ranges


def test_load_tuned_params_skips_fixed_and_single_values(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text(
        """parameters:
  lr: {distribution: log_uniform_values, min: 1e-5, max: 1e-1}
  num_ops: {values: [1, 2, 4]}
  only_one: {values: [3]}
  batch_size: {value: 8}
"""
    )
    tuned = load_tuned_params(config)
    assert list(tuned) == ["lr", "num_ops"]
    assert tuned["lr"] == TunedParam("lr", "log_uniform_values", low=1e-5, high=1e-1)
    assert tuned["num_ops"].values == (1, 2, 4)


def test_load_tuned_params_rejects_unknown_distribution(tmp_path: Path) -> None:
    config = tmp_path / "config.yaml"
    config.write_text("parameters:\n  x: {distribution: q_normal, min: 0, max: 1}\n")
    with pytest.raises(ValueError, match="q_normal"):
        load_tuned_params(config)


class TestSuggestRanges:
    def _df(self, xs: list[float], losses: list[float]) -> pd.DataFrame:
        return pd.DataFrame({"x": xs, "best_val_loss": losses})

    def test_narrows_to_padded_top_k_spread(self) -> None:
        p = {"x": TunedParam("x", "uniform", low=0.0, high=1.0)}
        df = self._df([0.4, 0.5, 0.9, 0.1], [1.0, 1.1, 3.0, 3.0])
        row = suggest_ranges(df, p, top_k=2).loc["x"]
        assert row.suggested_min == pytest.approx(0.3)
        assert row.suggested_max == pytest.approx(0.6)
        assert row.best_near_edge == ""

    def test_extends_past_bound_when_best_is_near_it(self) -> None:
        p = {"x": TunedParam("x", "log_uniform_values", low=1e-6, high=1e-1)}
        df = self._df([0.09, 1e-3, 1e-5], [1.0, 1.1, 3.0])
        row = suggest_ranges(df, p, top_k=2).loc["x"]
        assert row.best_near_edge == "high"
        assert row.suggested_max == pytest.approx(10 ** (-1 + 0.5))  # pad = 0.1 * 5 decades
        assert row.suggested_min == pytest.approx(10 ** (-3 - 0.5))

    def test_int_uniform_rounds_outward_and_categorical_skipped(self) -> None:
        params = {
            "n": TunedParam("n", "int_uniform", low=0, high=20),
            "c": TunedParam("c", "categorical", values=(1, 2)),
        }
        df = pd.DataFrame({"n": [9, 11, 20], "c": [1, 2, 1], "best_val_loss": [1.0, 1.1, 3.0]})
        result = suggest_ranges(df, params, top_k=2)
        assert list(result.index) == ["n"]
        assert (result.loc["n"].suggested_min, result.loc["n"].suggested_max) == (7.0, 13.0)
