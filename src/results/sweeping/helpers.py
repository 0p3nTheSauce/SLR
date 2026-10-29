"""Helpers for `suggest_sweep.ipynb` and `param_importance.ipynb`: flatten a wandb sweep's
finished runs (from the v1 Que or Que_v2's run.json files) into a DataFrame of sampled
hyperparameters vs. results, suggest refined parameter ranges for the next sweep iteration,
and rate each parameter's importance."""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal, get_args

import numpy as np
import pandas as pd
import yaml
from scipy import stats
from sklearn.ensemble import RandomForestRegressor

from src.que.core import CompExpInfo
from src.sweeping import (
    SWEEP_KEY_MAP_ATTR,
    SweepKeyMap,
    build_base_config,
    extract_sweep_values,
)

Distribution = Literal["log_uniform_values", "uniform", "int_uniform", "categorical"]
NumericDistribution = Literal["log_uniform_values", "uniform", "int_uniform"]


@dataclass(frozen=True)
class TunedParam:
    """A swept (not fixed) entry of a sweep yaml's `parameters` block. `low`/`high` are
    set for numeric distributions, `values` for categorical ones."""

    name: str
    distribution: Distribution
    low: float | None = None
    high: float | None = None
    values: tuple[Any, ...] | None = None

    @property
    def is_log(self) -> bool:
        return self.distribution == "log_uniform_values"


@dataclass(frozen=True)
class SweepRef:
    """A sweep to analyse: display label, wandb sweep id, and its `configfiles/sweeps`
    experiment directory (holding `config.yaml` and `base.py`)."""

    label: str
    sweep_id: str
    sweep_dir: Path


def load_tuned_params(config_yaml: Path) -> dict[str, TunedParam]:
    """Read the swept parameters from a sweep yaml, skipping fixed `{value: ...}` entries
    and single-option `values` lists."""
    with open(config_yaml) as f:
        parameters: dict[str, dict[str, Any]] = yaml.safe_load(f)["parameters"]

    tuned: dict[str, TunedParam] = {}
    for name, spec in parameters.items():
        if "distribution" in spec:
            dist = spec["distribution"]
            if dist not in get_args(NumericDistribution):
                raise ValueError(f"{config_yaml}: unsupported distribution {dist!r} for {name!r}")
            tuned[name] = TunedParam(name, dist, low=float(spec["min"]), high=float(spec["max"]))
        elif len(spec.get("values", [])) > 1:
            tuned[name] = TunedParam(name, "categorical", values=tuple(spec["values"]))
    return tuned


def load_sweep_key_map(base_py: Path) -> SweepKeyMap:
    """Load just the `sweep_key_map` attribute from a sweep's `base.py`."""
    return build_base_config(base_py, [SWEEP_KEY_MAP_ATTR])[SWEEP_KEY_MAP_ATTR]


def load_sweep_run_jsons(sweep_runs_dir: Path) -> list[CompExpInfo]:
    """The finished trials of a Que_v2 sweep, from each trial's `run.json` under
    `src/runs/<split>/<model>/sweep_<id>/job<job id>/` (unfinished trials have none), in job
    order. Que_v2 sweeps aren't in the v1 Que's `old_runs`, so `find_runs` doesn't see them."""
    paths = sorted(sweep_runs_dir.glob("job*/run.json"), key=lambda p: int(p.parent.name[3:]))
    return [CompExpInfo.model_validate_json(p.read_text()) for p in paths]


def sweep_frame(
    runs: Sequence[CompExpInfo], sweep_key_map: SweepKeyMap, param_names: list[str]
) -> pd.DataFrame:
    """One row per finished trial, in the order given (Que completion order for runs from
    `find_runs`): run id, split, best val loss/acc, test loss/top-1, then each swept
    parameter's sampled value (recovered from the nested run config)."""
    return pd.DataFrame(
        [
            {
                "run_id": run.wandb.run_id,
                "split": run.admin.split,
                "best_val_loss": run.results.best_val_loss,
                "best_val_acc": run.results.best_val_acc,
                "test_loss": run.results.test.average_loss,
                "test_top1": run.results.test.top_k_per_instance_acc.top1 * 100,
            }
            | extract_sweep_values(run.model_dump(), sweep_key_map, param_names)
            for run in runs
        ]
    )


def _param_matrix(df: pd.DataFrame, params: dict[str, TunedParam]) -> np.ndarray:
    """The swept parameters as a float matrix, categoricals as their index in `values`."""
    columns = [
        df[p.name].map(list(p.values).index) if p.values is not None else df[p.name]
        for p in params.values()
    ]
    return np.column_stack(columns).astype(float)


def param_importance(
    df: pd.DataFrame,
    params: dict[str, TunedParam],
    metric: str = "best_val_loss",
    n_boot: int = 200,
    n_trees: int = 300,
    interval: tuple[float, float] = (5, 95),
    seed: int = 0,
) -> pd.DataFrame:
    """Each swept parameter's random-forest importance for `metric`, and its Kendall tau with it.

    Importance is a random-forest regressor's impurity importance (mean decrease in variance,
    summing to 1 over the parameters), as in W&B's parameter-importance panel. With few trials
    one forest's importances are unstable, so a forest is fitted to each of `n_boot` bootstrap
    resamples of the trials: `importance` is their mean and `importance_low`/`_high` the
    `interval` percentiles. Impurity importance under-rates a parameter with few distinct
    values (e.g. a two-valued categorical); `oob_permutation_importance` doesn't. `tau`/`tau_p`
    are on all trials; negative means higher values went with lower `metric`. Categoricals are
    encoded by their position in `values`. Rows are sorted by importance.
    """
    x, y = _param_matrix(df, params), df[metric].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    boot = np.array(
        [
            RandomForestRegressor(n_estimators=n_trees, random_state=b)
            .fit(x[idx], y[idx])
            .feature_importances_
            for b, idx in enumerate(rng.integers(0, len(y), (n_boot, len(y))))
        ]
    )
    return (
        pd.DataFrame(
            {
                "importance": boot.mean(axis=0),
                "importance_low": np.percentile(boot, interval[0], axis=0),
                "importance_high": np.percentile(boot, interval[1], axis=0),
            },
            index=pd.Index(list(params), name="param"),
        )
        .join(kendall_taus(df, params, metric))
        .sort_values("importance", ascending=False)
    )


def kendall_taus(
    df: pd.DataFrame, params: dict[str, TunedParam], metric: str = "best_val_loss"
) -> pd.DataFrame:
    """Each swept parameter's Kendall tau (`tau`) and its p-value (`tau_p`) with `metric`, in
    `params` order; negative means higher values went with lower `metric`. Categoricals are
    encoded by their position in `values`, so they should be listed in ascending order."""
    x, y = _param_matrix(df, params), df[metric].to_numpy(dtype=float)
    taus = [tuple(map(float, stats.kendalltau(x[:, j], y))) for j in range(x.shape[1])]
    return pd.DataFrame(
        {"tau": [tau for tau, _ in taus], "tau_p": [p for _, p in taus]},
        index=pd.Index(list(params), name="param"),
    )


def oob_permutation_importance(
    df: pd.DataFrame,
    params: dict[str, TunedParam],
    metric: str = "best_val_loss",
    n_boot: int = 100,
    n_trees: int = 100,
    n_repeats: int = 5,
    seed: int = 0,
) -> pd.Series:
    """Out-of-bag permutation importance, a check on `param_importance`'s impurity importance.

    For each bootstrap resample, a forest is fitted to it, and each parameter is shuffled among
    the trials left out of it; the importance is the mean increase in squared error that causes,
    as a share of `metric`'s variance. Unlike impurity importance it doesn't favour parameters
    with many distinct values, and it can be negative (the parameter doesn't help predict).
    """
    x, y = _param_matrix(df, params), df[metric].to_numpy(dtype=float)
    rng = np.random.default_rng(seed)
    increases: list[np.ndarray] = []
    for b in range(n_boot):
        idx = rng.integers(0, len(y), len(y))
        oob = np.setdiff1d(np.arange(len(y)), idx)
        if len(oob) < 2:
            continue
        forest = RandomForestRegressor(n_estimators=n_trees, random_state=b).fit(x[idx], y[idx])
        base = np.mean((forest.predict(x[oob]) - y[oob]) ** 2)
        row = np.zeros(x.shape[1])
        for j in range(x.shape[1]):
            for _ in range(n_repeats):
                shuffled = x[oob].copy()
                shuffled[:, j] = rng.permutation(shuffled[:, j])
                row[j] += np.mean((forest.predict(shuffled) - y[oob]) ** 2) - base
        increases.append(row / n_repeats)
    return pd.Series(np.mean(increases, axis=0) / y.var(), index=pd.Index(list(params), name="param"))


def _to_search_space(p: TunedParam, v: float) -> float:
    return float(np.log10(v)) if p.is_log else float(v)


def _from_search_space(p: TunedParam, v: float) -> float:
    return 10**v if p.is_log else v


def suggest_ranges(
    df: pd.DataFrame,
    params: dict[str, TunedParam],
    top_k: int = 10,
    margin: float = 0.1,
    metric: Literal["best_val_loss", "test_loss"] = "best_val_loss",
) -> pd.DataFrame:
    """Suggest a narrowed range for each numeric swept parameter from the `top_k` trials
    with the lowest `metric`.

    The suggestion is the top-k spread padded by `margin` x the current range width (in
    log10 space for log-uniform parameters), clipped to the current range. The exception
    is when the single best trial lies within `margin` of a current bound
    (`best_near_edge`): that side is extended past the bound instead, since the optimum
    may lie outside what was searched. Categorical parameters are skipped -- compare their
    top-k value counts instead.
    """
    top = df.nsmallest(top_k, metric)
    best = top.iloc[0]
    rows = []
    for p in params.values():
        if p.distribution == "categorical" or p.low is None or p.high is None:
            continue
        lo, hi = _to_search_space(p, p.low), _to_search_space(p, p.high)
        t_lo, t_hi = _to_search_space(p, float(top[p.name].min())), _to_search_space(p, float(top[p.name].max()))
        b = _to_search_space(p, best[p.name])
        pad = margin * (hi - lo)

        near_low, near_high = b - lo < pad, hi - b < pad
        s_lo = lo - pad if near_low else max(lo, t_lo - pad)
        s_hi = hi + pad if near_high else min(hi, t_hi + pad)
        s_min, s_max = _from_search_space(p, s_lo), _from_search_space(p, s_hi)
        if p.distribution == "int_uniform":
            s_min, s_max = float(np.floor(s_min)), float(np.ceil(s_max))

        rows.append(
            {
                "param": p.name,
                "distribution": p.distribution,
                "current_min": p.low,
                "current_max": p.high,
                f"top{top_k}_min": top[p.name].min(),
                f"top{top_k}_median": top[p.name].median(),
                f"top{top_k}_max": top[p.name].max(),
                "best": best[p.name],
                "suggested_min": s_min,
                "suggested_max": s_max,
                "best_near_edge": "low" if near_low else "high" if near_high else "",
            }
        )
    return pd.DataFrame(rows).set_index("param")
