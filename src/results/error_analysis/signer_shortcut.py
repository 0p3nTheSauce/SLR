"""Helpers for `signer_shortcut.ipynb`: does a model, when wrong, predict a gloss the test video's
signer (or source) performs in training?

The rows are one per test prediction (`prediction_table`). The main test, `permutation_test`,
shuffles the wrong predictions among the wrong videos, which keeps how often each gloss is
predicted and each signer's gloss set; `strata` restricts the shuffle to within a group (e.g.
source), so the remaining excess can't come from that group. `conditional_logit` asks the same
question per candidate gloss, with covariates.
"""

from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Literal, TypeAlias

import numpy as np
import numpy.typing as npt
import pandas as pd
from scipy.optimize import minimize
from scipy.special import logsumexp

from src.results.dataset_analysis.find_duplicates import Member, Report
from src.run_types import InstanceTopK

BoolArray = npt.NDArray[np.bool_]
FloatArray = npt.NDArray[np.float64]
IntArray = npt.NDArray[np.int64]
GlossSets: TypeAlias = Mapping[int | str, set[str]]  # signer id or source -> training glosses


def train_gloss_sets(
    members: Iterable[Member], key: Literal['signer_id', 'source'], exclude: Iterable[str] = ()
) -> dict[int | str, set[str]]:
    """Map each value of `key` (`signer_id` or `source`) to the glosses it has a training video of,
    skipping the video ids in `exclude`."""
    skip = set(exclude)
    sets: defaultdict[int | str, set[str]] = defaultdict(set)
    for m in members:
        if m.set == 'train' and m.video_id not in skip:
            sets[getattr(m, key)].add(m.gloss)
    return dict(sets)


def train_copy_glosses(report: Report) -> dict[str, set[str]]:
    """Glosses of each video's identical training copies (exact duplicates only, itself excluded)."""
    copies: defaultdict[str, set[str]] = defaultdict(set)
    for group in report.groups:
        if group.kind != 'exact':
            continue
        for member in group.members:
            copies[member.video_id] |= {
                m.gloss for m in group.members if m.set == 'train' and m.video_id != member.video_id
            }
    return dict(copies)


def cross_gloss_pairs(report: Report) -> set[frozenset[str]]:
    """Gloss pairs filed on an identical video anywhere in the dataset: the same sign under two
    glosses, so a confusion between them isn't evidence of a shortcut."""
    pairs: set[frozenset[str]] = set()
    for group in report.groups:
        if group.kind != 'exact':
            continue
        glosses = sorted({m.gloss for m in group.members})
        pairs |= {frozenset((a, b)) for i, a in enumerate(glosses) for b in glosses[i + 1:]}
    return pairs


def prediction_table(
    predictions: Sequence[InstanceTopK],
    members: Mapping[str, Member],
    classes: Sequence[str],
    copies: Mapping[str, set[str]],
    top_n: int = 5,
) -> pd.DataFrame:
    """One row per prediction: the test video's signer, source and variation, its true and
    predicted gloss, `copy_pred` (the prediction is the gloss of one of its identical training
    copies), `has_train_copy` and its `top_n` predicted glosses."""
    rows: list[dict[str, object]] = []
    for p in predictions:
        m = members[p.video_id]
        gloss = classes[p.target]
        assert gloss == m.gloss, f'{p.video_id}: prediction target {gloss} but split says {m.gloss}'
        top = [classes[i] for i in p.topk_idxs[:top_n]]
        train_copies = copies.get(p.video_id, set())
        rows.append({
            'video_id': p.video_id,
            'signer': m.signer_id,
            'source': m.source,
            'variation': m.variation_id,
            'gloss': gloss,
            'predicted': top[0],
            'top1': top[0] == gloss,
            'copy_pred': top[0] in train_copies,
            'has_train_copy': bool(train_copies),
            'top_n': top,
        })
    return pd.DataFrame(rows)


def other_signer_confusions(df: pd.DataFrame) -> list[set[str]]:
    """For each row, the glosses in the top-n of the *other* signers' test videos of its true gloss:
    a signer-independent sign of which glosses look like it."""
    by_gloss: defaultdict[str, list[tuple[int, list[str]]]] = defaultdict(list)
    for gloss, signer, top in zip(df['gloss'], df['signer'], df['top_n'], strict=True):
        by_gloss[gloss].append((signer, top))
    return [
        {g for s, top in by_gloss[gloss] if s != signer for g in top} - {gloss}
        for gloss, signer in zip(df['gloss'], df['signer'], strict=True)
    ]


def hits(df: pd.DataFrame, key: str, sets: GlossSets, col: str = 'predicted') -> BoolArray:
    """Whether each row's `col` gloss is in the training gloss set of its `key` (signer/source)."""
    return np.array([g in sets.get(k, set()) for k, g in zip(df[key], df[col], strict=True)])


@dataclass(frozen=True)
class PermutationResult:
    """`observed`/`null_mean` are rates, `null_hi` the null's 97.5th percentile, `p` one-sided.
    `expected` is each row's mean hit under the null, so it sums to the expected hits of a subset."""

    n: int
    observed: float
    null_mean: float
    null_hi: float
    p: float
    expected: FloatArray

    def as_row(self) -> dict[str, float]:
        return {
            'n': self.n, 'observed (%)': 100 * self.observed, 'shuffled (%)': 100 * self.null_mean,
            'shuffled 97.5th pct (%)': 100 * self.null_hi, 'ratio': self.observed / self.null_mean,
            'p': self.p,
        }


def _membership(
    df: pd.DataFrame, key: str, sets: GlossSets
) -> tuple[BoolArray, IntArray, IntArray]:
    """A (row's key x predicted gloss) membership matrix and each row's index into both axes."""
    keys = sorted(set(df[key]), key=str)
    glosses = sorted(set(df['predicted']))
    g_idx = {g: i for i, g in enumerate(glosses)}
    table = np.array([[g in sets.get(k, set()) for g in glosses] for k in keys], dtype=bool)
    k_idx = {k: i for i, k in enumerate(keys)}
    return (
        table,
        np.array([k_idx[k] for k in df[key]], dtype=np.int64),
        np.array([g_idx[g] for g in df['predicted']], dtype=np.int64),
    )


def permutation_test(
    df: pd.DataFrame,
    key: str,
    sets: GlossSets,
    n_perm: int = 2000,
    seed: int = 0,
    strata: str | None = None,
) -> PermutationResult:
    """How often `df`'s predicted gloss is in its `key`'s training gloss set, against shuffling the
    predictions among `df`'s rows (within each `strata` group if given)."""
    table, k_idx, g_idx = _membership(df, key, sets)
    observed = float(table[k_idx, g_idx].mean())
    rng = np.random.default_rng(seed)
    groups = [np.arange(len(df))] if strata is None else [
        np.flatnonzero(df[strata].to_numpy() == v) for v in pd.unique(df[strata])
    ]
    perms = np.empty((n_perm, len(df)), dtype=np.int64)
    for rows in groups:
        perms[:, rows] = rows[rng.permuted(np.tile(np.arange(len(rows)), (n_perm, 1)), axis=1)]
    null = table[k_idx[None, :], g_idx[perms]]
    rates = null.mean(axis=1)
    return PermutationResult(
        n=len(df),
        observed=observed,
        null_mean=float(rates.mean()),
        null_hi=float(np.quantile(rates, 0.975)),
        p=float((np.sum(rates >= observed) + 1) / (n_perm + 1)),
        expected=null.mean(axis=0),
    )


@dataclass(frozen=True)
class LogitFit:
    """Coefficients with standard errors, from `conditional_logit`."""

    names: list[str]
    coef: FloatArray
    se: FloatArray

    def table(self) -> pd.DataFrame:
        z = self.coef / self.se
        return pd.DataFrame(
            {
                'coef': self.coef, 'se': self.se, 'odds ratio': np.exp(self.coef),
                'OR 95% low': np.exp(self.coef - 1.96 * self.se),
                'OR 95% high': np.exp(self.coef + 1.96 * self.se), 'z': z,
            },
            index=pd.Index(self.names),
        )


def _logsumexp(u: FloatArray) -> FloatArray:
    return np.asarray(logsumexp(u, axis=1), dtype=np.float64)


def conditional_logit(
    features: FloatArray, chosen: IntArray, available: BoolArray, names: list[str]
) -> LogitFit:
    """Fit a conditional (McFadden) logit: row i chose candidate `chosen[i]` among its `available`
    candidates (rows x candidates), described by `features` (rows x candidates x features), with
    P(c) proportional to exp(features[i, c] . beta)."""
    n, _, k = features.shape
    assert available[np.arange(n), chosen].all(), 'a chosen candidate is unavailable'
    x_chosen = features[np.arange(n), chosen]
    offset = np.where(available, 0.0, -np.inf)

    def utility(beta: FloatArray) -> FloatArray:
        return features @ beta + offset

    def objective(beta: FloatArray) -> tuple[float, FloatArray]:
        u = utility(beta)
        lse = _logsumexp(u)
        prob = np.exp(u - lse[:, None])
        expected = np.einsum('ic,ick->ik', prob, features)
        return float(lse.sum() - (x_chosen @ beta).sum()), (expected - x_chosen).sum(axis=0)

    fit = minimize(objective, np.zeros(k), jac=True, method='BFGS')
    beta: FloatArray = fit.x
    u = utility(beta)
    prob = np.exp(u - _logsumexp(u)[:, None])
    mean = np.einsum('ic,ick->ik', prob, features)
    second = np.einsum('ic,ick,icl->kl', prob, features, features)
    hessian = second - mean.T @ mean
    return LogitFit(names=names, coef=beta, se=np.sqrt(np.diag(np.linalg.inv(hessian))))


CANDIDATE_FEATURES = [
    'same signer',
    'same source',
    'log predicted count',
    'log train count',
    'other-signer confusion',
    'shares identical video',
]


def candidate_features(
    wrong: pd.DataFrame,
    others: Sequence[set[str]],
    classes: Sequence[str],
    signer_sets: GlossSets,
    source_sets: GlossSets,
    train_counts: Mapping[str, int],
    cross: set[frozenset[str]],
) -> tuple[FloatArray, IntArray, BoolArray]:
    """`conditional_logit` inputs for the wrong predictions in `wrong`: every gloss but the true one
    is a candidate, described by `CANDIDATE_FEATURES` (the signer/source performs it in training,
    how often the model predicts it on the *other* wrong videos, its training count, whether it is
    in `others` -- the row's `other_signer_confusions` -- and whether it shares an identical video
    with the true gloss)."""
    idx = {g: i for i, g in enumerate(classes)}
    n, n_glosses = len(wrong), len(classes)
    predicted = np.array([idx[g] for g in wrong['predicted']], dtype=np.int64)
    true = np.array([idx[g] for g in wrong['gloss']], dtype=np.int64)
    pred_count = np.bincount(predicted, minlength=n_glosses).astype(np.float64)
    log_train = np.log1p(np.array([train_counts.get(g, 0) for g in classes], dtype=np.float64))

    def mask(glosses: Iterable[str]) -> BoolArray:
        out = np.zeros(n_glosses, dtype=bool)
        out[[idx[g] for g in glosses]] = True
        return out

    signer_masks = {k: mask(v) for k, v in signer_sets.items()}
    source_masks = {k: mask(v) for k, v in source_sets.items()}
    partners: defaultdict[str, set[str]] = defaultdict(set)
    for pair in cross:
        a, b = sorted(pair)
        partners[a].add(b)
        partners[b].add(a)

    features = np.zeros((n, n_glosses, len(CANDIDATE_FEATURES)))
    empty = np.zeros(n_glosses, dtype=bool)
    for i, (signer, source, gloss) in enumerate(zip(wrong['signer'], wrong['source'], wrong['gloss'], strict=True)):
        own = pred_count.copy()
        own[predicted[i]] -= 1
        features[i, :, 0] = signer_masks.get(signer, empty)
        features[i, :, 1] = source_masks.get(source, empty)
        features[i, :, 2] = np.log1p(own)
        features[i, :, 3] = log_train
        features[i, :, 4] = mask(others[i])
        features[i, :, 5] = mask(partners[gloss])
    available = np.ones((n, n_glosses), dtype=bool)
    available[np.arange(n), true] = False
    return features, predicted, available
