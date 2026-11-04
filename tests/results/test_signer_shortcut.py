import numpy as np
import pandas as pd
import pytest

from src.results.error_analysis.signer_shortcut import (
    GlossSets,
    candidate_features,
    conditional_logit,
    cross_gloss_pairs,
    other_signer_confusions,
    permutation_test,
    train_copy_glosses,
    train_gloss_sets,
)


def instances(rows: list[tuple[str, str, str, int, str]]) -> pd.DataFrame:
    """`load_instances`-like rows of (video_id, gloss, set, signer_id, md5)."""
    return pd.DataFrame(rows, columns=pd.Index(['video_id', 'gloss', 'set', 'signer_id', 'md5']))


def test_train_gloss_sets_keep_only_training_videos_not_excluded() -> None:
    frame = instances([
        ('1', 'a', 'train', 1, 'w'),
        ('2', 'b', 'test', 1, 'x'),
        ('3', 'c', 'train', 1, 'y'),
        ('4', 'd', 'train', 2, 'z'),
    ])
    assert train_gloss_sets(frame, 'signer_id', exclude={'3'}) == {1: {'a'}, 2: {'d'}}


def test_copies_and_cross_pairs_come_from_identical_clips() -> None:
    frame = instances([
        ('1', 'a', 'train', 1, 'x'),
        ('2', 'b', 'test', 1, 'x'),
        ('3', 'c', 'val', 1, 'x'),
        ('4', 'd', 'train', 1, 'y'),
    ])
    assert train_copy_glosses(frame) == {'2': {'a'}, '3': {'a'}}
    assert cross_gloss_pairs(frame) == {frozenset('ab'), frozenset('ac'), frozenset('bc')}


def test_other_signer_confusions_ignore_the_rows_own_signer() -> None:
    df = pd.DataFrame({
        'gloss': ['a', 'a', 'a'],
        'signer': [1, 1, 2],
        'top_n': [['x', 'a'], ['y'], ['z', 'a']],
    })
    assert other_signer_confusions(df) == [{'z'}, {'z'}, {'x', 'y'}]


class TestPermutationTest:
    def frame(self, signers: list[int], predicted: list[str], sources: list[str]) -> pd.DataFrame:
        return pd.DataFrame({'signer': signers, 'predicted': predicted, 'source': sources})

    def test_every_prediction_in_its_signers_set_beats_the_shuffle(self) -> None:
        sets: GlossSets = {i: {f'g{i}'} for i in range(20)}
        df = self.frame(list(range(20)), [f'g{i}' for i in range(20)], ['s'] * 20)
        result = permutation_test(df, 'signer', sets, n_perm=200)
        assert result.observed == 1.0
        assert result.null_mean < 0.2 and result.p < 0.01
        assert result.expected.sum() == pytest.approx(result.null_mean * 20)

    def test_strata_shuffle_only_within_a_group(self) -> None:
        # Within each source every signer performs the source's predicted gloss, so a within-source
        # shuffle can't move the rate off the observed one.
        sets: GlossSets = {1: {'a'}, 2: {'a'}, 3: {'b'}, 4: {'b'}}
        df = self.frame([1, 2, 3, 4], ['a', 'a', 'b', 'b'], ['x', 'x', 'y', 'y'])
        assert permutation_test(df, 'signer', sets, n_perm=50, strata='source').null_mean == 1.0
        assert permutation_test(df, 'signer', sets, n_perm=500).null_mean < 1.0


def test_conditional_logit_recovers_a_known_coefficient() -> None:
    rng = np.random.default_rng(0)
    n, c, beta = 3000, 8, np.array([1.5, -0.5])
    features = rng.normal(size=(n, c, 2))
    available = np.ones((n, c), dtype=bool)
    available[:, 0] = False
    u = np.where(available, features @ beta, -np.inf)
    prob = np.exp(u - u.max(axis=1, keepdims=True))
    prob /= prob.sum(axis=1, keepdims=True)
    chosen = np.array([rng.choice(c, p=p) for p in prob], dtype=np.int64)
    fit = conditional_logit(features, chosen, available, ['x', 'y'])
    assert np.all(np.abs(fit.coef - beta) < 3 * fit.se)


def test_candidate_features_exclude_the_true_gloss_and_its_own_prediction_count() -> None:
    classes = ['a', 'b', 'c']
    wrong = pd.DataFrame({'signer': [1, 2], 'source': ['s', 't'], 'gloss': ['a', 'a'], 'predicted': ['b', 'b']})
    features, chosen, available = candidate_features(
        wrong, [set(), {'c'}], classes, {1: {'b'}}, {'t': {'c'}}, {'a': 3, 'b': 1}, {frozenset('ac')}
    )
    assert chosen.tolist() == [1, 1]
    assert not available[:, 0].any() and available[:, 1:].all()
    assert features[0, :, 0].tolist() == [0, 1, 0]  # signer 1 performs b
    assert features[1, :, 0].tolist() == [0, 0, 0]  # signer 2 has no training set
    assert features[1, :, 1].tolist() == [0, 0, 1]
    assert features[0, 1, 2] == pytest.approx(np.log(2))  # b predicted once more, elsewhere
    assert features[1, :, 4].tolist() == [0, 0, 1]
    assert features[0, :, 5].tolist() == [0, 0, 1]
