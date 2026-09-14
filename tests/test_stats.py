from src.stats import (
    HistoGram,
    create_class_stats_table,
    create_split_summary_table,
    instance_stats,
    set_stats,
    split_stats,
)


def _instance_stats(num_instances: int, signers: int, variations: int) -> instance_stats:
    return instance_stats(
        num_instances=num_instances,
        length_distribution=HistoGram({}),
        signers_distribution=HistoGram(dict.fromkeys(range(signers), 1)),
        source_distribution=HistoGram({}),
        url_distribution=HistoGram({}),
        variation_distribution=HistoGram(dict.fromkeys(range(variations), 1)),
    )


class TestCreateSplitSummaryTable:
    def test_one_row_per_set(self) -> None:
        split = split_stats(
            num_classes=2,
            num_instances=10,
            num_signers=5,
            per_set_stats={
                "train": set_stats(
                    num_instances=6,
                    num_signers=4,
                    per_instance_stats={
                        "book": _instance_stats(6, 4, 1),
                        "dog": _instance_stats(0, 0, 0),
                    },
                ),
                "test": set_stats(
                    num_instances=4,
                    num_signers=3,
                    per_instance_stats={"book": _instance_stats(4, 3, 1)},
                ),
            },
        )

        table = create_split_summary_table(split)

        assert list(table["Set"]) == ["train", "test"]
        assert list(table["Instances"]) == [6, 4]
        assert list(table["Signers"]) == [4, 3]
        assert list(table["Classes"]) == [2, 1]


class TestCreateClassStatsTable:
    def test_one_row_per_gloss(self) -> None:
        subset = set_stats(
            num_instances=10,
            num_signers=5,
            per_instance_stats={
                "book": _instance_stats(6, 4, 2),
                "dog": _instance_stats(4, 3, 1),
            },
        )

        table = create_class_stats_table(subset)

        assert list(table["Gloss"]) == ["book", "dog"]
        assert list(table["Instances"]) == [6, 4]
        assert list(table["Signers"]) == [4, 3]
        assert list(table["Variations"]) == [2, 1]
