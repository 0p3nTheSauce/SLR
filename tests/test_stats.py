import pytest

from src.preprocess import Instance
from src.stats import (
    HistoGram,
    create_class_stats_table,
    create_split_summary_table,
    instance_stats,
    reverse_preproc_format,
    set_stats,
    split_stats,
)


def _instance_stats(
    num_instances: int, signers: int, variations: int
) -> instance_stats:
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


def _instance(video_id: str, label_num: int, label_name: str) -> Instance:
    return Instance(
        bbox=[0, 0, 1, 1],
        frame_end=20,
        frame_start=0,
        instance_id=0,
        signer_id=0,
        source="test",
        split="train",
        url="",
        variation_id=0,
        video_id=video_id,
        label_num=label_num,
        label_name=label_name,
    )


class TestReversePreprocFormat:
    def test_groups_by_label_num(self) -> None:
        instances = [
            _instance("1", 1, "dog"),
            _instance("2", 0, "book"),
            _instance("3", 1, "dog"),
        ]

        classes = reverse_preproc_format(instances)

        assert [c["gloss"] for c in classes] == ["book", "dog"]
        assert [[i.video_id for i in c["instances"]] for c in classes] == [
            ["2"],
            ["1", "3"],
        ]

    def test_gloss_named_empty_keeps_all_instances(self) -> None:
        """Regression test: "empty" used to be the unfilled-slot sentinel, and WLASL has a
        real gloss called "empty", so all but its last instance were dropped."""
        instances = [_instance(str(n), 0, "empty") for n in range(3)]

        (empty,) = reverse_preproc_format(instances)

        assert empty["gloss"] == "empty"
        assert len(empty["instances"]) == 3

    def test_missing_class_gets_empty_slot_named_from_classes(self) -> None:
        instances = [_instance("1", 2, "cat")]

        classes = reverse_preproc_format(instances, classes=["book", "dog", "cat"])

        assert [c["gloss"] for c in classes] == ["book", "dog", "cat"]
        assert [len(c["instances"]) for c in classes] == [0, 0, 1]
        assert classes[0]["instances"] is not classes[1]["instances"]

    def test_missing_class_without_classes_raises(self) -> None:
        with pytest.raises(ValueError, match="label_num 0"):
            reverse_preproc_format([_instance("1", 1, "dog")])

    def test_no_instances(self) -> None:
        assert reverse_preproc_format([]) == []
