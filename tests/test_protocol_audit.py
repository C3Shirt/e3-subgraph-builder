from __future__ import annotations

from scripts.audit_protocol import count_episodes_from_timestamps, protocol_readiness


def test_count_episodes_from_timestamps_splits_on_gap_and_duration() -> None:
    timestamps = [
        0,
        10 * 1_000_000_000,
        200 * 1_000_000_000,
        810 * 1_000_000_000,
    ]

    assert count_episodes_from_timestamps(timestamps, max_duration_sec=600, inactivity_gap_sec=120) == 3


def test_protocol_a_not_ready_when_train_has_no_positive_subgraphs() -> None:
    report = protocol_readiness(
        {
            "label_counts_by_split": {
                "train": {0: 100},
                "val": {0: 10, 1: 1},
                "test": {0: 10, 1: 1},
            }
        },
        {},
    )

    assert report["protocol_a_supervised_ready"] is False
    assert "train has zero positive process-center/subgraph labels under the current protocol." in report["warnings"]
