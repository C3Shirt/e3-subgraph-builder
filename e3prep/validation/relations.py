from __future__ import annotations

from pathlib import Path

import pandas as pd

from e3prep.graph.direction import relation_name
from e3prep.io import read_parquet


def _top_group_counts(df: pd.DataFrame, columns: list[str], limit: int) -> list[dict]:
    if df.empty:
        return []
    grouped = (
        df.assign(**{column: df[column].fillna("UNKNOWN").astype(str) for column in columns})
        .groupby(columns, dropna=False)
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
        .head(limit)
    )
    return [
        {**{column: str(row[column]) for column in columns}, "count": int(row["count"])}
        for _, row in grouped.iterrows()
    ]


def relation_audit_report(store_dir: Path, top_n: int = 50) -> dict:
    events_path = store_dir / "events.parquet"
    events = (
        read_parquet(
            events_path,
            columns=[
                "actor_uuid",
                "actor_type",
                "object_uuid",
                "object_type",
                "event_type",
                "flow_src_uuid",
                "flow_src_type",
                "flow_dst_uuid",
                "flow_dst_type",
            ],
        )
        if events_path.exists()
        else pd.DataFrame()
    )
    if events.empty:
        return {
            "events_total": 0,
            "actor_object_top": [],
            "flow_relation_top": [],
            "reverse_flow_by_event_type": {},
            "unknown_endpoint_by_event_type": {},
        }

    events = events.copy()
    events["relation"] = events["event_type"].apply(relation_name)
    reversed_flow = (events["actor_uuid"] != events["flow_src_uuid"]) | (events["object_uuid"] != events["flow_dst_uuid"])
    unknown_endpoint = (events["actor_type"] == "UNKNOWN") | (events["object_type"] == "UNKNOWN")

    return {
        "events_total": int(len(events)),
        "actor_object_top": _top_group_counts(events, ["actor_type", "event_type", "object_type"], top_n),
        "flow_relation_top": _top_group_counts(events, ["flow_src_type", "relation", "flow_dst_type"], top_n),
        "reverse_flow_total": int(reversed_flow.sum()),
        "reverse_flow_by_event_type": (
            events.loc[reversed_flow, "event_type"].value_counts().sort_index().astype(int).to_dict()
        ),
        "unknown_endpoint_total": int(unknown_endpoint.sum()),
        "unknown_endpoint_by_event_type": (
            events.loc[unknown_endpoint, "event_type"].value_counts().sort_index().astype(int).to_dict()
        ),
    }
