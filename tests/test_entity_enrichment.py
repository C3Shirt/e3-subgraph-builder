from __future__ import annotations

import pandas as pd

from scripts.enrich_entities_from_events import enrich_entities


def test_enrich_entities_fills_missing_file_and_process_paths() -> None:
    entities = pd.DataFrame(
        [
            {"uuid": "F1", "node_type": "FILE", "name": None, "path": None},
            {"uuid": "P1", "node_type": "PROCESS", "name": None, "path": None},
            {"uuid": "F2", "node_type": "FILE", "name": "/keep", "path": "/keep"},
        ]
    )

    enriched, summary = enrich_entities(
        entities,
        file_paths={"F1": "/tmp/a", "F2": "/tmp/ignored"},
        process_exec_paths={"P1": "/bin/bash"},
    )

    by_uuid = {row.uuid: row for row in enriched.itertuples(index=False)}
    assert by_uuid["F1"].path == "/tmp/a"
    assert by_uuid["F1"].name == "/tmp/a"
    assert by_uuid["P1"].path == "/bin/bash"
    assert by_uuid["P1"].name == "/bin/bash"
    assert by_uuid["F2"].path == "/keep"
    assert summary["updated_by_type"] == {"FILE": 1, "PROCESS": 1}
