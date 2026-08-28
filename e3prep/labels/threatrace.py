from __future__ import annotations

from pathlib import Path
from typing import Iterable

from e3prep.labels.base import read_uuid_lines
from e3prep.schema.labels import LabelRecord


def read_threatrace_labels(
    path: Path,
    dataset: str,
    label_source: str = "threatrace",
) -> Iterable[LabelRecord]:
    for uuid in read_uuid_lines(path):
        yield LabelRecord(
            uuid=uuid,
            dataset=dataset,
            label=1,
            label_source=label_source,
            confidence="entity_level",
        )

