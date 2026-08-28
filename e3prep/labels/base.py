from __future__ import annotations

from pathlib import Path
from typing import Iterable

from e3prep.schema.labels import LabelRecord


def writeable_label_rows(labels: Iterable[LabelRecord]) -> Iterable[dict]:
    for label in labels:
        yield label.to_dict()


def clean_label_token(line: str) -> str | None:
    line = line.strip()
    if not line or line.startswith("#"):
        return None
    return line.split()[0].strip()


def read_uuid_lines(path: Path) -> list[str]:
    uuids: list[str] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            uuid = clean_label_token(line)
            if uuid:
                uuids.append(uuid)
    return uuids

