from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Iterable, Optional

from e3prep.schema.entities import EntityRecord
from e3prep.schema.events import EventRecord


class CdmParser(ABC):
    dataset: str

    @abstractmethod
    def collect_entities(
        self,
        paths: Iterable[Path],
        progress_every: Optional[int] = None,
        infer_from_events: bool = True,
    ) -> dict[str, EntityRecord]:
        raise NotImplementedError

    @abstractmethod
    def parse_events(
        self,
        paths: Iterable[Path],
        entities: dict[str, EntityRecord],
        split: str,
        progress_every: Optional[int] = None,
    ) -> Iterable[EventRecord]:
        raise NotImplementedError
