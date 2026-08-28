from __future__ import annotations

from pathlib import Path


MAGIC_REPRODUCTION_SPLITS = {
    "cadets": {
        "train": [
            "ta1-cadets-e3-official.json",
            "ta1-cadets-e3-official.json.1",
            "ta1-cadets-e3-official.json.2",
            "ta1-cadets-e3-official-2.json.1",
        ],
        "test": [
            "ta1-cadets-e3-official-2.json",
        ],
    },
    "theia": {
        "train": [
            "ta1-theia-e3-official-6r.json",
            "ta1-theia-e3-official-6r.json.1",
            "ta1-theia-e3-official-6r.json.2",
            "ta1-theia-e3-official-6r.json.3",
        ],
        "test": [
            "ta1-theia-e3-official-6r.json.8",
        ],
    },
    "trace": {
        "train": [
            "ta1-trace-e3-official-1.json",
            "ta1-trace-e3-official-1.json.1",
            "ta1-trace-e3-official-1.json.2",
            "ta1-trace-e3-official-1.json.3",
        ],
        "test": [
            "ta1-trace-e3-official-1.json",
            "ta1-trace-e3-official-1.json.1",
            "ta1-trace-e3-official-1.json.2",
            "ta1-trace-e3-official-1.json.3",
            "ta1-trace-e3-official-1.json.4",
        ],
    },
}

SPLIT_MODES = ("magic_reproduction", "chronological_disjoint", "all")

DATASET_JSON_PREFIXES = {
    "cadets": ("ta1-cadets-e3-official",),
    "theia": ("ta1-theia-e3-official",),
    "trace": ("ta1-trace-e3-official",),
}


def discover_json_chunks(raw_dir: Path, dataset: str | None = None) -> list[Path]:
    ignored = ("names", "types", "metadata", "malicious")
    compressed_suffixes = (".tar.gz", ".tgz", ".gz", ".xz", ".zip")
    prefixes = DATASET_JSON_PREFIXES.get(dataset or "", ())
    return sorted(
        path
        for path in raw_dir.iterdir()
        if path.is_file()
        and "json" in path.name
        and (not prefixes or path.name.startswith(prefixes))
        and not path.name.endswith(compressed_suffixes)
        and not path.name.endswith(".txt")
        and not any(token in path.name for token in ignored)
    )


def normalize_split_mode(split_mode: str) -> str:
    if split_mode == "magic":
        return "magic_reproduction"
    return split_mode


def split_mode_warnings(dataset: str, split_mode: str) -> list[str]:
    split_mode = normalize_split_mode(split_mode)
    warnings = []
    if split_mode == "magic_reproduction":
        warnings.append("magic_reproduction is for reproducing prior code paths, not for formal paper results.")
    if dataset == "trace" and split_mode == "magic_reproduction":
        warnings.append("TRACE magic_reproduction reuses overlapping files and is not eligible for formal results.")
    if split_mode == "all":
        warnings.append("all is diagnostic-only and does not provide train/val/test separation.")
    return warnings


def resolve_split_paths(
    dataset: str,
    raw_dir: Path,
    split_mode: str = "magic_reproduction",
    strict: bool = False,
) -> dict[str, list[Path]]:
    split_mode = normalize_split_mode(split_mode)
    if split_mode == "all":
        return {"all": discover_json_chunks(raw_dir, dataset)}
    if split_mode == "chronological_disjoint":
        return {"all": discover_json_chunks(raw_dir, dataset)}
    if split_mode != "magic_reproduction":
        raise ValueError(f"Unsupported split mode: {split_mode}")
    if dataset not in MAGIC_REPRODUCTION_SPLITS:
        raise ValueError(f"No MAGIC-compatible split is known for dataset: {dataset}")

    result: dict[str, list[Path]] = {}
    missing: list[Path] = []
    for split, names in MAGIC_REPRODUCTION_SPLITS[dataset].items():
        result[split] = []
        for name in names:
            path = raw_dir / name
            if path.exists():
                result[split].append(path)
            else:
                missing.append(path)
    if strict and missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing split files:\n{missing_text}")
    return result
