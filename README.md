# E3-Subgraph Builder v0.2

This module builds a canonical DARPA TC E3 event store and process-centered temporal subgraph dataset.

The first supported path is E3-CADETS:

1. Parse line-delimited CDM18 JSON into `entities.parquet` and `events.parquet`.
2. Optionally build `labels.parquet` from a ThreaTrace-style UUID file.
3. Split events before sampling, then build process-centered temporal causal subgraphs.
4. Export sharded PyG `HeteroData` samples plus `metadata.parquet`.
5. Validate dataset statistics and train/val/test event/sample overlap.

Reference code downloaded for inspection:

- `<reference-code-dir>/MAGIC/utils/trace_parser.py`
- `<reference-code-dir>/ProHunter/preprocess/dataset_preprocess.py`
- `<reference-code-dir>/PPG/main.cpp`
- `<reference-code-dir>/OCR-APT/src/encode_to_PyG.py`
- `<reference-code-dir>/OCR-APT/src/detect_anomalous_subgraphs.py`

Example commands:

```powershell
cd <repo-root>

conda run -n intel_sports python .\scripts\parse.py `
  --dataset cadets `
  --raw-dir <raw-e3-dir> `
  --out-dir <processed-dir>\cadets `
  --split-mode chronological_disjoint `
  --chronological-val-fraction 0.1 `
  --chronological-test-fraction 0.2

conda run -n intel_sports python .\scripts\build_labels.py `
  --dataset cadets `
  --groundtruth <groundtruth-file> `
  --out-dir <processed-dir>\cadets `
  --label-source threatrace

conda run -n intel_sports python .\scripts\build_subgraphs.py `
  --dataset cadets `
  --store-dir <processed-dir>\cadets `
  --out-dir <processed-dir>\cadets\subgraphs `
  --index-cache-dir <processed-dir>\cadets\index_cache `
  --write-sidecar

conda run -n intel_sports python .\scripts\validate.py `
  --store-dir <processed-dir>\cadets `
  --subgraph-root <processed-dir>\cadets\subgraphs `
  --fail-on-event-leakage

conda run -n intel_sports python .\scripts\visualize_subgraphs.py `
  --subgraph-dir <processed-dir>\cadets\subgraphs\test `
  --positive 20 `
  --negative 20
```

Formal CADETS validation command shape:

```powershell
cd <repo-root>

conda run -n intel_sports python .\scripts\parse.py `
  --dataset cadets `
  --raw-dir <raw-e3-dir> `
  --out-dir <repo-root>\runs\cadets_chrono_full `
  --split-mode chronological_disjoint `
  --chronological-val-fraction 0.1 `
  --chronological-test-fraction 0.2 `
  --strict-split-files

conda run -n intel_sports python .\scripts\build_labels.py `
  --dataset cadets `
  --groundtruth <reference-code-dir>\OCR-APT\groundtruth\cadets_ground_truth.txt `
  --out-dir <repo-root>\runs\cadets_chrono_full `
  --label-source ocr_apt_groundtruth

conda run -n intel_sports python .\scripts\build_subgraphs.py `
  --dataset cadets `
  --store-dir <repo-root>\runs\cadets_chrono_full `
  --out-dir <repo-root>\runs\cadets_chrono_full\subgraphs `
  --samples-per-shard 200 `
  --max-edges-per-pair 32 `
  --max-edges-per-expansion-node 128 `
  --index-cache-dir <repo-root>\runs\cadets_chrono_full\index_cache `
  --write-sidecar

conda run -n intel_sports python .\scripts\validate.py `
  --store-dir <repo-root>\runs\cadets_chrono_full `
  --subgraph-root <repo-root>\runs\cadets_chrono_full\subgraphs `
  --fail-on-event-leakage `
  --out <repo-root>\runs\cadets_chrono_full\dataset_report.json
```

Diagnostic commands:

```powershell
conda run -n intel_sports python .\scripts\parse.py `
  --dataset cadets `
  --raw-dir <raw-e3-dir> `
  --out-dir <repo-root>\runs\cadets_magic_reproduction `
  --split-mode magic_reproduction `
  --strict-split-files

conda run -n intel_sports python .\scripts\visualize_subgraphs.py `
  --subgraph-dir <repo-root>\runs\cadets_chrono_full\subgraphs\train `
  --positive 20 `
  --negative 20

conda run -n intel_sports python .\scripts\train_unsupervised_baseline.py `
  --model hgt `
  --method both `
  --train-dir <repo-root>\runs\cadets_chrono_full\balanced_sanity\train `
  --test-dir <repo-root>\runs\cadets_chrono_full\balanced_sanity\test `
  --out-dir <repo-root>\runs\cadets_chrono_full\balanced_sanity\unsup_hgt `
  --train-labels 0 `
  --require-train-positive-node-count zero `
  --contamination 0.01 `
  --epochs 20
```

Important design choices:

- Keep `actor_uuid/object_uuid` and `flow_src_uuid/flow_dst_uuid` together.
- New parses write `event_edge_id`, a stable hash over dataset/event/object role/endpoints/type/time/sequence. Old Parquet files are still supported; sampling derives this ID on read.
- Use information-flow direction for sampling; `READ`, `RECV`, `LOAD`, and `EXECUTE` are reversed by default.
- Build entity metadata from all JSON chunks before parsing split-specific events.
- `chronological_disjoint` is the formal split mode. It creates train, validation, and test event intervals before sampling, so subgraphs are never randomly split after construction.
- `magic_reproduction` exists only to reproduce earlier code paths. TRACE `magic_reproduction` is explicitly not eligible for formal paper results because the current MAGIC-style split reuses overlapping files.
- Formal validation should use `scripts/validate.py --subgraph-root <subgraphs> --fail-on-event-leakage`; `shared_event_edge_ids` must be zero.
- The MVP sampler outputs only `PROCESS`, `FILE`, and `SOCKET` nodes. Canonical tables still retain all parsed entity types for later schema expansion.
- Do not compute LLM embeddings in preprocessing.
- Labels are stored outside graphs so ThreaTrace, ORTHRUS, REAPr, and DARPA-original labels can be compared later.
- For OCR-APT's `cadets_ground_truth.txt`, use `--label-source ocr_apt_groundtruth`; do not report it as ThreaTrace unless the labels are actually from ThreaTrace.
- Label-aware center selection flags such as `--labeled-centers-only` and `--centers-touching-labels` are diagnostic-only and require `--allow-diagnostic-label-selection`.
- Do not use supervised benign/malicious cross-entropy as the formal baseline. The formal baseline path is unsupervised anomaly detection: edge-type prediction NLL and one-class deviation over graph embeddings. `scripts/train_baseline.py` is retained only as a code-path sanity check.
- `scripts/train_unsupervised_baseline.py` never passes attack labels into the loss. If `--train-labels 0` is used, report it as a clean-normal one-class protocol, not as a supervised classifier.
- Do not use `scripts/build_balanced_dataset.py` output as a formal test set. It filters by label and `positive_node_count`, and is retained only for quick sanity experiments.
- Rebuild subgraphs with `--write-sidecar` before visualization or manual audit. The `.pt` shards intentionally avoid storing UUID/path strings; sample-level strings live in `nodes.parquet` and `edges.parquet`.
- For full CADETS samples, prefer starting with `--max-edges-per-pair 32 --max-edges-per-expansion-node 128`. Budget pruning is label-agnostic; it ranks by hop distance, center touch, temporal distance, direction, and stable edge IDs.
- Use `--index-cache-dir` for repeated full-split sampling. Cache files are keyed by the source `events.parquet` path, size, mtime, split, and row count; pass `--rebuild-index-cache` after changing event contents or index semantics.
- Historical exploratory OCR-APT IOC notes may exist under `<repo-root>\runs\...`; treat them as diagnostic artifacts, not formal paper results.
