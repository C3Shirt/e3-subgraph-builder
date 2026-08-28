# E3-Subgraph Builder v0.1

This module builds a canonical DARPA TC E3 event store and process-centered temporal subgraph dataset.

The first supported path is E3-CADETS:

1. Parse line-delimited CDM18 JSON into `entities.parquet` and `events.parquet`.
2. Optionally build `labels.parquet` from a ThreaTrace-style UUID file.
3. Split before sampling, then build process-centered temporal causal subgraphs.
4. Export sharded PyG `HeteroData` samples plus `metadata.parquet`.
5. Validate dataset statistics and train/test event/sample overlap.

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
  --split-mode magic

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
  --subgraph-dir <processed-dir>\cadets\subgraphs\test

conda run -n intel_sports python .\scripts\visualize_subgraphs.py `
  --subgraph-dir <processed-dir>\cadets\subgraphs\test `
  --positive 20 `
  --negative 20
```

CADETS validation commands used for the current local run:

```powershell
cd <repo-root>

conda run -n intel_sports python .\scripts\parse.py `
  --dataset cadets `
  --raw-dir <raw-e3-dir> `
  --out-dir <repo-root>\runs\cadets_magic_full `
  --split-mode magic `
  --strict-split-files

conda run -n intel_sports python .\scripts\build_labels.py `
  --dataset cadets `
  --groundtruth <reference-code-dir>\OCR-APT\groundtruth\cadets_ground_truth.txt `
  --out-dir <repo-root>\runs\cadets_magic_full `
  --label-source ocr_apt_groundtruth

conda run -n intel_sports python .\scripts\build_subgraphs.py `
  --dataset cadets `
  --store-dir <repo-root>\runs\cadets_magic_full `
  --out-dir <repo-root>\runs\cadets_magic_full\subgraphs `
  --max-samples 1000 `
  --samples-per-shard 200 `
  --max-edges-per-pair 32 `
  --max-edges-per-expansion-node 128 `
  --index-cache-dir <repo-root>\runs\cadets_magic_full\index_cache `
  --write-sidecar

conda run -n intel_sports python .\scripts\validate.py `
  --store-dir <repo-root>\runs\cadets_magic_full `
  --subgraph-dir <repo-root>\runs\cadets_magic_full\subgraphs\train `
  --out <repo-root>\runs\cadets_magic_full\dataset_report_train_sample.json

conda run -n intel_sports python .\scripts\visualize_subgraphs.py `
  --subgraph-dir <repo-root>\runs\cadets_magic_full\subgraphs\train `
  --positive 20 `
  --negative 20

conda run -n intel_sports python .\scripts\train_unsupervised_baseline.py `
  --model hgt `
  --method both `
  --train-dir <repo-root>\runs\cadets_magic_full\ocr_apt_ioc_balanced_v01\train `
  --test-dir <repo-root>\runs\cadets_magic_full\ocr_apt_ioc_balanced_v01\test `
  --out-dir <repo-root>\runs\cadets_magic_full\ocr_apt_ioc_balanced_v01\unsup_hgt `
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
- Do not compute LLM embeddings in preprocessing.
- Labels are stored outside graphs so ThreaTrace, ORTHRUS, REAPr, and DARPA-original labels can be compared later.
- For OCR-APT's `cadets_ground_truth.txt`, use `--label-source ocr_apt_groundtruth`; do not report it as ThreaTrace unless the labels are actually from ThreaTrace.
- Do not use supervised benign/malicious cross-entropy as the formal baseline. The formal baseline path is unsupervised anomaly detection: edge-type prediction NLL and one-class deviation over graph embeddings. `scripts/train_baseline.py` is retained only as a code-path sanity check.
- `scripts/train_unsupervised_baseline.py` never passes attack labels into the loss. If `--train-labels 0` is used, report it as a clean-normal one-class protocol, not as a supervised classifier.
- Rebuild subgraphs with `--write-sidecar` before visualization or manual audit. The `.pt` shards intentionally avoid storing UUID/path strings; sample-level strings live in `nodes.parquet` and `edges.parquet`.
- For full CADETS samples, prefer starting with `--max-edges-per-pair 32 --max-edges-per-expansion-node 128`. The sampler also prioritizes edges touching labeled nodes before applying these caps, so IOC-derived positives are not silently removed by budget pruning.
- Use `--index-cache-dir` for repeated full-split sampling. Cache files are keyed by the source `events.parquet` path, size, mtime, split, and row count; pass `--rebuild-index-cache` after changing event contents or index semantics.
- Current exploratory OCR-APT IOC run is recorded at `<repo-root>\runs\cadets_magic_full\ocr_apt_ioc_v01\RUN_NOTES.md`.
- Current fixed balanced OCR-APT IOC baseline is recorded at `<repo-root>\runs\cadets_magic_full\ocr_apt_ioc_balanced_v01\RUN_NOTES.md`.
