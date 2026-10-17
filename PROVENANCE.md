# Benchmark provenance

The benchmark evidence records two GPU runs on 2026-10-17: full-chunk CRF search and sampled CRF search. Their reports retain the measured timing, file sizes, quality scores, hardware details, settings and recorded revision identifier.

- `compression/app.py` implements splitting, CRF search, encoding, concatenation and scoring.
- `image/Dockerfile` pins the media-toolchain source revisions. The accompanying patches implement CUDA VMAF integration, Rust compatibility and harmonic pooling.
- `modal_app.py` provides the current serverless interface, isolated run directories, checksums and explicit quality-failure reporting.
- `benchmarks/2026-10-17/` contains reports, logs, input metadata and archived benchmark wrappers. Identifier and path labels in those archives have been normalized to this project's names. Numeric measurements, media commands and encoding/scoring settings are preserved; these are normalized evidence copies, not byte-for-byte raw captures.
- `benchmarks/2026-10-17/provenance.json` records the public input URL, checksum and hashes of the normalized evidence files.

The archived revision identifier records the benchmark's code state; it is not a commit ID in this repository. Published results are historical measurements, not a fresh GPU execution of the current entrypoint. The current entrypoint adds run isolation and reporting without changing the default measured encoding/search profile.

This repository grants no project-wide license. Dependencies retain their own terms; see `THIRD_PARTY.md`.
