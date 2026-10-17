# Measured benchmark: 2026-10-17

Source URL: https://tile.loc.gov/storage-services/service/mbrs/ntscrm/00021068/00021068.mp4

Input SHA-256: `ea51543b2c5a36391ea6b4dd6347655f3f9ecd8820ed3bc047092a7823ff1d6d`

Input: 2,083,898,324 bytes; 3,279.416667 seconds; 1920×1080 progressive H.264 at 24 fps, with stereo AAC audio.

Hardware: one NVIDIA RTX PRO 6000 Blackwell Server Edition, reported memory 97,887 MiB, driver 610.57.04, with 16 requested Modal CPU cores. Four chunk jobs shared that GPU. Chunk targets were 600 seconds; actual cuts used existing source keyframes.

`report.json` and `run-vmaf89-sampled.log` are the completed sampled-search run. `baseline-full-search/` contains the prior full-chunk-search report, log and wrapper. The input probe is preserved separately. No input/output video is committed.

The baseline took 1,679.280159 seconds of processing plus 176.918820 seconds of final verification. The sampled run took 642.333517 seconds of processing plus 196.527007 seconds of final verification. Final full-file NEG scores were 89.0634 and 89.52667 respectively, both passing a target of 89.

Settings changed between runs: full-chunk search enabled → disabled; thorough search enabled → disabled; CRF increment 0.25 → 1; sample-every 6m (not used during full search) → 2m. Encoder configuration, chunk parallelism, GPU allocation, target quality and final scoring convention stayed the same.

The worker uses a patched unshifted harmonic mean of positive frame scores. Search sampling does not weaken the separate final full-video measurement. The sampled output was 7.75% larger, at higher measured quality, so this is a common-threshold comparison, not exactly matched quality/bitrate.

The logs' summed chunk encode duration is not elapsed wall time. Actual billed costs, repeat-run variance, output frame-count validation and an AWS MediaConvert comparison were not captured by these reports. The input frame count is not proof of an identical output frame count. The new standalone wrapper records an output probe on future runs.

`source/benchmark_modal_neg.py` records the sampled benchmark wrapper with project identifiers and paths normalized. Archived wrappers and logs preserve the measured configuration and numeric results, but are not byte-identical raw captures. Use the root `modal_app.py` and `scripts/reproduce.sh` to reproduce the workload. Historical measurements are not presented as a fresh GPU run of the current entrypoint.
