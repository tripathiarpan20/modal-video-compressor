# Modal Video Compressor

## 54 minutes of video. 88% smaller. Compressed in under 11 minutes.

**2.08 GB → 240 MB. One RTX PRO 6000 Blackwell. Full-video VMAF NEG: 89.53.**

Serverless AV1 compression that puts four parallel chunk jobs on one GPU—and joins the results without another encode. Run it from your laptop; Modal handles the GPU.

| The result | Measured on the public benchmark video |
|---|---:|
| **Shrink the file** | **8.69× compression · 88.49% smaller** |
| **Finish the compression** | **10m 42s · 5.11× realtime** |
| **Include the full-video quality check** | **13m 59s total** |
| **Hit the quality target** | **VMAF NEG 89.53 · target 89 passed** |

Measured on one 54m 39s, 1080p24 video on **October 17, 2026**. Compression time includes splitting, CRF search, encoding and merging; the total adds final verification. Both exclude data transfers and image build/startup. File sizes above use decimal units. [Read the report](benchmarks/2026-10-17/report.json).

**[Reproduce it →](#from-scratch-reproduce-the-public-video-benchmark)** · **[See the evidence →](#latest-measured-benchmark)** · **[Bring your own video →](#compress-another-video)**

### What makes it fast?

- **One GPU, four concurrent jobs.** RTX PRO 6000 Blackwell combines four NVENC encoders, four NVDEC decoders and CUDA compute for encoding and quality scoring.
- **Cut at keyframes. Skip the second encode.** Split near existing source keyframes, compress chunks in parallel, then concatenate with stream copy. No full-video re-encode just to join the pieces.
- **Search samples. Verify the whole video.** Sampled CRF search was **2.61× faster** than our full-chunk search for processing this video; a separate full-video VMAF NEG check verifies the final result.
- **Run it yourself.** Public source video, checksummed input, saved reports, logs and a reproduction script are included. No local GPU required.

These are measured results for this input, not a universal speed or quality guarantee. AWS speed and cost comparisons are still pending.

## Latest measured benchmark

A 54m 39.42s, 1920×1080, 24 fps H.264 input was encoded to AV1 on 2026-10-17:

| Metric | Full-chunk CRF search | Sampled CRF search (default) |
|---|---:|---:|
| Processing wall time | 1,679.28 s | **642.33 s** |
| Final quality verification | 176.92 s | 196.53 s |
| Processing + verification | 1,856.20 s | **838.86 s** |
| Processing speed versus realtime | 1.95× | **5.11×** |
| Original file bytes | 2,083,898,324 | 2,083,898,324 |
| Output file bytes | 222,615,964 | **239,871,925** |
| Compression factor (original / output) | 9.36:1 | **8.69:1** |
| File-size reduction | 89.32% | **88.49%** |
| Measured full-video VMAF NEG | 89.0634 | **89.52667** |
| Target met | Yes | **Yes** |

The sampled run was **2.61× faster for processing** and **2.21× faster including verification**, with an output 7.75% larger than the original search. These are one measured run per configuration, not a statistical throughput guarantee. Both passed the same threshold; their actual quality scores differ.

This does **not** establish 10× faster or cheaper than AWS MediaConvert. No AWS comparison or billed-cost measurement is included. Modal credits do not make consumed compute free.

Evidence: [full report](benchmarks/2026-10-17/report.json), [logs](benchmarks/2026-10-17/run-vmaf89-sampled.log), [baseline](benchmarks/2026-10-17/baseline-full-search/report.json), [provenance](benchmarks/2026-10-17/provenance.json), and [benchmark details](benchmarks/2026-10-17/README.md).

## Features and design

### RTX PRO 6000 Blackwell for AV1 encoding and CUDA scoring

The worker selects the **RTX PRO 6000 Blackwell Server Edition** for its combination of dedicated video engines and CUDA processing capacity. The full GPU has **four NVENC encoders and four NVDEC decoders**. FFmpeg's `av1_nvenc` uses the dedicated encoders for AV1 compression; CUDA compute supports the `libvmaf_cuda` quality analysis alongside that media workload. NVIDIA documents the engine counts in its [RTX PRO 6000 hardware profiles](https://docs.nvidia.com/datacenter/tesla/mig-user-guide/supported-mig-profiles.html#rtx-pro-6000-blackwell-mig-profiles) and describes the ninth-generation NVENC / sixth-generation NVDEC hardware on the [Server Edition product page](https://www.nvidia.com/en-us/data-center/rtx-pro-6000-blackwell-server-edition/).

This combination suits multiple concurrent encode-and-score jobs. The 96 GB of memory provides headroom, but the published workload does not establish a need for that capacity or prove this GPU is optimal across all prices and workloads.

### Keyframe-aligned chunks and concatenation without another encode

The splitter uses `-c copy -f segment` to cut near the requested duration at **existing source keyframes**. Cutting compressed video arbitrarily inside a group of pictures (GOP) can separate predicted frames from the reference frames they need, causing missing/corrupted frames or visible discontinuities around joins. Respecting random-access boundaries avoids that class of unnecessary dependency breakage.

Each chunk is encoded independently with a consistent output configuration. The encoded chunks are then joined in order using FFmpeg's concat demuxer and **`-c copy`**. This avoids a second full-video encode after concatenation, saving time and avoiding another generation of encoding loss.

Keyframe alignment is not a universal guarantee of invisible seams: open GOPs may contain cross-boundary dependencies, and timestamps, codec parameters and audio continuity still matter. Source cuts are keyframe aligned, not guaranteed scene boundaries. See FFmpeg's [segment](https://ffmpeg.org/ffmpeg-formats.html#segment_002c-stream_005fsegment_002c-ssegment) and [concat](https://ffmpeg.org/ffmpeg-formats.html#concat) documentation. The final quality check measures the complete result; seam-specific inspection remains useful.

### Four parallel chunk jobs on one GPU

An asynchronous queue with a semaphore runs **up to four chunk searches/encodes concurrently** inside one Modal GPU container. As one finishes, the next waiting chunk starts; concatenation waits for all chunks and preserves source order. This overlaps encode, decode and scoring work and exposes multiple encoding sessions to the GPU. NVIDIA's driver manages NVENC engine scheduling; the code does not pin one chunk to one engine. [NVIDIA's NVENC application note](https://docs.nvidia.com/video-technologies/video-codec-sdk/13.0/nvenc-application-note/index.html#nvenc-performance) explains multi-session scheduling.

`--parallelism` controls the concurrency limit. The default benchmark uses four jobs, 600-second target chunks, and one GPU. Higher concurrency is not assumed to improve performance: jobs share encoder capacity, CUDA compute, CPU and storage bandwidth.

### Sampled search with a full-video quality check

CRF search evaluates short samples with `--sample-every 2m` and CRF steps of 1. This avoids repeatedly encoding whole chunks during the search. After the final chunk encodes and merge, the complete video is scored with **VMAF NEG**, harmonic pooling and CUDA acceleration. The CLI reports a quality failure if the measured score misses the target.

## From scratch: reproduce the public-video benchmark

You need Python 3.11+, Git, curl, and a Modal account with GPU access and billing configured. Modal documents its GPU requirements [here](https://modal.com/docs/guide/gpu). The first run compiles a substantial FFmpeg/libvmaf/ab-av1 image remotely. It can take time and consume build resources; subsequent unchanged builds can use Modal's cache.

On Ubuntu/Debian, install the local tools first (use a distribution providing Python 3.11 or newer):

```bash
sudo apt-get update
sudo apt-get install -y git curl python3 python3-venv
python3 --version
```

On macOS with Homebrew:

```bash
brew install python@3.11 git curl
```

Clone the public repository and set up the local CLI:

```bash
git clone https://github.com/tripathiarpan20/modal-video-compressor.git
cd modal-video-compressor

# On macOS, use python3.11 here if python3 points at an older interpreter.
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt

# Authenticate once; skip setup if the intended profile is already active.
modal setup
modal profile list

bash scripts/reproduce.sh
```

The script:

1. Downloads the [public Library of Congress MP4](https://tile.loc.gov/storage-services/service/mbrs/ntscrm/00021068/00021068.mp4) to `data/input.mp4` and checks its SHA-256 against the measured input.
2. Creates a separate Modal Volume and uploads the input before GPU processing starts.
3. Runs the sampled-search profile at target NEG 89 with 600-second chunk targets, four concurrent chunk jobs, one RTX PRO 6000, and 16 requested CPU cores.
4. Measures processing wall time and the merged output's full-video NEG score separately.
5. Saves a JSON report and logs in `results/<run-id>/` and downloads the compressed MP4 when quality passes.

Allow space locally for the approximately 2.08 GB input plus output. The source bytes are not stored in Git. Downloading the same URL later is not proof of identical input; the script refuses a checksum mismatch.

Inspect the latest result:

```bash
RUN_ID="$(cat results/latest-run-id.txt)"
python scripts/summarize.py "results/$RUN_ID/report.json" \
  --baseline benchmarks/2026-10-17/report.json
```

The final quality check must pass. If the score is below 89 or verification fails, the CLI exits nonzero **after saving the report**, leaving output on the Volume for diagnosis. It does not silently lower the target or automatically repair the merged output. Failed compression writes `failure.json` to the run directory when possible.

## Compress another video

The standalone worker supports AV1 and HEVC; only AV1 has the published benchmark above. Upload a local file to a path below `/inputs/`, then invoke the CLI. Use a new volume name or reuse one you own, and use a new run ID for each invocation.

```bash
export COMPRESSION_VOLUME="my-compression-$(date -u +%Y%m%d-%H%M%S)"
modal volume create "$COMPRESSION_VOLUME"
modal volume put "$COMPRESSION_VOLUME" /absolute/path/to/video.mp4 /inputs/input.mp4

RUN_ID="custom-$(date -u +%Y%m%d-%H%M%S)"
modal run modal_app.py \
  --input-path /inputs/input.mp4 \
  --target 89 --codec AV1 \
  --chunk-seconds 600 --parallelism 4 \
  --run-id "$RUN_ID"

modal volume get "$COMPRESSION_VOLUME" \
  "/runs/$RUN_ID/input_compressed.mp4" "results/$RUN_ID/output.mp4"
python scripts/summarize.py "results/$RUN_ID/report.json"
```

Paths passed to `--input-path` are **Volume paths**, not host paths or URLs. The source filename determines the output filename; `input.mp4` produces `input_compressed.mp4`. The precise output location is also in `output_volume_path`. Reports and outputs have per-run directories; reusing a run ID is rejected. `COMPRESSION_VOLUME` is explicitly propagated into the remote image to make container imports independent of the local shell.

To retain the same uploaded input for repeated benchmark runs, set `COMPRESSION_VOLUME` before calling `scripts/reproduce.sh`. The script reuses that volume and allocates a new run directory. Do not concurrently replace the same input file while a worker is reading it.

## Algorithm and quality settings

- FFmpeg splits with `-c copy -f segment` on **existing keyframes** near the requested chunk duration. Source GOPs determine the actual boundaries; these are not guaranteed scene cuts or exactly 600-second segments.
- ab-av1 searches each chunk with `--sample-every 2m`, CRF increment 1, and thorough/full-chunk search disabled. The pinned upstream default sample length is used. `sample-every` controls the sample count/distribution; it is not the duration of an encoded sample.
- NVENC uses preset `p4`, UHQ tuning and full-resolution multipass for AV1, plus the recorded lookahead, B-frame, AQ and tiling settings. These encoder settings match the published run.
- Search and final verification use `version=vmaf_v0.6.1neg`, harmonic pooling, CUDA VMAF, all frames, and no scoring downscale.
- A small libvmaf patch implements an **unshifted harmonic mean over positive frame scores**. This pooling convention must also be used for an apples-to-apples quality comparison; do not substitute ordinary VMAF or a different aggregation.
- Encoded chunks are concatenated in order with `-c copy`. Full-file NEG verification checks the completed output. A global score does not establish that every scene or seam exceeds 89.
- A source that cannot be split into multiple keyframe chunks can fall back to the single-pass path. Check logs for `Chunking unavailable` and record the actual path when comparing results.

The CLI uses local paths on a Modal Volume and disables the worker's optional HTTP/S3 remote I/O and storage startup checks. Configuration uses neutral `COMPRESSION_*` environment variables. The benchmark profile is set explicitly in `WORKER_ENV` in `modal_app.py`; importing the underlying service by itself uses its service defaults.

## Interpreting timings and cost

`processing_wall_seconds` covers splitting, CRF search, final chunk encoding and concatenation. It excludes upload/download, cold start, input hashing/probing, and the separate final verification pass. `processing_plus_verification_seconds` includes the final NEG check. `remote_call_wall_seconds` additionally captures scheduling/startup and other work during the remote invocation, but excludes the initial image build and local data staging.

The worker log's `encode` field adds concurrent chunk durations. Its `effective_speed` therefore is **not** overall wall-clock throughput. Use the report's `processing_realtime_multiple` instead.

The report records GPU/driver, requested CPU cores, image ID when available, input/output probes, input checksum, effective configuration, source hashes, revision and dirty-tree state. Historical reports contain fewer fields; see their provenance. Compression ratios use whole-file bytes including audio and container overhead.

Record Modal billing separately: GPU, CPU, memory, storage, transfer and build charges where applicable. The two-hour GPU-function timeout and one-container limit are not monetary spending caps. For MediaConvert comparisons, keep source bytes, output dimensions/frame rate, codec, audio treatment, and measured quality comparable; report processing and queue/transfer times separately.

## Cleanup

Volumes persist after the application exits. After checking your local downloads:

```bash
RUN_ID="$(cat results/latest-run-id.txt)"
COMPRESSION_VOLUME="$(cat "results/$RUN_ID/volume-name.txt")"
modal volume delete "$COMPRESSION_VOLUME"
```

Deleting a reused volume removes **all runs in that volume**, so download anything you need first. Local media and generated results are ignored by Git.

## Development and validation

```bash
python -m pip install -r requirements-dev.txt
python -m pytest -q
bash -n scripts/reproduce.sh
```

Tests exercise command generation, rate-control/quality behavior, profile selection and input validation without a GPU. They do not validate NVENC performance, CUDA VMAF, or reproduce the recorded measurements. The published GPU runs used the archived benchmark wrapper; rerun the current entrypoint on Modal to validate its packaging and reporting behavior.

Upstream toolchain revisions are pinned in [image/Dockerfile](image/Dockerfile). Base-image tags and package-manager dependencies are not fully digest/lock pinned, so this is a source/configuration reproduction, not a promise of bit-identical image builds or timings.

See [PROVENANCE.md](PROVENANCE.md) and [THIRD_PARTY.md](THIRD_PARTY.md) for source and dependency details.
