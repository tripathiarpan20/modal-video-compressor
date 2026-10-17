import asyncio
import json
import os
import subprocess
import sys
import time
from pathlib import Path

import modal

ROOT = Path(__file__).resolve().parents[3]
BUILD = ROOT / "image"
SERVICE = ROOT / "compression"

app = modal.App("modal-video-compressor-benchmark")
# Resolve the saved name locally, then explicitly pass it into the container.
VOLUME_NAME = os.environ.get("COMPRESSION_VOLUME", "").strip()
if modal.is_local() and not VOLUME_NAME:
    VOLUME_NAME = (ROOT / "benchmark-results/volume-name.txt").read_text().strip()
if not VOLUME_NAME:
    raise RuntimeError("COMPRESSION_VOLUME must be configured for the benchmark")
volume = modal.Volume.from_name(VOLUME_NAME)

image = (
    modal.Image.from_dockerfile(
        str(BUILD / "Dockerfile"),
        context_dir=str(BUILD),
    )
    .pip_install_from_requirements(str(SERVICE / "requirements.txt"))
    .env({
        "COMPRESSION_VOLUME": VOLUME_NAME,
        "NVIDIA_DRIVER_CAPABILITIES": "graphics,video,compute,utility",
        "AB_AV1_BIN": "/app/ab-av1",
        "AB_AV1_EXECUTION_MODE": "local",
        "FFMPEG_BIN": "ffmpeg",
        "FFPROBE_BIN": "ffprobe",
        "SHARED_VOLUME_PATH": "/bench",
        "DISABLE_REMOTE_IO": "true",
        "COMPRESSION_STORAGE_STARTUP_CHECK_ENABLED": "false",
        "COMPRESSION_CLEANUP_ENABLED": "false",
        "MAX_CONCURRENT_COMPRESSION": "1",
        "COMPRESSION_CHUNKING_ENABLED": "true",
        "COMPRESSION_CHUNK_TARGET_SECONDS": "600",
        "COMPRESSION_CHUNK_PARALLELISM": "4",
        "AB_AV1_VMAF_MODEL": "version=vmaf_v0.6.1neg",
        "AB_AV1_VMAF_POOL": "harmonic_mean",
        "AB_AV1_VMAF_CUDA": "true",
        "AB_AV1_VMAF_N_SUBSAMPLE": "1",
        "AB_AV1_VMAF_FPS": "0",
        "AB_AV1_VMAF_SCALE": "none",
        # Search short samples, then encode each chunk once at the selected CRF.
        # Keep the full merged-file NEG measurement below as the quality gate.
        "AB_AV1_NVENC_FULL_SEARCH": "false",
        "AB_AV1_NVENC_THOROUGH": "false",
        "AB_AV1_NVENC_CRF_INCREMENT": "1",
        "AB_AV1_SAMPLE_EVERY": "2m",
        "AB_AV1_NVENC_VERIFY_TIMEOUT_SECONDS": "3600",
        # Keep the requested quality target fixed during the benchmark.
        "AB_AV1_VMAF_FALLBACK_ENABLED": "false",
    })
    .add_local_file(str(SERVICE / "app.py"), "/app/app.py")
)

@app.function(
    image=image,
    gpu="RTX-PRO-6000",
    cpu=16,
    volumes={"/bench": volume},
    timeout=7200,
    max_containers=1,
    scaledown_window=2,
    retries=0,
)
def benchmark(target: float, revision: str):
    sys.path.insert(0, "/app")
    import app as compression

    volume.reload()
    source = Path("/bench/input.mp4")

    probe = json.loads(subprocess.check_output([
        "ffprobe", "-v", "error",
        "-show_format", "-show_streams", "-of", "json", str(source),
    ], text=True))
    duration = float(probe["format"]["duration"])
    gpu = subprocess.check_output([
        "nvidia-smi",
        "--query-gpu=name,driver_version,memory.total",
        "--format=csv,noheader",
    ], text=True).strip()

    async def run():
        request = compression.CompressRequest(
            video_paths=[str(source)],
            task_id="loc-neg-benchmark",
            codec="AV1",
            codec_mode="CRF",
            target_vmaf=target,
            chunked=True,
            chunk_duration_seconds=600,
            chunk_parallelism=4,
        )

        started = time.monotonic()
        result = await compression.compress(request)
        processing_seconds = time.monotonic() - started

        if not result.success:
            raise RuntimeError(result.model_dump_json())

        output = Path(result.output_paths[0])

        # Chunked compression does not automatically verify the merged file.
        score, verification_seconds, error = await compression._measure_full_vmaf(
            "loc-neg-final", str(source), str(output)
        )

        report = {
            "revision": revision,
            "gpu": gpu,
            "codec": "AV1",
            "mode": "CRF",
            "model": compression.AB_AV1_VMAF_MODEL,
            "pool": compression.AB_AV1_VMAF_POOL,
            "target_vmaf_neg": target,
            "chunk_target_seconds": 600,
            "chunk_parallelism": 4,
            "input_duration_seconds": duration,
            "input_bytes": source.stat().st_size,
            "output_bytes": output.stat().st_size,
            "output_to_input_size_ratio": (
                output.stat().st_size / source.stat().st_size
            ),
            "compression_factor": source.stat().st_size / output.stat().st_size,
            "size_reduction_percent": 100 * (
                1 - output.stat().st_size / source.stat().st_size
            ),
            "processing_wall_seconds": processing_seconds,
            "processing_realtime_multiple": duration / processing_seconds,
            "final_verification_seconds": verification_seconds,
            "processing_plus_verification_seconds": (
                processing_seconds + verification_seconds
            ),
            "measured_full_video_vmaf_neg": score,
            "quality_pass": score is not None and score >= target,
            "verification_error": error,
            "output_volume_path": "/" + output.name,
            "input_probe": probe,
            "compression_config": compression._compression_config_status(),
        }
        Path("/bench/report.json").write_text(json.dumps(report, indent=2))
        return report

    report = asyncio.run(run())
    volume.commit()
    return report

@app.local_entrypoint()
def main(target: float = 89.0):
    revision = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()
    started = time.monotonic()
    report = benchmark.remote(target, revision)
    report["remote_call_wall_seconds"] = time.monotonic() - started
    Path("benchmark-results/report.json").write_text(
        json.dumps(report, indent=2)
    )
    print(json.dumps(report, indent=2))
