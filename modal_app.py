"""Modal RTX PRO 6000 compression with sampled CRF search and final NEG scoring."""
from __future__ import annotations

import asyncio
import hashlib
import importlib.util
import json
import os
import re
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

import modal

ROOT = Path(__file__).resolve().parent
VOLUME_NAME = os.environ.get("COMPRESSION_VOLUME", "modal-video-compressor")
app = modal.App("modal-video-compressor")
volume = modal.Volume.from_name(VOLUME_NAME, create_if_missing=True)

# The settings used by the published sampled-search benchmark.
WORKER_ENV = {
    "COMPRESSION_VOLUME": VOLUME_NAME,
    "NVIDIA_DRIVER_CAPABILITIES": "graphics,video,compute,utility",
    "AB_AV1_BIN": "/app/ab-av1",
    "AB_AV1_EXECUTION_MODE": "local",
    "FFMPEG_BIN": "ffmpeg",
    "FFPROBE_BIN": "ffprobe",
    "SHARED_VOLUME_PATH": "/data/runs",
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
    "AB_AV1_NVENC_FULL_SEARCH": "false",
    "AB_AV1_NVENC_THOROUGH": "false",
    "AB_AV1_NVENC_CRF_INCREMENT": "1",
    "AB_AV1_SAMPLE_EVERY": "2m",
    "AB_AV1_NVENC_VERIFY_TIMEOUT_SECONDS": "3600",
    "AB_AV1_VMAF_FALLBACK_ENABLED": "false",
}

image = (
    modal.Image.from_dockerfile(str(ROOT / "image/Dockerfile"), context_dir=str(ROOT / "image"))
    .pip_install_from_requirements(str(ROOT / "compression/requirements.txt"))
    .env(WORKER_ENV)
    .add_local_file(str(ROOT / "compression/app.py"), "/app/compression_worker.py")
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def validate_request(input_path: str, run_id: str, target: float, chunk_seconds: int, parallelism: int):
    path = PurePosixPath(input_path)
    if not path.is_absolute() or ".." in path.parts or path.parts[1:2] != ("inputs",) or len(path.parts) < 3:
        raise ValueError("input-path must be a file below /inputs/ on the Modal Volume")
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", run_id):
        raise ValueError("run-id must contain 1-80 letters, digits, underscores or hyphens")
    if not 0 < target <= 100 or chunk_seconds < 1 or not 1 <= parallelism <= 16:
        raise ValueError("target must be (0, 100], chunk-seconds positive, parallelism 1-16")


def probe(path: Path) -> dict:
    return json.loads(subprocess.check_output([
        "ffprobe", "-v", "error", "-show_format", "-show_streams", "-of", "json", str(path),
    ], text=True))


@app.function(image=image, gpu="RTX-PRO-6000", cpu=16, volumes={"/data": volume},
              timeout=7200, max_containers=1, scaledown_window=2, retries=0)
@modal.concurrent(max_inputs=1)
def compress(input_path: str, target: float, codec: str, chunk_seconds: int,
             parallelism: int, run_id: str, provenance: dict, expected_sha256: str = "") -> dict:
    validate_request(input_path, run_id, target, chunk_seconds, parallelism)
    volume.reload()
    source = (Path("/data") / input_path.lstrip("/")).resolve()
    if not source.is_relative_to(Path("/data/inputs")) or not source.is_file():
        raise ValueError("Input file is missing or outside /data/inputs")
    work = Path("/data/runs") / run_id
    work.mkdir(parents=True, exist_ok=False)
    input_hash = sha256(source)
    if expected_sha256 and input_hash != expected_sha256:
        raise ValueError("Input SHA-256 does not match the requested source")

    # A fresh module gives every invocation its own asyncio locks and configuration.
    spec = importlib.util.spec_from_file_location("compression_worker", "/app/compression_worker.py")
    worker = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = worker
    spec.loader.exec_module(worker)
    worker.SHARED_VOLUME_PATH = str(work)
    input_probe = probe(source)
    duration = float(input_probe["format"]["duration"])
    gpu = subprocess.check_output([
        "nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader",
    ], text=True).strip()

    async def run():
        req = worker.CompressRequest(video_paths=[str(source)], task_id=run_id,
                                     codec=codec, codec_mode="CRF", target_vmaf=target,
                                     chunked=True, chunk_duration_seconds=chunk_seconds,
                                     chunk_parallelism=parallelism)
        started = time.monotonic()
        response = await worker.compress(req)
        processing_seconds = time.monotonic() - started
        if not response.success:
            raise RuntimeError(response.model_dump_json())
        output = Path(response.output_paths[0])
        score, verification_seconds, error = await worker._measure_full_vmaf(
            f"{run_id}-final", str(source), str(output)
        )
        input_bytes, output_bytes = source.stat().st_size, output.stat().st_size
        return {
            **provenance,
            "run_id": run_id,
            "recorded_at_utc": datetime.now(timezone.utc).isoformat(),
            "gpu": gpu,
            "requested_cpu_cores": 16,
            "modal_image_id": os.getenv("MODAL_IMAGE_ID"),
            "codec": codec, "mode": "CRF",
            "model": worker.AB_AV1_VMAF_MODEL, "pool": worker.AB_AV1_VMAF_POOL,
            "target_vmaf_neg": target,
            "chunk_target_seconds": chunk_seconds, "chunk_parallelism": parallelism,
            "input_duration_seconds": duration, "input_sha256": input_hash,
            "input_bytes": input_bytes, "output_bytes": output_bytes,
            "output_to_input_size_ratio": output_bytes / input_bytes,
            "compression_factor": input_bytes / output_bytes,
            "size_reduction_percent": 100 * (1 - output_bytes / input_bytes),
            "processing_wall_seconds": processing_seconds,
            "processing_realtime_multiple": duration / processing_seconds,
            "final_verification_seconds": verification_seconds,
            "processing_plus_verification_seconds": processing_seconds + verification_seconds,
            "measured_full_video_vmaf_neg": score,
            "quality_pass": score is not None and score >= target and error is None,
            "verification_error": error,
            "output_volume_path": "/" + str(output.relative_to("/data")),
            "report_volume_path": f"/runs/{run_id}/report.json",
            "input_probe": input_probe, "output_probe": probe(output),
            "compression_config": worker._compression_config_status(),
        }

    try:
        report = asyncio.run(run())
        (work / "report.json").write_text(json.dumps(report, indent=2) + "\n")
        return report
    except Exception as exc:
        (work / "failure.json").write_text(json.dumps({"run_id": run_id, "error": str(exc)}, indent=2))
        raise
    finally:
        volume.commit()


@app.local_entrypoint()
def main(input_path: str = "/inputs/input.mp4", target: float = 89.0, codec: str = "AV1",
         chunk_seconds: int = 600, parallelism: int = 4, run_id: str = "",
         output_dir: str = "results", expected_sha256: str = "", source_url: str = ""):
    run_id = run_id or datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ-") + uuid.uuid4().hex[:8]
    codec = codec.upper()
    if codec not in {"AV1", "HEVC"}:
        raise ValueError("codec must be AV1 or HEVC")
    validate_request(input_path, run_id, target, chunk_seconds, parallelism)
    local_dir = Path(output_dir) / run_id
    local_dir.mkdir(parents=True, exist_ok=True)
    if (local_dir / "report.json").exists():
        raise ValueError("Run already has a report; use a new run-id")
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True).strip())
    source_files = [ROOT / "modal_app.py", ROOT / "requirements.txt", ROOT / "compression/app.py",
                    ROOT / "compression/requirements.txt", *sorted((ROOT / "image").glob("*"))]
    provenance = {"revision": revision, "working_tree_dirty": dirty,
                  "source_url": source_url, "volume_name": VOLUME_NAME,
                  "source_file_sha256": {str(p.relative_to(ROOT)): sha256(p) for p in source_files if p.is_file()}}
    started = time.monotonic()
    report = compress.remote(input_path, target, codec, chunk_seconds, parallelism,
                             run_id, provenance, expected_sha256)
    report["remote_call_wall_seconds"] = time.monotonic() - started
    (local_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
    if not report["quality_pass"]:
        raise RuntimeError(f"Final VMAF NEG quality check failed; report saved to {local_dir / 'report.json'}")
