"""
Compression microservice — wraps ab-av1 and ffmpeg installed in this container.

Accepts a video path (on shared volume) OR a URL, codec, and quality settings,
runs VMAF-targeted compression, returns the output path or S3 URL.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import math
import os
import re
import shutil
import time
import uuid
from contextlib import asynccontextmanager, suppress
from datetime import datetime, timedelta, timezone
from typing import Optional

import boto3
import httpx
from botocore.config import Config
from botocore.exceptions import ClientError
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, field_validator, model_validator

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger("compression")


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.lower() not in {"0", "false", "no", "off"}


@asynccontextmanager
async def lifespan(app: FastAPI):
    await asyncio.to_thread(_startup_storage_check)
    cleanup_task = asyncio.create_task(_cleanup_worker())
    try:
        yield
    finally:
        cleanup_task.cancel()
        with suppress(asyncio.CancelledError):
            await cleanup_task


app = FastAPI(title="Video Compression Service", lifespan=lifespan)

SHARED_VOLUME_PATH = os.getenv("SHARED_VOLUME_PATH", "/tmp/video-compressor")
MAX_CONCURRENT = int(os.getenv("MAX_CONCURRENT_COMPRESSION", "2"))
MAX_QUEUE_SIZE = int(os.getenv("MAX_QUEUE_SIZE_COMPRESSION") or os.getenv("MAX_QUEUE_SIZE", "5"))
DISABLE_REMOTE_IO = os.getenv("DISABLE_REMOTE_IO", "false").lower() in ("1", "true", "yes")

# Storage provider label only. Uploads use one S3-compatible code path.
STORAGE_PROVIDER = os.getenv("COMPRESSION_STORAGE_PROVIDER", "s3").lower()
S3_REGION = os.getenv("COMPRESSION_STORAGE_S3_REGION", "us-east-1").strip() or "us-east-1"
S3_BUCKET = os.getenv("COMPRESSION_STORAGE_S3_BUCKET_NAME", "").strip()
S3_ACCESS_KEY_ID = os.getenv("COMPRESSION_STORAGE_S3_ACCESS_KEY_ID", "").strip()
S3_SECRET_ACCESS_KEY = os.getenv("COMPRESSION_STORAGE_S3_SECRET_ACCESS_KEY", "").strip()
S3_ENDPOINT_URL = (
    os.getenv("COMPRESSION_STORAGE_S3_ENDPOINT_URL")
    or os.getenv("COMPRESSION_STORAGE_S3_ENDPOINT")
    or ""
).strip()
MINIO_ALLOWED_DOWNLOAD_CIDRS = [
    cidr.strip()
    for cidr in os.getenv("COMPRESSION_MINIO_ALLOWED_DOWNLOAD_CIDRS", "").split(",")
    if cidr.strip()
]
STORAGE_STARTUP_CHECK_ENABLED = os.getenv("COMPRESSION_STORAGE_STARTUP_CHECK_ENABLED", "false").lower() in (
    "1",
    "true",
    "yes",
)
S3_PRESIGNED_EXPIRY = int(
    os.getenv("COMPRESSION_STORAGE_S3_PRESIGNED_EXPIRY")
    or os.getenv("S3_PRESIGNED_EXPIRY")
    or "3600"
)
PRESIGNED_URL_CLEANUP_GRACE_SECONDS = int(os.getenv("PRESIGNED_URL_CLEANUP_GRACE_SECONDS", "600"))
TEMP_FILE_TTL_SECONDS = int(
    os.getenv("COMPRESSION_TEMP_FILE_TTL_SECONDS")
    or os.getenv("TEMP_FILE_TTL_SECONDS")
    or str(min(S3_PRESIGNED_EXPIRY, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS)
)
CLEANUP_INTERVAL_SECONDS = int(
    os.getenv("COMPRESSION_CLEANUP_INTERVAL_SECONDS") or os.getenv("CLEANUP_INTERVAL_SECONDS") or "300"
)
CLEANUP_MAX_VOLUME_BYTES = int(
    os.getenv("COMPRESSION_CLEANUP_MAX_VOLUME_BYTES") or os.getenv("CLEANUP_MAX_VOLUME_BYTES") or "9000000000"
)
CLEANUP_MIN_FILE_AGE_SECONDS = int(
    os.getenv("COMPRESSION_CLEANUP_MIN_FILE_AGE_SECONDS") or os.getenv("CLEANUP_MIN_FILE_AGE_SECONDS") or "60"
)
CLEANUP_ENABLED = os.getenv("COMPRESSION_CLEANUP_ENABLED", os.getenv("CLEANUP_ENABLED", "true")).lower() in (
    "1",
    "true",
    "yes",
)
STORAGE_CLEANUP_ENABLED = os.getenv(
    "COMPRESSION_STORAGE_CLEANUP_ENABLED", os.getenv("STORAGE_CLEANUP_ENABLED", "true")
).lower() in ("1", "true", "yes")
STORAGE_CLEANUP_PREFIXES = [
    prefix.strip()
    for prefix in os.getenv("COMPRESSION_STORAGE_CLEANUP_PREFIXES", "processing/,upscaling/").split(",")
    if prefix.strip()
]
STORAGE_OBJECT_TTL_SECONDS = int(
    os.getenv("COMPRESSION_STORAGE_OBJECT_TTL_SECONDS")
    or os.getenv("STORAGE_OBJECT_TTL_SECONDS")
    or str(min(S3_PRESIGNED_EXPIRY, 604800) + PRESIGNED_URL_CLEANUP_GRACE_SECONDS)
)
COMPRESSION_CHUNKING_ENABLED = os.getenv("COMPRESSION_CHUNKING_ENABLED", "true").lower() in (
    "1",
    "true",
    "yes",
)
COMPRESSION_CHUNK_MIN_DURATION_SECONDS = int(os.getenv("COMPRESSION_CHUNK_MIN_DURATION_SECONDS", "1200"))
COMPRESSION_CHUNK_TARGET_SECONDS = int(os.getenv("COMPRESSION_CHUNK_TARGET_SECONDS", "600"))
COMPRESSION_CHUNK_PARALLELISM = max(1, int(os.getenv("COMPRESSION_CHUNK_PARALLELISM", "2")))
FFPROBE_BIN = os.getenv("FFPROBE_BIN", "ffprobe")
FFMPEG_BIN = os.getenv("FFMPEG_BIN", "ffmpeg")
AB_AV1_BIN = os.getenv("AB_AV1_BIN", "ab-av1").strip() or "ab-av1"
AB_AV1_DEFAULT_TARGET_VMAF = float(os.getenv("AB_AV1_DEFAULT_TARGET_VMAF", "93.0"))
AB_AV1_SAMPLE_EVERY = os.getenv("AB_AV1_SAMPLE_EVERY", "6m")
AB_AV1_SHORT_SAMPLE_EVERY = os.getenv("AB_AV1_SHORT_SAMPLE_EVERY", "5s")
AB_AV1_SHORT_SAMPLE_MAX_DURATION_SECONDS = float(
    os.getenv("AB_AV1_SHORT_SAMPLE_MAX_DURATION_SECONDS", "120")
)
AB_AV1_VMAF_MODEL = os.getenv("AB_AV1_VMAF_MODEL", "version=vmaf_v0.6.1neg").strip()
AB_AV1_VMAF_POOL = os.getenv("AB_AV1_VMAF_POOL", "harmonic_mean").strip()
AB_AV1_VMAF_N_SUBSAMPLE = os.getenv("AB_AV1_VMAF_N_SUBSAMPLE", "1").strip()
AB_AV1_VMAF_FPS = os.getenv("AB_AV1_VMAF_FPS", "0").strip()
AB_AV1_VMAF_SCALE = os.getenv("AB_AV1_VMAF_SCALE", "none").strip()
AB_AV1_VMAF_CUDA = os.getenv("AB_AV1_VMAF_CUDA", "true").lower() not in {
    "0", "false", "no", "off"
}
AB_AV1_VMAF_N_THREADS = (
    "0" if AB_AV1_VMAF_CUDA else os.getenv("AB_AV1_VMAF_N_THREADS", "").strip()
)
AB_AV1_MAX_ENCODED_PERCENT = os.getenv("AB_AV1_MAX_ENCODED_PERCENT", "")
AB_AV1_VMAF_FALLBACK_ENABLED = os.getenv("AB_AV1_VMAF_FALLBACK_ENABLED", "true").lower() not in {
    "0", "false", "no", "off"
}
AB_AV1_VMAF_FALLBACK_FLOOR_DELTA = 5.0
AB_AV1_VMAF_FALLBACK_STEP = max(0.1, float(os.getenv("AB_AV1_VMAF_FALLBACK_STEP", "2.0")))
AB_AV1_VMAF_FALLBACK_MAX_ATTEMPTS = max(1, int(os.getenv("AB_AV1_VMAF_FALLBACK_MAX_ATTEMPTS", "5")))
AB_AV1_HWACCEL = os.getenv("AB_AV1_HWACCEL", "cuda")
AB_AV1_ENCODER_AV1 = os.getenv("AB_AV1_ENCODER_AV1", "av1_nvenc")
AB_AV1_ENCODER_H264 = os.getenv("AB_AV1_ENCODER_H264", "h264_nvenc")
AB_AV1_ENCODER_HEVC = os.getenv("AB_AV1_ENCODER_HEVC", "hevc_nvenc")
AB_AV1_ENCODER_VP9 = os.getenv("AB_AV1_ENCODER_VP9", "libvpx-vp9")
AB_AV1_NVENC_MIN_CRF = os.getenv("AB_AV1_NVENC_MIN_CRF", "1").strip()
AB_AV1_NVENC_MAX_CRF = os.getenv("AB_AV1_NVENC_MAX_CRF", "51").strip()
AB_AV1_NVENC_CRF_INCREMENT = max(
    0.01,
    float(os.getenv("AB_AV1_NVENC_CRF_INCREMENT", "0.25")),
)
AB_AV1_NVENC_TARGETED_MIN_CRF = _env_bool("AB_AV1_NVENC_TARGETED_MIN_CRF", True)
AB_AV1_NVENC_MIN_CRF_VMAF_85 = float(os.getenv("AB_AV1_NVENC_MIN_CRF_VMAF_85", "28"))
AB_AV1_NVENC_MIN_CRF_VMAF_89 = float(os.getenv("AB_AV1_NVENC_MIN_CRF_VMAF_89", "22"))
AB_AV1_NVENC_MIN_CRF_VMAF_93 = float(os.getenv("AB_AV1_NVENC_MIN_CRF_VMAF_93", "16"))
AB_AV1_NVENC_THOROUGH = _env_bool("AB_AV1_NVENC_THOROUGH", True)
AB_AV1_NVENC_FULL_SEARCH = _env_bool("AB_AV1_NVENC_FULL_SEARCH", True)
AB_AV1_NVENC_DIRECT_CQ_ENABLED = _env_bool("AB_AV1_NVENC_DIRECT_CQ_ENABLED", True)
AB_AV1_NVENC_INITIAL_CRF_VMAF_85 = float(
    os.getenv("AB_AV1_NVENC_INITIAL_CRF_VMAF_85", "47.75")
)
AB_AV1_NVENC_INITIAL_CRF_VMAF_89 = float(
    os.getenv("AB_AV1_NVENC_INITIAL_CRF_VMAF_89", "42.5")
)
AB_AV1_NVENC_INITIAL_CRF_VMAF_93 = float(
    os.getenv("AB_AV1_NVENC_INITIAL_CRF_VMAF_93", "36.75")
)
AB_AV1_HEVC_NVENC_DIRECT_CQ_ENABLED = _env_bool("AB_AV1_HEVC_NVENC_DIRECT_CQ_ENABLED", True)
AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_85 = float(
    os.getenv("AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_85", "46.25")
)
AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_89 = float(
    os.getenv("AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_89", "35.5")
)
AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_93 = float(
    os.getenv("AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_93", "30")
)
AB_AV1_NVENC_FULL_VMAF_VERIFY = _env_bool("AB_AV1_NVENC_FULL_VMAF_VERIFY", True)
AB_AV1_NVENC_VERIFY_RETRY = _env_bool("AB_AV1_NVENC_VERIFY_RETRY", True)
AB_AV1_NVENC_VMAF_PER_CRF = max(
    0.01,
    float(os.getenv("AB_AV1_NVENC_VMAF_PER_CRF", "1.5")),
)
AB_AV1_HEVC_NVENC_VMAF_PER_CRF = max(
    0.01,
    float(os.getenv("AB_AV1_HEVC_NVENC_VMAF_PER_CRF", "0.18")),
)
AB_AV1_NVENC_RETRY_MIN_OVERSHOOT = max(
    AB_AV1_NVENC_CRF_INCREMENT,
    float(os.getenv("AB_AV1_NVENC_RETRY_MIN_OVERSHOOT", "0.25")),
)
AB_AV1_NVENC_MAX_CRF_ADJUSTMENT = max(
    AB_AV1_NVENC_CRF_INCREMENT,
    float(os.getenv("AB_AV1_NVENC_MAX_CRF_ADJUSTMENT", "4.0")),
)
AB_AV1_NVENC_RATE_CAP_SATURATION_RATIO = min(
    1.0,
    max(0.0, float(os.getenv("AB_AV1_NVENC_RATE_CAP_SATURATION_RATIO", "0.95"))),
)
AB_AV1_NVENC_VERIFY_TIMEOUT_SECONDS = max(
    1.0,
    float(os.getenv("AB_AV1_NVENC_VERIFY_TIMEOUT_SECONDS", "60")),
)
AB_AV1_NVENC_BFRAME_ARGS = tuple(
    arg.strip()
    for arg in os.getenv(
        "AB_AV1_NVENC_BFRAME_ARGS",
        "bf=5,b_ref_mode=middle",
    ).split(",")
    if arg.strip()
)
AB_AV1_NVENC_TEMPORAL_AQ = _env_bool("AB_AV1_NVENC_TEMPORAL_AQ", True)
AB_AV1_NVENC_TEMPORAL_FILTER_LEVEL = int(
    os.getenv("AB_AV1_NVENC_TEMPORAL_FILTER_LEVEL", "0")
)
if AB_AV1_NVENC_TEMPORAL_FILTER_LEVEL not in {0, 4}:
    raise ValueError("AB_AV1_NVENC_TEMPORAL_FILTER_LEVEL must be 0 or 4")
AB_AV1_AV1_NVENC_ENCODER_ARGS = tuple(
    arg.strip()
    for arg in os.getenv(
        "AB_AV1_AV1_NVENC_ENCODER_ARGS",
        (
            "rc=vbr,b:v=0,tune=uhq,multipass=fullres,"
            "rc-lookahead=24,lookahead_level=3,g=900,"
            "tile-columns=1,tile-rows=1,qp_cb_offset=2,qp_cr_offset=2"
        ),
    ).split(",")
    if arg.strip()
)
AB_AV1_HEVC_NVENC_PRESET = os.getenv("AB_AV1_HEVC_NVENC_PRESET", "p4").strip() or "p4"
AB_AV1_HEVC_NVENC_MIN_CRF = (
    os.getenv("AB_AV1_HEVC_NVENC_MIN_CRF") or AB_AV1_NVENC_MIN_CRF
).strip()
AB_AV1_HEVC_NVENC_MAX_CRF = (
    os.getenv("AB_AV1_HEVC_NVENC_MAX_CRF") or AB_AV1_NVENC_MAX_CRF
).strip()
AB_AV1_HEVC_NVENC_THOROUGH = _env_bool("AB_AV1_HEVC_NVENC_THOROUGH", AB_AV1_NVENC_THOROUGH)
AB_AV1_HEVC_NVENC_FULL_SEARCH = _env_bool("AB_AV1_HEVC_NVENC_FULL_SEARCH", AB_AV1_NVENC_FULL_SEARCH)
AB_AV1_HEVC_NVENC_ENCODER_ARGS = tuple(
    arg.strip()
    for arg in os.getenv(
        "AB_AV1_HEVC_NVENC_ENCODER_ARGS",
        (
            "rc=vbr,b:v=0,tune=hq,multipass=qres,"
            "rc-lookahead=24,lookahead_level=3,g=900,"
            "qp_cb_offset=2,qp_cr_offset=2"
        ),
    ).split(",")
    if arg.strip()
)
AB_AV1_VBR_VALIDATION_TOLERANCE = float(
    os.getenv("AB_AV1_VBR_VALIDATION_TOLERANCE", "0.10")
)
AB_AV1_VBR_MAX_RATE_RATIO = float(
    os.getenv(
        "AB_AV1_VBR_MAX_RATE_RATIO",
        str(1.0 + AB_AV1_VBR_VALIDATION_TOLERANCE),
    )
)
AB_AV1_LIBSVTAV1_PRESET = os.getenv("AB_AV1_LIBSVTAV1_PRESET", "8")
AB_AV1_X264_PRESET = os.getenv("AB_AV1_X264_PRESET", "medium")
AB_AV1_X265_PRESET = os.getenv("AB_AV1_X265_PRESET", "medium")
COMPRESSION_PROGRESS_INTERVAL_SECONDS = float(os.getenv("COMPRESSION_PROGRESS_INTERVAL_SECONDS", "5"))
COMPRESSION_LOG_AB_AV1_OUTPUT = os.getenv("COMPRESSION_LOG_AB_AV1_OUTPUT", "true").lower() not in {
    "0", "false", "no", "off"
}
PROCESS_OUTPUT_CAPTURE_CHARS = 100000

_semaphore = asyncio.Semaphore(MAX_CONCURRENT)
_queue_size = 0
_active_count = 0
_active_file_paths: set[str] = set()
_lock = asyncio.Lock()

# Codec name -> ab-av1/ffmpeg encoder mapping.
CODEC_MAP = {
    "AV1": AB_AV1_ENCODER_AV1,
    "H264": AB_AV1_ENCODER_H264,
    "H.264": AB_AV1_ENCODER_H264,
    "HEVC": AB_AV1_ENCODER_HEVC,
    "H265": AB_AV1_ENCODER_HEVC,
    "H.265": AB_AV1_ENCODER_HEVC,
    "VP9": AB_AV1_ENCODER_VP9,
}


def _is_url(path: str) -> bool:
    return path.startswith("http://") or path.startswith("https://")


def _get_s3_client():
    client_kwargs = {
        "region_name": S3_REGION,
        "aws_access_key_id": S3_ACCESS_KEY_ID or None,
        "aws_secret_access_key": S3_SECRET_ACCESS_KEY or None,
        "config": Config(signature_version="s3v4"),
    }
    if S3_ENDPOINT_URL:
        client_kwargs["endpoint_url"] = S3_ENDPOINT_URL
    return boto3.client("s3", **client_kwargs)


def _storage_config_status() -> dict[str, object]:
    return {
        "provider": STORAGE_PROVIDER,
        "region": S3_REGION,
        "bucket_configured": bool(S3_BUCKET),
        "access_key_configured": bool(S3_ACCESS_KEY_ID),
        "secret_key_configured": bool(S3_SECRET_ACCESS_KEY),
        "endpoint_configured": bool(S3_ENDPOINT_URL),
    }


def _compression_config_status() -> dict[str, object]:
    return {
        "backend": "ab-av1",
        "ab_av1_bin": AB_AV1_BIN,
        "encoders": CODEC_MAP,
        "default_target_vmaf": AB_AV1_DEFAULT_TARGET_VMAF,
        "vmaf_fallback": {
            "enabled": AB_AV1_VMAF_FALLBACK_ENABLED,
            "floor_delta": AB_AV1_VMAF_FALLBACK_FLOOR_DELTA,
            "step": AB_AV1_VMAF_FALLBACK_STEP,
            "max_attempts": AB_AV1_VMAF_FALLBACK_MAX_ATTEMPTS,
        },
        "sample_every": AB_AV1_SAMPLE_EVERY,
        "short_sample_every": AB_AV1_SHORT_SAMPLE_EVERY,
        "short_sample_max_duration_seconds": AB_AV1_SHORT_SAMPLE_MAX_DURATION_SECONDS,
        "vmaf_model": AB_AV1_VMAF_MODEL,
        "vmaf_pool": AB_AV1_VMAF_POOL,
        "vmaf_n_subsample": AB_AV1_VMAF_N_SUBSAMPLE,
        "vmaf_n_threads": AB_AV1_VMAF_N_THREADS,
        "vmaf_fps": AB_AV1_VMAF_FPS,
        "vmaf_scale": AB_AV1_VMAF_SCALE,
        "vmaf_cuda": AB_AV1_VMAF_CUDA,
        "nvenc": {
            "min_crf": AB_AV1_NVENC_MIN_CRF,
            "max_crf": AB_AV1_NVENC_MAX_CRF,
            "crf_increment": AB_AV1_NVENC_CRF_INCREMENT,
            "targeted_min_crf": AB_AV1_NVENC_TARGETED_MIN_CRF,
            "target_min_crfs": {
                "85": AB_AV1_NVENC_MIN_CRF_VMAF_85,
                "89": AB_AV1_NVENC_MIN_CRF_VMAF_89,
                "93": AB_AV1_NVENC_MIN_CRF_VMAF_93,
            },
            "bframe_args": list(AB_AV1_NVENC_BFRAME_ARGS),
            "temporal_aq": AB_AV1_NVENC_TEMPORAL_AQ,
            "temporal_filter_level": AB_AV1_NVENC_TEMPORAL_FILTER_LEVEL,
            "thorough": AB_AV1_NVENC_THOROUGH,
            "full_search": AB_AV1_NVENC_FULL_SEARCH,
            "full_vmaf_verify": AB_AV1_NVENC_FULL_VMAF_VERIFY,
            "verify_retry": AB_AV1_NVENC_VERIFY_RETRY,
            "direct_cq": {
                "enabled_for_av1": AB_AV1_NVENC_DIRECT_CQ_ENABLED,
                "enabled_for_hevc": AB_AV1_HEVC_NVENC_DIRECT_CQ_ENABLED,
                "effective_for_av1": (
                    AB_AV1_NVENC_DIRECT_CQ_ENABLED and AB_AV1_NVENC_FULL_VMAF_VERIFY
                ),
                "effective_for_hevc": (
                    AB_AV1_HEVC_NVENC_DIRECT_CQ_ENABLED and AB_AV1_NVENC_FULL_VMAF_VERIFY
                ),
                "av1_initial_crfs": {
                    "85": AB_AV1_NVENC_INITIAL_CRF_VMAF_85,
                    "89": AB_AV1_NVENC_INITIAL_CRF_VMAF_89,
                    "93": AB_AV1_NVENC_INITIAL_CRF_VMAF_93,
                },
                "hevc_initial_crfs": {
                    "85": AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_85,
                    "89": AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_89,
                    "93": AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_93,
                },
                "vmaf_per_crf": AB_AV1_NVENC_VMAF_PER_CRF,
                "hevc_vmaf_per_crf": AB_AV1_HEVC_NVENC_VMAF_PER_CRF,
                "retry_min_overshoot": AB_AV1_NVENC_RETRY_MIN_OVERSHOOT,
                "max_crf_adjustment": AB_AV1_NVENC_MAX_CRF_ADJUSTMENT,
                "rate_cap_saturation_ratio": AB_AV1_NVENC_RATE_CAP_SATURATION_RATIO,
            },
            "verify_timeout_seconds": AB_AV1_NVENC_VERIFY_TIMEOUT_SECONDS,
        },
        "av1_nvenc": {
            "encoder_args": list(AB_AV1_AV1_NVENC_ENCODER_ARGS),
        },
        "hevc_nvenc": {
            "default_preset": AB_AV1_HEVC_NVENC_PRESET,
            "min_crf": AB_AV1_HEVC_NVENC_MIN_CRF,
            "max_crf": AB_AV1_HEVC_NVENC_MAX_CRF,
            "thorough": AB_AV1_HEVC_NVENC_THOROUGH,
            "full_search": AB_AV1_HEVC_NVENC_FULL_SEARCH,
            "encoder_args": list(AB_AV1_HEVC_NVENC_ENCODER_ARGS),
        },
        "vbr": {
            "supported_encoders": "*_nvenc",
            "max_rate_ratio": AB_AV1_VBR_MAX_RATE_RATIO,
            "validation_tolerance": AB_AV1_VBR_VALIDATION_TOLERANCE,
        },
        "chunking_enabled": COMPRESSION_CHUNKING_ENABLED,
        "chunk_min_duration_seconds": COMPRESSION_CHUNK_MIN_DURATION_SECONDS,
        "chunk_target_seconds": COMPRESSION_CHUNK_TARGET_SECONDS,
        "chunk_parallelism": COMPRESSION_CHUNK_PARALLELISM,
    }


def _validate_s3_config():
    missing = []
    if not S3_BUCKET:
        missing.append("COMPRESSION_STORAGE_S3_BUCKET_NAME")
    if not S3_ACCESS_KEY_ID:
        missing.append("COMPRESSION_STORAGE_S3_ACCESS_KEY_ID")
    if not S3_SECRET_ACCESS_KEY:
        missing.append("COMPRESSION_STORAGE_S3_SECRET_ACCESS_KEY")

    if missing:
        raise RuntimeError(f"Missing storage configuration: {', '.join(missing)}")


def _s3_error_code(exc: ClientError) -> str:
    return str(exc.response.get("Error", {}).get("Code", ""))


def _s3_status_code(exc: ClientError) -> int | None:
    status = exc.response.get("ResponseMetadata", {}).get("HTTPStatusCode")
    return int(status) if isinstance(status, int) else None


def _ensure_s3_bucket(client):
    try:
        client.head_bucket(Bucket=S3_BUCKET)
        return
    except ClientError as exc:
        code = _s3_error_code(exc)
        status = _s3_status_code(exc)
        if status != 404 and code not in {"404", "NoSuchBucket", "NotFound"}:
            raise

    kwargs = {"Bucket": S3_BUCKET}
    if S3_REGION and S3_REGION != "us-east-1":
        kwargs["CreateBucketConfiguration"] = {"LocationConstraint": S3_REGION}

    try:
        client.create_bucket(**kwargs)
        log.info(f"Created {STORAGE_PROVIDER} bucket: {S3_BUCKET}")
    except ClientError as exc:
        code = _s3_error_code(exc)
        if code != "BucketAlreadyOwnedByYou" and not (S3_ENDPOINT_URL and code == "BucketAlreadyExists"):
            raise


def _ensure_minio_download_policy(client):
    if STORAGE_PROVIDER != "minio" or not MINIO_ALLOWED_DOWNLOAD_CIDRS:
        return

    policy = {
        "Version": "2012-10-17",
        "Statement": [
            {
                "Sid": "DenyOutputDownloadsExceptWhitelistedIps",
                "Effect": "Deny",
                "Principal": "*",
                "Action": ["s3:GetObject", "s3:GetObjectVersion"],
                "Resource": [
                    f"arn:aws:s3:::{S3_BUCKET}/processing/*",
                    f"arn:aws:s3:::{S3_BUCKET}/upscaling/*",
                ],
                "Condition": {
                    "NotIpAddress": {
                        "aws:SourceIp": MINIO_ALLOWED_DOWNLOAD_CIDRS,
                    }
                },
            }
        ],
    }
    client.put_bucket_policy(Bucket=S3_BUCKET, Policy=json.dumps(policy))


def _startup_storage_check():
    if not STORAGE_STARTUP_CHECK_ENABLED:
        return

    _validate_s3_config()
    client = _get_s3_client()
    _ensure_s3_bucket(client)
    _ensure_minio_download_policy(client)

    key = f".compression-healthcheck/{uuid.uuid4().hex}.multipart"
    upload_id = None
    try:
        response = client.create_multipart_upload(Bucket=S3_BUCKET, Key=key)
        upload_id = response.get("UploadId")
    finally:
        if upload_id:
            with suppress(Exception):
                client.abort_multipart_upload(Bucket=S3_BUCKET, Key=key, UploadId=upload_id)

    log.info(f"{STORAGE_PROVIDER} startup check passed for bucket {S3_BUCKET}")


async def _download_url(url: str, dest: str):
    log.info(f"Downloading {url[:80]}... → {dest}")
    async with httpx.AsyncClient(timeout=600.0, follow_redirects=True) as client:
        async with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                async for chunk in resp.aiter_bytes(chunk_size=8192):
                    f.write(chunk)
    log.info(f"Downloaded {os.path.getsize(dest) / (1024*1024):.1f} MB → {dest}")


def _upload_to_s3(local_path: str, key: str) -> str:
    _validate_s3_config()
    client = _get_s3_client()
    _ensure_s3_bucket(client)
    _ensure_minio_download_policy(client)
    client.upload_file(local_path, S3_BUCKET, key)
    url = client.generate_presigned_url(
        "get_object",
        Params={"Bucket": S3_BUCKET, "Key": key},
        ExpiresIn=min(S3_PRESIGNED_EXPIRY, 604800),
    )
    log.info(f"Uploaded to {STORAGE_PROVIDER} storage: s3://{S3_BUCKET}/{key}")
    return url


def _cleanup(*paths: str):
    for p in paths:
        if p and os.path.exists(p):
            try:
                os.remove(p)
            except OSError:
                pass


def _cleanup_tree(path: str):
    if path and os.path.isdir(path):
        shutil.rmtree(path, ignore_errors=True)


def _format_process_output(stdout: bytes, stderr: bytes) -> str:
    parts = []
    stdout_msg = stdout.decode(errors="replace").strip()
    stderr_msg = stderr.decode(errors="replace").strip()
    if stdout_msg:
        parts.append(f"stdout:\n{stdout_msg}")
    if stderr_msg:
        parts.append(f"stderr:\n{stderr_msg}")
    return "\n\n".join(parts)


async def _run_process(cmd: list[str], task_label: str, step: str) -> tuple[int | None, bytes, bytes, str]:
    try:
        log.info(f"[{task_label}] {step}: {' '.join(cmd)}")
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        stdout, stderr = await proc.communicate()
        return proc.returncode, stdout, stderr, ""
    except FileNotFoundError:
        return None, b"", b"", f"{cmd[0]} binary not found"
    except OSError as e:
        return None, b"", b"", f"Failed to start {cmd[0]}: {e}"


async def _run_capture(args: list[str], timeout: Optional[float] = 10.0) -> tuple[int, str, str]:
    proc = await asyncio.create_subprocess_exec(
        *args,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    try:
        if timeout is None:
            stdout, stderr = await proc.communicate()
        else:
            stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
    except asyncio.TimeoutError:
        proc.kill()
        await proc.wait()
        return 124, "", f"Timed out after {timeout:.1f}s"
    return (
        proc.returncode,
        stdout.decode(errors="replace"),
        stderr.decode(errors="replace"),
    )


def _human_bytes(num_bytes: Optional[int]) -> str:
    if num_bytes is None:
        return "unknown"
    value = float(num_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if value < 1024.0 or unit == "TB":
            return f"{value:.1f} {unit}"
        value /= 1024.0
    return f"{value:.1f} TB"


def _format_seconds(seconds: Optional[float]) -> str:
    if seconds is None:
        return "unknown"
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{int(minutes)}m {sec:.1f}s"
    hours, minutes = divmod(minutes, 60)
    return f"{int(hours)}h {int(minutes)}m {sec:.1f}s"


def _parse_fraction(value: Optional[str]) -> Optional[float]:
    if not value or value == "0/0":
        return None
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            denominator = float(den)
            if denominator == 0:
                return None
            return float(num) / denominator
        except (TypeError, ValueError):
            return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


async def _probe_input_video(task_label: str, local_input: str) -> dict[str, object]:
    input_size = os.path.getsize(local_input) if os.path.exists(local_input) else None
    probe: dict[str, object] = {
        "size_bytes": input_size,
        "duration": None,
        "width": None,
        "height": None,
        "fps": None,
        "codec": None,
        "bit_rate": None,
    }
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "format=duration,size,bit_rate:stream=codec_name,width,height,avg_frame_rate,r_frame_rate",
        "-of", "json",
        local_input,
    ]
    rc, stdout, stderr = await _run_capture(cmd, timeout=30.0)
    if rc != 0:
        log.warning(f"[{task_label}] ffprobe metadata unavailable (rc={rc}): {stderr.strip()}")
        return probe

    try:
        data = json.loads(stdout)
    except json.JSONDecodeError as exc:
        log.warning(f"[{task_label}] ffprobe metadata parse failed: {exc}")
        return probe

    fmt = data.get("format") or {}
    streams = data.get("streams") or []
    stream = streams[0] if streams else {}
    try:
        probe["duration"] = float(fmt["duration"]) if fmt.get("duration") else None
    except (TypeError, ValueError):
        pass
    if fmt.get("size"):
        try:
            probe["size_bytes"] = int(fmt["size"])
        except (TypeError, ValueError):
            pass
    if fmt.get("bit_rate"):
        try:
            probe["bit_rate"] = int(fmt["bit_rate"])
        except (TypeError, ValueError):
            pass
    probe["width"] = stream.get("width")
    probe["height"] = stream.get("height")
    probe["codec"] = stream.get("codec_name")
    probe["fps"] = _parse_fraction(stream.get("avg_frame_rate")) or _parse_fraction(stream.get("r_frame_rate"))

    resolution = f"{probe['width']}x{probe['height']}" if probe["width"] and probe["height"] else "unknown"
    fps = f"{probe['fps']:.2f}" if probe["fps"] else "unknown"
    bit_rate = f"{probe['bit_rate'] / 1_000_000:.2f} Mbps" if probe["bit_rate"] else "unknown"
    log.info(
        f"[{task_label}] Input metadata: duration={_format_seconds(probe['duration'])}, "
        f"size={_human_bytes(probe['size_bytes'])}, resolution={resolution}, fps={fps}, "
        f"codec={probe['codec'] or 'unknown'}, bitrate={bit_rate}"
    )
    return probe


async def _probe_video_bitrate_bps(task_label: str, path: str) -> Optional[int]:
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=bit_rate:format=bit_rate",
        "-of", "json",
        path,
    ]
    rc, stdout, stderr = await _run_capture(cmd, timeout=30.0)
    if rc != 0:
        log.warning(f"[{task_label}] Output bitrate probe failed (rc={rc}): {stderr.strip()}")
        return None

    try:
        data = json.loads(stdout)
        streams = data.get("streams") or []
        stream = streams[0] if streams else {}
        value = stream.get("bit_rate") or (data.get("format") or {}).get("bit_rate")
        return int(value) if value else None
    except (AttributeError, TypeError, ValueError, json.JSONDecodeError) as exc:
        log.warning(f"[{task_label}] Output bitrate probe parse failed: {exc}")
        return None


async def _validate_vbr_output(
    req: "CompressRequest",
    output_path: str,
    task_label: str,
) -> Optional[str]:
    if _codec_mode(req) != "VBR":
        return None
    if req.target_bitrate is None:
        return "target_bitrate is required for VBR mode"

    actual_bitrate = await _probe_video_bitrate_bps(task_label, output_path)
    if actual_bitrate is None:
        return "VBR output bitrate information is missing"

    allowed_bitrate = int(req.target_bitrate * (1 + AB_AV1_VBR_VALIDATION_TOLERANCE))
    if actual_bitrate > allowed_bitrate:
        return (
            "VBR output bitrate exceeded the configured limit: "
            f"target={req.target_bitrate / 1_000_000:.2f} Mbps, "
            f"actual={actual_bitrate / 1_000_000:.2f} Mbps, "
            f"maximum={allowed_bitrate / 1_000_000:.2f} Mbps"
        )

    log.info(
        f"[{task_label}] VBR bitrate validation passed: "
        f"actual={actual_bitrate / 1_000_000:.2f} Mbps, "
        f"target={req.target_bitrate / 1_000_000:.2f} Mbps, "
        f"maximum={allowed_bitrate / 1_000_000:.2f} Mbps"
    )
    return None


def _append_captured(existing: str, text: str) -> str:
    combined = existing + text
    if len(combined) > PROCESS_OUTPUT_CAPTURE_CHARS:
        return combined[-PROCESS_OUTPUT_CAPTURE_CHARS:]
    return combined


async def _stream_process_output(task_label: str, stream: asyncio.StreamReader, stream_name: str) -> str:
    captured = ""
    pending = ""
    while True:
        chunk = await stream.read(4096)
        if not chunk:
            break
        text = chunk.decode(errors="replace")
        captured = _append_captured(captured, text)
        if not COMPRESSION_LOG_AB_AV1_OUTPUT:
            continue
        pending += text.replace("\r", "\n")
        while "\n" in pending:
            line, pending = pending.split("\n", 1)
            line = line.strip()
            if line:
                log.info(f"[{task_label}] ab-av1 {stream_name}: {line[:1200]}")

    line = pending.strip()
    if COMPRESSION_LOG_AB_AV1_OUTPUT and line:
        log.info(f"[{task_label}] ab-av1 {stream_name}: {line[:1200]}")
    return captured


async def _monitor_output_status(
    task_label: str,
    output_path: str,
    input_duration: Optional[float],
    encode_start: float,
) -> None:
    interval = max(1.0, COMPRESSION_PROGRESS_INTERVAL_SECONDS)
    last_size = 0
    last_time = encode_start

    while True:
        await asyncio.sleep(interval)
        now = time.monotonic()
        elapsed = now - encode_start
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else 0
        delta_seconds = max(now - last_time, 0.001)
        write_rate = max(output_size - last_size, 0) / delta_seconds
        last_size = output_size
        last_time = now
        speed_hint = ""
        if input_duration and elapsed > 0:
            speed_hint = f", input_duration/elapsed={input_duration / elapsed:.2f}x"
        log.info(
            f"[{task_label}] Compression status: elapsed={_format_seconds(elapsed)}, "
            f"output={_human_bytes(output_size)}, write_rate={_human_bytes(int(write_rate))}/s"
            f"{speed_hint}"
        )


async def _probe_duration_seconds(path: str, task_label: str) -> float | None:
    cmd = [
        FFPROBE_BIN,
        "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    returncode, stdout, stderr, run_error = await _run_process(cmd, task_label, "ffprobe duration")
    if returncode != 0 or run_error:
        detail = run_error or stderr.decode(errors="replace").strip()
        log.warning(f"[{task_label}] Failed to probe duration: {detail}")
        return None
    try:
        return float(stdout.decode().strip())
    except ValueError:
        log.warning(f"[{task_label}] ffprobe returned invalid duration: {stdout!r}")
        return None


async def _probe_segment_duration_seconds(path: str, task_label: str) -> float:
    duration = await _probe_duration_seconds(path, task_label)
    if duration is None:
        raise RuntimeError(f"Unable to probe segment duration: {path}")
    return duration


def _format_timestamp(seconds: float) -> str:
    milliseconds = int(round(seconds * 1000))
    hours, remainder = divmod(milliseconds, 3_600_000)
    minutes, remainder = divmod(remainder, 60_000)
    whole_seconds, milliseconds = divmod(remainder, 1000)
    return f"{hours:02d}:{minutes:02d}:{whole_seconds:02d}.{milliseconds:03d}"


async def _log_chunk_seams(segments: list[str], task_label: str) -> list[float]:
    durations = [
        await _probe_segment_duration_seconds(segment, task_label)
        for segment in segments
    ]
    seams: list[float] = []
    elapsed = 0.0
    for duration in durations[:-1]:
        elapsed += duration
        seams.append(elapsed)

    log.info(
        f"[{task_label}] chunk_seams "
        + json.dumps(
            {
                "seam_seconds": [round(seam, 3) for seam in seams],
                "seam_timestamps": [_format_timestamp(seam) for seam in seams],
                "segment_durations_seconds": [round(duration, 3) for duration in durations],
            }
        )
    )
    return seams


def _format_float(value: float) -> str:
    return f"{value:g}"


def _encoder_for_codec(codec: str) -> str:
    return CODEC_MAP.get(codec.upper(), AB_AV1_ENCODER_AV1)


def _is_nvenc_encoder(encoder: str) -> bool:
    return encoder.lower().endswith("_nvenc")


def _is_hevc_nvenc_encoder(encoder: str) -> bool:
    return encoder.lower() == "hevc_nvenc"


def _request_has_field(req: "CompressRequest", field: str) -> bool:
    fields = getattr(req, "model_fields_set", None)
    if fields is None:
        fields = getattr(req, "__fields_set__", set())
    return field in fields


def _preset_for_encoder(req: "CompressRequest", encoder: str) -> str:
    if _is_hevc_nvenc_encoder(encoder) and not _request_has_field(req, "preset"):
        return AB_AV1_HEVC_NVENC_PRESET
    return req.preset


def _normalize_preset(encoder: str, preset: str) -> str:
    if _is_nvenc_encoder(encoder):
        return preset
    if encoder == "libsvtav1" and preset.startswith("p") and preset[1:].isdigit():
        return AB_AV1_LIBSVTAV1_PRESET
    if encoder == "libx264" and preset.startswith("p") and preset[1:].isdigit():
        return AB_AV1_X264_PRESET
    if encoder == "libx265" and preset.startswith("p") and preset[1:].isdigit():
        return AB_AV1_X265_PRESET
    if encoder == "libvpx-vp9" and preset.startswith("p") and preset[1:].isdigit():
        return preset[1:]
    return preset


def _target_vmaf(req: "CompressRequest") -> float:
    return req.target_vmaf if req.target_vmaf is not None else AB_AV1_DEFAULT_TARGET_VMAF


def _codec_mode(req: "CompressRequest") -> str:
    return req.codec_mode.upper()


def _ab_av1_enc_args(enc_args: tuple[str, ...] | list[str]) -> list[str]:
    args: list[str] = []
    for enc_arg in enc_args:
        args.extend(["--enc", enc_arg])
    return args


def _ab_av1_vbr_args(req: "CompressRequest", encoder: str) -> list[str]:
    if _codec_mode(req) != "VBR":
        return []
    if not _is_nvenc_encoder(encoder):
        raise ValueError(
            f"VBR mode requires an NVENC encoder, but codec {req.codec!r} maps to {encoder!r}"
        )
    if req.target_bitrate is None:
        raise ValueError("target_bitrate is required for VBR mode")

    max_rate = max(1, int(req.target_bitrate * AB_AV1_VBR_MAX_RATE_RATIO))
    return _ab_av1_enc_args([f"maxrate={max_rate}"])


def _ab_av1_vmaf_args() -> list[str]:
    args: list[str] = []
    if AB_AV1_VMAF_MODEL:
        args.extend(["--vmaf", f"model={AB_AV1_VMAF_MODEL}"])
    if AB_AV1_VMAF_POOL:
        args.extend(["--vmaf", f"pool={AB_AV1_VMAF_POOL}"])
    if AB_AV1_VMAF_N_SUBSAMPLE:
        args.extend(["--vmaf", f"n_subsample={AB_AV1_VMAF_N_SUBSAMPLE}"])
    if AB_AV1_VMAF_N_THREADS:
        args.extend(["--vmaf", f"n_threads={AB_AV1_VMAF_N_THREADS}"])
    if AB_AV1_VMAF_FPS:
        args.extend(["--vmaf-fps", AB_AV1_VMAF_FPS])
    if AB_AV1_VMAF_SCALE:
        args.extend(["--vmaf-scale", AB_AV1_VMAF_SCALE])
    if AB_AV1_VMAF_CUDA:
        args.append("--vmaf-cuda")
    return args


def _nvenc_target_min_crf(target_vmaf: float) -> float:
    if target_vmaf >= 93:
        return AB_AV1_NVENC_MIN_CRF_VMAF_93
    if target_vmaf >= 89:
        return AB_AV1_NVENC_MIN_CRF_VMAF_89
    return AB_AV1_NVENC_MIN_CRF_VMAF_85


def _nvenc_crf_bounds(
    encoder: str,
    target_vmaf: float,
    full_crf_range: bool = False,
) -> tuple[float, float]:
    hevc = _is_hevc_nvenc_encoder(encoder)
    configured_min = AB_AV1_HEVC_NVENC_MIN_CRF if hevc else AB_AV1_NVENC_MIN_CRF
    configured_max = AB_AV1_HEVC_NVENC_MAX_CRF if hevc else AB_AV1_NVENC_MAX_CRF
    min_crf = float(configured_min)
    max_crf = float(configured_max)

    if AB_AV1_NVENC_TARGETED_MIN_CRF and not full_crf_range:
        targeted_min = max(min_crf, _nvenc_target_min_crf(target_vmaf))
        if targeted_min < max_crf:
            min_crf = targeted_min

    return min_crf, max_crf


def _uses_targeted_nvenc_crf_range(encoder: str, target_vmaf: float) -> bool:
    if not _is_nvenc_encoder(encoder):
        return False
    return _nvenc_crf_bounds(encoder, target_vmaf) != _nvenc_crf_bounds(
        encoder,
        target_vmaf,
        full_crf_range=True,
    )


def _ab_av1_nvenc_search_args(
    encoder: str,
    target_vmaf: float,
    full_crf_range: bool = False,
) -> list[str]:
    if not _is_nvenc_encoder(encoder):
        return []

    hevc = _is_hevc_nvenc_encoder(encoder)
    min_crf, max_crf = _nvenc_crf_bounds(encoder, target_vmaf, full_crf_range)
    thorough = AB_AV1_HEVC_NVENC_THOROUGH if hevc else AB_AV1_NVENC_THOROUGH
    full_search = AB_AV1_HEVC_NVENC_FULL_SEARCH if hevc else AB_AV1_NVENC_FULL_SEARCH

    args = [
        "--min-crf", _format_float(min_crf),
        "--max-crf", _format_float(max_crf),
        "--crf-increment", _format_float(AB_AV1_NVENC_CRF_INCREMENT),
    ]
    if thorough:
        args.append("--thorough")
    if full_search:
        args.extend(["--sample-duration", "0s"])
    return args


def _ab_av1_nvenc_codec_args(encoder: str) -> list[str]:
    encoder_name = encoder.lower()
    enc_args: list[str] = []
    if encoder_name == "av1_nvenc":
        enc_args.extend(AB_AV1_AV1_NVENC_ENCODER_ARGS)
    if _is_hevc_nvenc_encoder(encoder):
        enc_args.extend(AB_AV1_HEVC_NVENC_ENCODER_ARGS)
    if encoder_name in {"av1_nvenc", "hevc_nvenc"}:
        enc_args.extend(AB_AV1_NVENC_BFRAME_ARGS)
        if AB_AV1_NVENC_TEMPORAL_AQ:
            enc_args.append("temporal-aq=1")
        if AB_AV1_NVENC_TEMPORAL_FILTER_LEVEL:
            enc_args.append(f"tf_level={AB_AV1_NVENC_TEMPORAL_FILTER_LEVEL}")
    return _ab_av1_enc_args(enc_args)


def _uses_full_ab_av1_search(encoder: str) -> bool:
    if not _is_nvenc_encoder(encoder):
        return False
    if _is_hevc_nvenc_encoder(encoder):
        return AB_AV1_HEVC_NVENC_FULL_SEARCH
    return AB_AV1_NVENC_FULL_SEARCH


def _uses_direct_nvenc_cq(encoder: str) -> bool:
    encoder_name = encoder.lower()
    if encoder_name == "av1_nvenc":
        direct_enabled = AB_AV1_NVENC_DIRECT_CQ_ENABLED
    elif encoder_name == "hevc_nvenc":
        direct_enabled = AB_AV1_HEVC_NVENC_DIRECT_CQ_ENABLED
    else:
        direct_enabled = False
    return direct_enabled and AB_AV1_NVENC_FULL_VMAF_VERIFY


def _quantize_crf(crf: float) -> float:
    increment = AB_AV1_NVENC_CRF_INCREMENT
    return round(crf / increment) * increment


def _nvenc_initial_crf(encoder: str, target_vmaf: float) -> float:
    if _is_hevc_nvenc_encoder(encoder):
        anchors = (
            (85.0, AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_85),
            (89.0, AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_89),
            (93.0, AB_AV1_HEVC_NVENC_INITIAL_CRF_VMAF_93),
        )
    else:
        anchors = (
            (85.0, AB_AV1_NVENC_INITIAL_CRF_VMAF_85),
            (89.0, AB_AV1_NVENC_INITIAL_CRF_VMAF_89),
            (93.0, AB_AV1_NVENC_INITIAL_CRF_VMAF_93),
        )
    if target_vmaf <= anchors[1][0]:
        lower, upper = anchors[0], anchors[1]
    else:
        lower, upper = anchors[1], anchors[2]

    target_span = upper[0] - lower[0]
    fraction = (target_vmaf - lower[0]) / target_span
    estimated_crf = lower[1] + fraction * (upper[1] - lower[1])
    min_crf, max_crf = _nvenc_crf_bounds(encoder, target_vmaf, full_crf_range=True)
    return min(max(_quantize_crf(estimated_crf), min_crf), max_crf)


def _adaptive_retry_crf(
    encoder: str,
    selected_crf: float,
    actual_vmaf: float,
    target_vmaf: float,
) -> Optional[float]:
    score_delta = actual_vmaf - target_vmaf
    vmaf_per_crf = (
        AB_AV1_HEVC_NVENC_VMAF_PER_CRF
        if _is_hevc_nvenc_encoder(encoder)
        else AB_AV1_NVENC_VMAF_PER_CRF
    )
    adjustment = max(
        -AB_AV1_NVENC_MAX_CRF_ADJUSTMENT,
        min(score_delta / vmaf_per_crf, AB_AV1_NVENC_MAX_CRF_ADJUSTMENT),
    )
    candidate = selected_crf + adjustment
    # Always round toward higher quality. This lets an overshooting encode use
    # more compression without predicting below the threshold, while a miss is
    # guaranteed to move at least one complete CQ increment toward quality.
    retry_crf = (
        math.floor((candidate + 1e-9) / AB_AV1_NVENC_CRF_INCREMENT)
        * AB_AV1_NVENC_CRF_INCREMENT
    )
    if actual_vmaf < target_vmaf:
        retry_crf = min(retry_crf, selected_crf - AB_AV1_NVENC_CRF_INCREMENT)

    min_crf, max_crf = _nvenc_crf_bounds(encoder, target_vmaf, full_crf_range=True)
    retry_crf = min(max(retry_crf, min_crf), max_crf)
    if abs(retry_crf - selected_crf) < AB_AV1_NVENC_CRF_INCREMENT / 2:
        return None
    return retry_crf


def _ab_av1_sample_every(req: "CompressRequest", input_duration: Optional[float]) -> str:
    if req.sample_every:
        return req.sample_every
    if (
        input_duration is not None
        and AB_AV1_SHORT_SAMPLE_EVERY
        and AB_AV1_SHORT_SAMPLE_MAX_DURATION_SECONDS > 0
        and input_duration <= AB_AV1_SHORT_SAMPLE_MAX_DURATION_SECONDS
    ):
        return AB_AV1_SHORT_SAMPLE_EVERY
    return AB_AV1_SAMPLE_EVERY


def _vmaf_fallback_targets(initial_target: float) -> list[float]:
    if not AB_AV1_VMAF_FALLBACK_ENABLED:
        return [initial_target]

    floor = max(0.0, initial_target - AB_AV1_VMAF_FALLBACK_FLOOR_DELTA)
    targets: list[float] = []
    current = initial_target

    while len(targets) < AB_AV1_VMAF_FALLBACK_MAX_ATTEMPTS:
        target = max(current, floor)
        if not targets or abs(target - targets[-1]) > 0.0001:
            targets.append(target)
        if target <= floor:
            break
        current = max(target - AB_AV1_VMAF_FALLBACK_STEP, floor)

    return targets or [initial_target]


def _is_no_suitable_crf_error(output: str) -> bool:
    text = output.lower()
    return (
        re.search(r"failed to find (?:a )?suitable crf", text) is not None
        or "no suitable crf" in text
    )


def _max_encoded_percent(req: "CompressRequest") -> Optional[float]:
    if req.max_encoded_percent is not None:
        return req.max_encoded_percent
    if AB_AV1_MAX_ENCODED_PERCENT:
        return float(AB_AV1_MAX_ENCODED_PERCENT)
    return None


def _build_ab_av1_args(
    req: "CompressRequest",
    encoder: str,
    local_input: str,
    output_path: str,
    target_vmaf: float,
    input_duration: Optional[float],
    full_crf_range: bool = False,
) -> list[str]:
    args = [
        "auto-encode",
        "-i", local_input,
        "-o", output_path,
        "-e", encoder,
        "--preset", _normalize_preset(encoder, _preset_for_encoder(req, encoder)),
        "--min-vmaf", _format_float(target_vmaf),
    ]
    args.extend(_ab_av1_vmaf_args())
    args.extend(_ab_av1_nvenc_search_args(encoder, target_vmaf, full_crf_range))
    # VBR is scored exactly like CRF with one extra validity constraint: the
    # completed video bitrate must stay below target + tolerance. Keep the CRF
    # encoder settings (including HEVC's quality-mode defaults) and add only a
    # peak cap; setting b:v to the requested bitrate and enabling full-resolution
    # multipass makes every VMAF probe slower without improving the score.
    args.extend(_ab_av1_nvenc_codec_args(encoder))
    args.extend(_ab_av1_vbr_args(req, encoder))

    sample_every = _ab_av1_sample_every(req, input_duration)
    if sample_every and not _uses_full_ab_av1_search(encoder):
        args.extend(["--sample-every", sample_every])

    max_encoded_percent = _max_encoded_percent(req)
    if max_encoded_percent is not None:
        args.extend(["--max-encoded-percent", _format_float(max_encoded_percent)])

    if AB_AV1_HWACCEL and _is_nvenc_encoder(encoder):
        args.extend(["--enc-input", f"hwaccel={AB_AV1_HWACCEL}"])

    if req.target_width and req.target_height:
        w = req.target_width if req.target_width % 2 == 0 else req.target_width - 1
        h = req.target_height if req.target_height % 2 == 0 else req.target_height - 1
        args.extend(["--vfilter", f"scale={w}:{h},setsar=1"])
    else:
        args.extend(["--vfilter", "setsar=1"])

    return args


def _build_ab_av1_encode_args(
    req: "CompressRequest",
    encoder: str,
    local_input: str,
    output_path: str,
    crf: float,
) -> list[str]:
    args = [
        "encode",
        "-i", local_input,
        "-o", output_path,
        "-e", encoder,
        "--preset", _normalize_preset(encoder, _preset_for_encoder(req, encoder)),
        "--crf", _format_float(crf),
    ]
    args.extend(_ab_av1_nvenc_codec_args(encoder))
    args.extend(_ab_av1_vbr_args(req, encoder))

    if AB_AV1_HWACCEL and _is_nvenc_encoder(encoder):
        args.extend(["--enc-input", f"hwaccel={AB_AV1_HWACCEL}"])

    if req.target_width and req.target_height:
        w = req.target_width if req.target_width % 2 == 0 else req.target_width - 1
        h = req.target_height if req.target_height % 2 == 0 else req.target_height - 1
        args.extend(["--vfilter", f"scale={w}:{h},setsar=1"])
    else:
        args.extend(["--vfilter", "setsar=1"])

    return args


_CRF_VMAF_RE = re.compile(
    r"\bcrf\s+([0-9]+(?:[._][0-9]+)?)\s+VMAF\s+([0-9]+(?:\.[0-9]+)?)",
    re.IGNORECASE,
)
_ANSI_ESCAPE_RE = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")


def _parse_selected_crf(stdout_text: str, stderr_text: str, target_vmaf: float) -> Optional[float]:
    output = _ANSI_ESCAPE_RE.sub("", f"{stdout_text}\n{stderr_text}")
    passing_crfs = [
        float(crf.replace("_", "."))
        for crf, vmaf in _CRF_VMAF_RE.findall(output)
        if float(vmaf) + 0.000001 >= target_vmaf
    ]
    return max(passing_crfs) if passing_crfs else None


def _parse_vmaf_score(stdout_text: str, stderr_text: str) -> Optional[float]:
    output = _ANSI_ESCAPE_RE.sub("", f"{stdout_text}\n{stderr_text}")
    for line in reversed(output.splitlines()):
        value = line.strip()
        if re.fullmatch(r"[0-9]+(?:\.[0-9]+)?", value):
            return float(value)
    matches = re.findall(r"VMAF(?:\s+score)?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", output, re.I)
    return float(matches[-1]) if matches else None


def _build_ab_av1_cmd(ab_av1_args: list[str]) -> list[str]:
    return [AB_AV1_BIN] + ab_av1_args


async def _run_ab_av1_attempt(
    task_label: str,
    cmd: list[str],
    output_path: str,
    input_duration: Optional[float],
) -> tuple[int, str, str, float]:
    encode_start = time.monotonic()
    monitor_task: Optional[asyncio.Task] = None
    stdout_task: Optional[asyncio.Task] = None
    stderr_task: Optional[asyncio.Task] = None
    stdout_text = ""
    stderr_text = ""

    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        monitor_task = asyncio.create_task(
            _monitor_output_status(task_label, output_path, input_duration, encode_start)
        )
        stdout_task = asyncio.create_task(_stream_process_output(task_label, proc.stdout, "stdout"))
        stderr_task = asyncio.create_task(_stream_process_output(task_label, proc.stderr, "stderr"))
        returncode = await proc.wait()
        stdout_text, stderr_text = await asyncio.gather(stdout_task, stderr_task)
        return returncode, stdout_text, stderr_text, time.monotonic() - encode_start
    except FileNotFoundError:
        return 127, "", f"{AB_AV1_BIN} binary not found", 0.0
    except OSError as e:
        return 1, "", f"Failed to start {AB_AV1_BIN}: {e}", 0.0
    finally:
        if monitor_task:
            monitor_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await monitor_task
        for stream_task in (stdout_task, stderr_task):
            if stream_task and not stream_task.done():
                stream_task.cancel()
                with contextlib.suppress(asyncio.CancelledError):
                    await stream_task


async def _run_direct_nvenc_encode(
    task_label: str,
    req: "CompressRequest",
    encoder: str,
    local_input: str,
    output_path: str,
    input_duration: Optional[float],
    requested_target_vmaf: float,
) -> dict[str, object]:
    initial_crf = _nvenc_initial_crf(encoder, requested_target_vmaf)
    cmd = _build_ab_av1_cmd(
        _build_ab_av1_encode_args(
            req,
            encoder,
            local_input,
            output_path,
            initial_crf,
        )
    )
    _cleanup(output_path)
    log.info(
        f"[{task_label}] Direct NVENC warm-start encode "
        f"(target_vmaf={requested_target_vmaf:.2f}, initial_cq={initial_crf:g})"
    )
    log.info(f"[{task_label}] Command: {' '.join(cmd)}")
    returncode, stdout_text, stderr_text, encode_seconds = await _run_ab_av1_attempt(
        task_label,
        cmd,
        output_path,
        input_duration,
    )
    return {
        "returncode": returncode,
        "stdout_text": stdout_text,
        "stderr_text": stderr_text,
        "encode_seconds": encode_seconds,
        "attempt_count": 1,
        "effective_target_vmaf": requested_target_vmaf,
        "selected_crf": initial_crf,
    }


async def _measure_full_vmaf(
    task_label: str,
    reference_path: str,
    distorted_path: str,
) -> tuple[Optional[float], float, Optional[str]]:
    cmd = _build_ab_av1_cmd(
        [
            "vmaf",
            "--reference", reference_path,
            "--distorted", distorted_path,
            *_ab_av1_vmaf_args(),
        ]
    )
    log.info(f"[{task_label}] Full VMAF verification command: {' '.join(cmd)}")
    verify_start = time.monotonic()
    try:
        returncode, stdout_text, stderr_text = await _run_capture(
            cmd,
            timeout=AB_AV1_NVENC_VERIFY_TIMEOUT_SECONDS,
        )
    except (FileNotFoundError, OSError) as exc:
        elapsed = time.monotonic() - verify_start
        return None, elapsed, f"Could not start full VMAF verification: {exc}"

    elapsed = time.monotonic() - verify_start
    if returncode != 0:
        detail = (stderr_text or stdout_text).strip() or "no process output"
        return None, elapsed, f"Full VMAF verification failed (rc={returncode}): {detail}"

    score = _parse_vmaf_score(stdout_text, stderr_text)
    if score is None:
        detail = (stdout_text or stderr_text).strip() or "no process output"
        return None, elapsed, f"Could not parse full VMAF verification result: {detail}"
    return score, elapsed, None


async def _vbr_output_saturates_rate_cap(
    req: "CompressRequest",
    output_path: str,
    task_label: str,
) -> tuple[bool, Optional[int], Optional[int]]:
    if _codec_mode(req) != "VBR" or req.target_bitrate is None:
        return False, None, None

    actual_bitrate = await _probe_video_bitrate_bps(task_label, output_path)
    if actual_bitrate is None:
        return False, None, None

    encoder_cap = max(1, int(req.target_bitrate * AB_AV1_VBR_MAX_RATE_RATIO))
    saturation = actual_bitrate / encoder_cap
    log.info(
        f"[{task_label}] VBR rate-cap utilization: actual={actual_bitrate / 1_000_000:.2f} Mbps, "
        f"encoder_cap={encoder_cap / 1_000_000:.2f} Mbps, utilization={saturation:.3f}"
    )
    return saturation >= AB_AV1_NVENC_RATE_CAP_SATURATION_RATIO, actual_bitrate, encoder_cap


async def _verify_and_retry_nvenc_output(
    task_label: str,
    req: "CompressRequest",
    encoder: str,
    local_input: str,
    output_path: str,
    input_duration: Optional[float],
    requested_target_vmaf: float,
    selected_crf: Optional[float],
) -> dict[str, object]:
    required_vmaf = requested_target_vmaf
    actual_vmaf, verification_seconds, error = await _measure_full_vmaf(
        task_label,
        local_input,
        output_path,
    )
    if error:
        return {
            "error": error,
            "actual_vmaf": actual_vmaf,
            "verification_seconds": verification_seconds,
            "retry_encode_seconds": 0.0,
            "retry_count": 0,
        }

    log.info(
        f"[{task_label}] Full VMAF NEG verification: actual={actual_vmaf:.6f}, "
        f"required={required_vmaf:.6f}, elapsed={_format_seconds(verification_seconds)}"
    )
    score_delta = actual_vmaf - required_vmaf
    needs_quality_retry = score_delta < 0
    needs_compression_retry = score_delta >= AB_AV1_NVENC_RETRY_MIN_OVERSHOOT
    if not needs_quality_retry and not needs_compression_retry:
        return {
            "error": None,
            "actual_vmaf": actual_vmaf,
            "verification_seconds": verification_seconds,
            "retry_encode_seconds": 0.0,
            "retry_count": 0,
        }

    if needs_quality_retry:
        saturated, actual_bitrate, encoder_cap = await _vbr_output_saturates_rate_cap(
            req,
            output_path,
            task_label,
        )
        if saturated:
            return {
                "error": (
                    f"Full VMAF NEG {actual_vmaf:.6f} is below required {required_vmaf:.6f}, "
                    f"while output bitrate {actual_bitrate / 1_000_000:.2f} Mbps already uses "
                    f"at least {AB_AV1_NVENC_RATE_CAP_SATURATION_RATIO:.0%} of the "
                    f"{encoder_cap / 1_000_000:.2f} Mbps encoder cap; lower CQ cannot add "
                    "enough bitrate, so the futile CRF search was stopped"
                ),
                "actual_vmaf": actual_vmaf,
                "verification_seconds": verification_seconds,
                "retry_encode_seconds": 0.0,
                "retry_count": 0,
            }

    if not AB_AV1_NVENC_VERIFY_RETRY and needs_quality_retry:
        return {
            "error": (
                f"Full VMAF NEG {actual_vmaf:.6f} is below required {required_vmaf:.6f}; "
                "quality retry is disabled"
            ),
            "actual_vmaf": actual_vmaf,
            "verification_seconds": verification_seconds,
            "retry_encode_seconds": 0.0,
            "retry_count": 0,
        }
    if not AB_AV1_NVENC_VERIFY_RETRY:
        return {
            "error": None,
            "actual_vmaf": actual_vmaf,
            "verification_seconds": verification_seconds,
            "retry_encode_seconds": 0.0,
            "retry_count": 0,
        }
    if selected_crf is None:
        if not needs_quality_retry:
            log.warning(
                f"[{task_label}] Cannot optimize VMAF overshoot because the selected CQ "
                "was not available; retaining the verified first encode"
            )
            return {
                "error": None,
                "actual_vmaf": actual_vmaf,
                "verification_seconds": verification_seconds,
                "retry_encode_seconds": 0.0,
                "retry_count": 0,
            }
        return {
            "error": (
                f"Full VMAF NEG {actual_vmaf:.6f} is below required {required_vmaf:.6f}, "
                "but the selected CQ could not be parsed for a bounded retry"
            ),
            "actual_vmaf": actual_vmaf,
            "verification_seconds": verification_seconds,
            "retry_encode_seconds": 0.0,
            "retry_count": 0,
        }

    retry_crf = _adaptive_retry_crf(
        encoder,
        selected_crf,
        actual_vmaf,
        required_vmaf,
    )
    if retry_crf is None:
        if not needs_quality_retry:
            return {
                "error": None,
                "actual_vmaf": actual_vmaf,
                "verification_seconds": verification_seconds,
                "retry_encode_seconds": 0.0,
                "retry_count": 0,
            }
        return {
            "error": (
                f"Full VMAF NEG {actual_vmaf:.6f} is below required {required_vmaf:.6f}, "
                f"and CQ {selected_crf:g} cannot move further toward quality"
            ),
            "actual_vmaf": actual_vmaf,
            "verification_seconds": verification_seconds,
            "retry_encode_seconds": 0.0,
            "retry_count": 0,
        }

    retry_output_path = f"{output_path}.quality-retry.mp4"
    retry_args = _build_ab_av1_encode_args(
        req,
        encoder,
        local_input,
        retry_output_path,
        retry_crf,
    )
    retry_cmd = _build_ab_av1_cmd(retry_args)
    _cleanup(retry_output_path)
    await _track_temp_files(retry_output_path)
    retry_reason = "recover quality" if needs_quality_retry else "reduce size"
    log.warning(
        f"[{task_label}] Adaptive NVENC retry to {retry_reason} at CQ {retry_crf:g} "
        f"(selected_cq={selected_crf:g}, measured_vmaf={actual_vmaf:.6f}, "
        f"target_vmaf={required_vmaf:.6f}, command={' '.join(retry_cmd)})"
    )
    retry_encode_seconds = 0.0
    try:
        returncode, stdout_text, stderr_text, retry_encode_seconds = await _run_ab_av1_attempt(
            task_label,
            retry_cmd,
            retry_output_path,
            input_duration,
        )
        if returncode != 0:
            detail = (stderr_text or stdout_text).strip() or "retry failed without output"
            if not needs_quality_retry:
                log.warning(
                    f"[{task_label}] Compression retry failed; retaining verified CQ "
                    f"{selected_crf:g}: {detail}"
                )
                return {
                    "error": None,
                    "actual_vmaf": actual_vmaf,
                    "verification_seconds": verification_seconds,
                    "retry_encode_seconds": retry_encode_seconds,
                    "retry_count": 1,
                }
            return {
                "error": f"Full VMAF quality retry failed (rc={returncode}): {detail}",
                "actual_vmaf": actual_vmaf,
                "verification_seconds": verification_seconds,
                "retry_encode_seconds": retry_encode_seconds,
                "retry_count": 1,
            }
        if not os.path.exists(retry_output_path):
            retry_error = "Adaptive NVENC retry did not create an output file"
            if not needs_quality_retry:
                log.warning(f"[{task_label}] {retry_error}; retaining verified first encode")
                retry_error = None
            return {
                "error": retry_error,
                "actual_vmaf": actual_vmaf,
                "verification_seconds": verification_seconds,
                "retry_encode_seconds": retry_encode_seconds,
                "retry_count": 1,
            }

        retry_vmaf, retry_verify_seconds, error = await _measure_full_vmaf(
            task_label,
            local_input,
            retry_output_path,
        )
        verification_seconds += retry_verify_seconds
        if error:
            if not needs_quality_retry:
                log.warning(f"[{task_label}] {error}; retaining verified first encode")
                return {
                    "error": None,
                    "actual_vmaf": actual_vmaf,
                    "verification_seconds": verification_seconds,
                    "retry_encode_seconds": retry_encode_seconds,
                    "retry_count": 1,
                }
            return {
                "error": error,
                "actual_vmaf": retry_vmaf,
                "verification_seconds": verification_seconds,
                "retry_encode_seconds": retry_encode_seconds,
                "retry_count": 1,
            }

        log.info(
            f"[{task_label}] Full VMAF NEG verification after retry: actual={retry_vmaf:.6f}, "
            f"required={required_vmaf:.6f}, elapsed={_format_seconds(retry_verify_seconds)}"
        )
        if retry_vmaf + 0.000001 < required_vmaf:
            if not needs_quality_retry:
                log.warning(
                    f"[{task_label}] Compression retry undershot VMAF; retaining verified "
                    f"CQ {selected_crf:g} output"
                )
                return {
                    "error": None,
                    "actual_vmaf": actual_vmaf,
                    "verification_seconds": verification_seconds,
                    "retry_encode_seconds": retry_encode_seconds,
                    "retry_count": 1,
                }
            return {
                "error": (
                    f"Full VMAF NEG {retry_vmaf:.6f} remains below required "
                    f"{required_vmaf:.6f} after the single adaptive quality retry"
                ),
                "actual_vmaf": retry_vmaf,
                "verification_seconds": verification_seconds,
                "retry_encode_seconds": retry_encode_seconds,
                "retry_count": 1,
            }

        os.replace(retry_output_path, output_path)
        return {
            "error": None,
            "actual_vmaf": retry_vmaf,
            "verification_seconds": verification_seconds,
            "retry_encode_seconds": retry_encode_seconds,
            "retry_count": 1,
        }
    finally:
        _cleanup(retry_output_path)
        await _untrack_temp_files(retry_output_path)


async def _run_ab_av1_with_fallback(
    task_label: str,
    req: "CompressRequest",
    encoder: str,
    local_input: str,
    output_path: str,
    input_duration: Optional[float],
    requested_target_vmaf: float,
    vmaf_targets: list[float],
) -> dict[str, object]:
    stdout_text = ""
    stderr_text = ""
    returncode = 1
    encode_seconds = 0.0
    effective_target_vmaf = requested_target_vmaf
    attempt_count = 0
    target_index = 0
    full_crf_range = False
    max_attempts = sum(
        1 + int(_uses_targeted_nvenc_crf_range(encoder, target))
        for target in vmaf_targets
    )

    while target_index < len(vmaf_targets):
        target_vmaf = vmaf_targets[target_index]
        attempt_count += 1
        effective_target_vmaf = target_vmaf
        sample_every = _ab_av1_sample_every(req, input_duration)
        ab_av1_args = _build_ab_av1_args(
            req,
            encoder,
            local_input,
            output_path,
            target_vmaf,
            input_duration,
            full_crf_range=full_crf_range,
        )
        cmd = _build_ab_av1_cmd(ab_av1_args)
        if attempt_count > 1:
            _cleanup(output_path)

        log.info(
            f"[{task_label}] Compression attempt {attempt_count}/{max_attempts} "
            f"(target_vmaf={target_vmaf:.2f}, "
            f"crf_range={'full' if full_crf_range else 'targeted'}, "
            f"sample_every={sample_every or 'ab-av1 default'})"
        )
        log.info(f"[{task_label}] Command: {' '.join(cmd)}")

        returncode, stdout_text, stderr_text, attempt_seconds = await _run_ab_av1_attempt(
            task_label,
            cmd,
            output_path,
            input_duration,
        )
        encode_seconds += attempt_seconds

        if returncode == 0:
            break

        combined_output = f"{stderr_text}\n{stdout_text}".strip()
        if not _is_no_suitable_crf_error(combined_output):
            break

        if (
            not full_crf_range
            and _uses_targeted_nvenc_crf_range(encoder, target_vmaf)
        ):
            full_crf_range = True
            log.warning(
                f"[{task_label}] ab-av1 could not find a suitable CRF in the targeted range at "
                f"target_vmaf={target_vmaf:.2f}; retrying the same target with the configured "
                f"full CRF range (attempt_elapsed={_format_seconds(attempt_seconds)}, "
                f"total_encode_elapsed={_format_seconds(encode_seconds)})"
            )
            continue

        target_index += 1
        full_crf_range = False
        if target_index >= len(vmaf_targets):
            break

        next_target = vmaf_targets[target_index]
        log.warning(
            f"[{task_label}] ab-av1 could not find a suitable CRF at "
            f"target_vmaf={target_vmaf:.2f}; retrying with target_vmaf={next_target:.2f} "
            f"(attempt_elapsed={_format_seconds(attempt_seconds)}, "
            f"total_encode_elapsed={_format_seconds(encode_seconds)})"
        )

    return {
        "returncode": returncode,
        "stdout_text": stdout_text,
        "stderr_text": stderr_text,
        "encode_seconds": encode_seconds,
        "attempt_count": attempt_count,
        "effective_target_vmaf": effective_target_vmaf,
        "selected_crf": (
            _parse_selected_crf(stdout_text, stderr_text, effective_target_vmaf)
            if returncode == 0 and _is_nvenc_encoder(encoder)
            else None
        ),
    }


def _should_chunk(req: "CompressRequest", duration_seconds: float | None) -> bool:
    if req.chunked is not None:
        return req.chunked
    return (
        COMPRESSION_CHUNKING_ENABLED
        and duration_seconds is not None
        and duration_seconds >= COMPRESSION_CHUNK_MIN_DURATION_SECONDS
    )


class ChunkingUnavailable(RuntimeError):
    pass


def _concat_file_line(path: str) -> str:
    escaped = path.replace("'", "'\\''")
    return f"file '{escaped}'\n"


async def _split_at_keyframes(
    local_input: str,
    segments_dir: str,
    task_label: str,
    chunk_duration_seconds: int,
) -> list[str]:
    os.makedirs(segments_dir, exist_ok=True)
    segment_pattern = os.path.join(segments_dir, "input_%05d.mp4")
    cmd = [
        FFMPEG_BIN,
        "-hide_banner",
        "-y",
        "-i", local_input,
        "-map", "0:v:0",
        "-map", "0:a?",
        "-c", "copy",
        "-sn",
        "-dn",
        "-f", "segment",
        "-segment_time", str(chunk_duration_seconds),
        "-reset_timestamps", "1",
        "-segment_format", "mp4",
        segment_pattern,
    ]
    returncode, stdout, stderr, run_error = await _run_process(cmd, task_label, "split segments")
    if returncode != 0 or run_error:
        detail = run_error or _format_process_output(stdout, stderr) or "segment split failed"
        raise RuntimeError(detail)

    segments = sorted(
        os.path.join(segments_dir, filename)
        for filename in os.listdir(segments_dir)
        if filename.startswith("input_") and filename.endswith(".mp4")
    )
    if not segments:
        raise RuntimeError("segment split produced no files")
    if len(segments) < 2:
        raise ChunkingUnavailable("segment split produced fewer than two chunks")
    return segments


async def _compress_chunked(
    local_input: str,
    output_path: str,
    req: "CompressRequest",
    encoder: str,
    task_label: str,
    requested_target_vmaf: float,
    vmaf_targets: list[float],
) -> dict[str, object]:
    chunk_duration_seconds = req.chunk_duration_seconds or COMPRESSION_CHUNK_TARGET_SECONDS
    parallelism = max(1, req.chunk_parallelism or COMPRESSION_CHUNK_PARALLELISM)
    work_dir = os.path.join(SHARED_VOLUME_PATH, f"{task_label}_chunks")
    input_dir = os.path.join(work_dir, "input")
    encoded_dir = os.path.join(work_dir, "encoded")
    tracked_paths: list[str] = []

    try:
        input_segments = await _split_at_keyframes(local_input, input_dir, task_label, chunk_duration_seconds)
        await _log_chunk_seams(input_segments, task_label)

        os.makedirs(encoded_dir, exist_ok=True)
        encoded_segments = [
            os.path.join(encoded_dir, f"encoded_{index:05d}.mp4")
            for index, _ in enumerate(input_segments)
        ]
        tracked_paths = [*input_segments, *encoded_segments]
        await _track_temp_files(*tracked_paths)

        log.info(
            f"[{task_label}] Compressing {len(input_segments)} keyframe chunks with ab-av1 "
            f"(target={chunk_duration_seconds}s, parallelism={parallelism})"
        )
        semaphore = asyncio.Semaphore(parallelism)

        async def _compress_one(index: int, segment_input: str, segment_output: str) -> dict[str, object]:
            async with semaphore:
                chunk_label = f"{task_label}-chunk-{index + 1:05d}"
                segment_duration = await _probe_duration_seconds(segment_input, chunk_label)
                result = await _run_ab_av1_with_fallback(
                    chunk_label,
                    req,
                    encoder,
                    segment_input,
                    segment_output,
                    segment_duration,
                    requested_target_vmaf,
                    vmaf_targets,
                )
                if result["returncode"] != 0:
                    detail = (str(result["stderr_text"]) or str(result["stdout_text"])).strip()
                    raise RuntimeError(f"chunk {index + 1} failed: {detail}")
                if not os.path.exists(segment_output):
                    raise RuntimeError(f"chunk {index + 1} output missing: {segment_output}")
                return result

        chunk_results = await asyncio.gather(
            *[
                _compress_one(index, segment_input, encoded_segments[index])
                for index, segment_input in enumerate(input_segments)
            ],
            return_exceptions=True,
        )
        chunk_errors = [result for result in chunk_results if isinstance(result, Exception)]
        if chunk_errors:
            raise RuntimeError(str(chunk_errors[0]))

        concat_list = os.path.join(work_dir, "concat.txt")
        with open(concat_list, "w", encoding="utf-8") as file:
            for segment in encoded_segments:
                file.write(_concat_file_line(segment))
        tracked_paths.append(concat_list)
        await _track_temp_files(concat_list)

        cmd = [
            FFMPEG_BIN,
            "-hide_banner",
            "-y",
            "-f", "concat",
            "-safe", "0",
            "-i", concat_list,
            "-c", "copy",
            "-movflags", "+faststart",
            output_path,
        ]
        returncode, stdout, stderr, run_error = await _run_process(cmd, task_label, "merge chunks")
        if returncode != 0 or run_error:
            detail = run_error or _format_process_output(stdout, stderr) or "chunk merge failed"
            raise RuntimeError(detail)

        successful_results = [result for result in chunk_results if isinstance(result, dict)]
        return {
            "returncode": 0,
            "stdout_text": "",
            "stderr_text": "",
            "encode_seconds": sum(float(result["encode_seconds"]) for result in successful_results),
            "attempt_count": sum(int(result["attempt_count"]) for result in successful_results),
            "effective_target_vmaf": min(
                float(result["effective_target_vmaf"]) for result in successful_results
            ),
            "chunk_count": len(input_segments),
        }
    finally:
        await _untrack_temp_files(*tracked_paths)
        _cleanup_tree(work_dir)


def _cleanup_config_status() -> dict[str, object]:
    return {
        "enabled": CLEANUP_ENABLED,
        "interval_seconds": CLEANUP_INTERVAL_SECONDS,
        "temp_file_ttl_seconds": TEMP_FILE_TTL_SECONDS,
        "max_volume_bytes": CLEANUP_MAX_VOLUME_BYTES,
        "min_file_age_seconds": CLEANUP_MIN_FILE_AGE_SECONDS,
        "presigned_url_expiry_seconds": min(S3_PRESIGNED_EXPIRY, 604800),
        "presigned_url_cleanup_grace_seconds": PRESIGNED_URL_CLEANUP_GRACE_SECONDS,
        "storage_cleanup_enabled": STORAGE_CLEANUP_ENABLED,
        "storage_object_ttl_seconds": STORAGE_OBJECT_TTL_SECONDS,
        "storage_cleanup_prefixes": STORAGE_CLEANUP_PREFIXES,
    }


def _shared_root() -> str:
    return os.path.abspath(SHARED_VOLUME_PATH)


def _normalize_path(path: str) -> str:
    return os.path.abspath(path)


def _is_shared_path(path: str) -> bool:
    try:
        return os.path.commonpath([_shared_root(), _normalize_path(path)]) == _shared_root()
    except ValueError:
        return False


async def _track_temp_files(*paths: str):
    async with _lock:
        for path in paths:
            if path and _is_shared_path(path):
                _active_file_paths.add(_normalize_path(path))


async def _untrack_temp_files(*paths: str):
    async with _lock:
        for path in paths:
            if path:
                _active_file_paths.discard(_normalize_path(path))


async def _protected_paths_snapshot() -> set[str]:
    async with _lock:
        return set(_active_file_paths)


def _remove_stale_file(path: str, reason: str) -> int:
    try:
        size = os.path.getsize(path)
        os.remove(path)
        log.info(f"Removed {reason} temp file: {path} ({size} bytes)")
        return size
    except FileNotFoundError:
        return 0
    except OSError as e:
        log.warning(f"Failed to remove temp file {path}: {e}")
        return 0


async def _cleanup_shared_volume_once():
    if not CLEANUP_ENABLED or not os.path.isdir(SHARED_VOLUME_PATH):
        return

    now = time.time()
    protected_paths = await _protected_paths_snapshot()
    total_bytes = 0
    candidates: list[tuple[float, str, int]] = []

    for root, _, files in os.walk(SHARED_VOLUME_PATH):
        for filename in files:
            path = os.path.abspath(os.path.join(root, filename))
            try:
                stat = os.stat(path)
            except FileNotFoundError:
                continue

            total_bytes += stat.st_size
            if path in protected_paths:
                continue

            candidates.append((stat.st_mtime, path, stat.st_size))
            if now - stat.st_mtime >= TEMP_FILE_TTL_SECONDS:
                total_bytes -= _remove_stale_file(path, "expired")

    if CLEANUP_MAX_VOLUME_BYTES <= 0 or total_bytes <= CLEANUP_MAX_VOLUME_BYTES:
        return

    for _, path, size in sorted(candidates):
        if total_bytes <= CLEANUP_MAX_VOLUME_BYTES:
            break
        if path in protected_paths or not os.path.exists(path):
            continue
        try:
            file_age = now - os.path.getmtime(path)
        except FileNotFoundError:
            continue
        if file_age < CLEANUP_MIN_FILE_AGE_SECONDS:
            continue
        total_bytes -= _remove_stale_file(path, "over-quota")


def _storage_cleanup_ready() -> bool:
    return (
        STORAGE_CLEANUP_ENABLED
        and STORAGE_OBJECT_TTL_SECONDS > 0
        and bool(STORAGE_CLEANUP_PREFIXES)
        and bool(S3_BUCKET)
        and bool(S3_ACCESS_KEY_ID)
        and bool(S3_SECRET_ACCESS_KEY)
    )


def _cleanup_expired_storage_objects_once() -> int:
    if not _storage_cleanup_ready():
        return 0

    cutoff = datetime.now(timezone.utc) - timedelta(seconds=STORAGE_OBJECT_TTL_SECONDS)
    client = _get_s3_client()
    deleted = 0

    for prefix in STORAGE_CLEANUP_PREFIXES:
        paginator = client.get_paginator("list_objects_v2")
        try:
            pages = paginator.paginate(Bucket=S3_BUCKET, Prefix=prefix)
            for page in pages:
                expired_objects = []
                for obj in page.get("Contents", []):
                    last_modified = obj.get("LastModified")
                    if last_modified is None:
                        continue
                    if last_modified.tzinfo is None:
                        last_modified = last_modified.replace(tzinfo=timezone.utc)
                    if last_modified < cutoff:
                        expired_objects.append({"Key": obj["Key"]})

                for index in range(0, len(expired_objects), 1000):
                    batch = expired_objects[index : index + 1000]
                    if not batch:
                        continue
                    response = client.delete_objects(
                        Bucket=S3_BUCKET,
                        Delete={"Objects": batch, "Quiet": True},
                    )
                    deleted += len(batch) - len(response.get("Errors", []))
        except ClientError as exc:
            code = _s3_error_code(exc)
            status = _s3_status_code(exc)
            if status == 404 or code in {"404", "NoSuchBucket", "NotFound"}:
                log.info(f"Skipping {STORAGE_PROVIDER} cleanup because bucket does not exist: {S3_BUCKET}")
                return deleted
            raise

    if deleted:
        log.info(f"Deleted {deleted} expired {STORAGE_PROVIDER} object(s)")
    return deleted


async def _cleanup_worker():
    if not CLEANUP_ENABLED and not _storage_cleanup_ready():
        log.info("Cleanup worker disabled")
        return

    log.info(
        "Starting cleanup worker "
        f"(path={SHARED_VOLUME_PATH}, ttl={TEMP_FILE_TTL_SECONDS}s, "
        f"interval={CLEANUP_INTERVAL_SECONDS}s, max_bytes={CLEANUP_MAX_VOLUME_BYTES}, "
        f"storage_ttl={STORAGE_OBJECT_TTL_SECONDS}s)"
    )

    while True:
        try:
            if CLEANUP_ENABLED:
                await _cleanup_shared_volume_once()
            if _storage_cleanup_ready():
                await asyncio.to_thread(_cleanup_expired_storage_objects_once)
        except Exception as e:
            log.warning(f"Cleanup pass failed: {e}")
        await asyncio.sleep(CLEANUP_INTERVAL_SECONDS)


def _queue_capacity() -> int:
    return MAX_CONCURRENT + MAX_QUEUE_SIZE


def _queue_snapshot_locked() -> dict[str, int]:
    queued_tasks = max(0, _queue_size - _active_count)
    return {
        "max_concurrent": MAX_CONCURRENT,
        "max_queue_size": MAX_QUEUE_SIZE,
        "active_tasks": _active_count,
        "queued_tasks": queued_tasks,
        "total_pending": _queue_size,
        "queue_capacity_remaining": max(0, _queue_capacity() - _queue_size),
    }


async def _queue_snapshot() -> dict[str, int]:
    async with _lock:
        return _queue_snapshot_locked()


@asynccontextmanager
async def _queued_task(task_label: str):
    global _queue_size

    async with _lock:
        if _queue_size >= _queue_capacity():
            snapshot = _queue_snapshot_locked()
            detail = (
                f"Compression queue full: {snapshot['queued_tasks']}/{MAX_QUEUE_SIZE} queued, "
                f"{snapshot['active_tasks']}/{MAX_CONCURRENT} active"
            )
            log.warning(f"[{task_label}] {detail}")
            raise HTTPException(status_code=429, detail=detail)

        _queue_size += 1
        queue_position = max(0, _queue_size - MAX_CONCURRENT)
        snapshot = _queue_snapshot_locked()

    try:
        yield queue_position, snapshot
    finally:
        async with _lock:
            _queue_size = max(0, _queue_size - 1)


@asynccontextmanager
async def _running_task():
    global _active_count

    async with _semaphore:
        async with _lock:
            _active_count += 1
            snapshot = _queue_snapshot_locked()

        try:
            yield snapshot
        finally:
            async with _lock:
                _active_count = max(0, _active_count - 1)


class CompressRequest(BaseModel):
    video_paths: list[str] = Field(
        ...,
        min_length=1,
        max_length=5,
        description="Input video path or URL list, up to 5 items",
    )
    task_id: str = Field("", description="Task ID for logging")
    codec: str = Field("AV1", description="Target codec: AV1, H264, HEVC, VP9")
    codec_mode: str = Field("CRF", description="Rate control mode: CRF or VBR")
    cq: int = Field(35, description="Legacy constant quality value; target_vmaf controls ab-av1 quality")
    preset: str = Field("p4", description="Encoder preset")
    target_vmaf: Optional[float] = Field(None, description="Minimum VMAF target for ab-av1 auto-encode")
    sample_every: Optional[str] = Field(None, description="ab-av1 sample interval, e.g. 6m")
    max_encoded_percent: Optional[float] = Field(None, description="Optional ab-av1 max encoded size percent")
    target_bitrate: Optional[int] = Field(
        None,
        gt=0,
        description="Target video bitrate in bps (required for VBR mode)",
    )
    target_width: Optional[int] = Field(None, description="Target width for downscaling")
    target_height: Optional[int] = Field(None, description="Target height for downscaling")
    chunked: Optional[bool] = Field(None, description="Override automatic long-video chunking")
    chunk_duration_seconds: Optional[int] = Field(None, description="Target chunk duration")
    chunk_parallelism: Optional[int] = Field(None, description="Parallel chunk encodes per request")

    @field_validator("codec_mode")
    @classmethod
    def validate_codec_mode(cls, value: str) -> str:
        mode = value.strip().upper()
        if mode not in {"CRF", "VBR"}:
            raise ValueError("codec_mode must be CRF or VBR")
        return mode

    @model_validator(mode="after")
    def validate_vbr_target_bitrate(self) -> "CompressRequest":
        if self.codec_mode == "VBR" and self.target_bitrate is None:
            raise ValueError("target_bitrate is required for VBR mode")
        return self


class CompressResponse(BaseModel):
    output_paths: list[str] = Field(default_factory=list, description="Per-input local output paths")
    output_urls: list[str] = Field(default_factory=list, description="Per-input S3 presigned URLs")
    errors: list[Optional[str]] = Field(default_factory=list, description="Per-input errors")
    success: bool
    active_tasks: Optional[int] = None
    queued_tasks: Optional[int] = None


@app.get("/health")
async def health():
    snapshot = await _queue_snapshot()
    return {
        "status": "ok",
        "backend": "ab-av1",
        **snapshot,
        "storage": _storage_config_status(),
        "cleanup": _cleanup_config_status(),
        "compression": _compression_config_status(),
    }


@app.get("/queue")
async def queue_status():
    return await _queue_snapshot()


async def _compress_one(req: CompressRequest, input_video: str, task_label: str) -> CompressResponse:
    total_start = time.monotonic()
    remote_mode = _is_url(input_video)
    encoder = _encoder_for_codec(req.codec)
    direct_nvenc_cq = _uses_direct_nvenc_cq(encoder)
    download_seconds: Optional[float] = None
    upload_seconds: Optional[float] = None
    encode_seconds = 0.0
    verification_seconds = 0.0
    actual_vmaf: Optional[float] = None
    quality_retry_count = 0
    input_probe: dict[str, object] = {"duration": None, "size_bytes": None}
    requested_target_vmaf = _target_vmaf(req)
    vmaf_targets = _vmaf_fallback_targets(requested_target_vmaf)
    effective_target_vmaf = requested_target_vmaf
    attempt_count = 0
    queue_enter = time.monotonic()

    if remote_mode and DISABLE_REMOTE_IO:
        return CompressResponse(success=False, errors=["Remote URL input is disabled for this local-only service"])

    local_input = ""
    output_path = ""
    output_filename = ""
    returncode: int | None = None
    stdout_text = ""
    stderr_text = ""
    use_chunked = False

    try:
        async with _queued_task(task_label) as (queue_position, snapshot):
            log.info(
                f"[{task_label}] Queued compression "
                f"(codec={encoder}, codec_mode={req.codec_mode}, "
                f"target_bitrate={req.target_bitrate or 'not set'}, "
                f"target_vmaf={requested_target_vmaf:.2f}, "
                f"strategy={'direct-cq' if direct_nvenc_cq else 'auto-encode'}, "
                f"fallback_floor={requested_target_vmaf if direct_nvenc_cq else vmaf_targets[-1]:.2f}, "
                f"position={queue_position}, "
                f"waiting={snapshot['queued_tasks']}, remote={remote_mode})"
            )

            os.makedirs(SHARED_VOLUME_PATH, exist_ok=True)

            # --- Resolve input to local path ---
            if remote_mode:
                local_input = os.path.join(SHARED_VOLUME_PATH, f"{task_label}_input.mp4")
                await _track_temp_files(local_input)
                try:
                    download_start = time.monotonic()
                    await _download_url(input_video, local_input)
                    download_seconds = time.monotonic() - download_start
                    log.info(
                        f"[{task_label}] Download timing: elapsed={_format_seconds(download_seconds)}, "
                        f"size={_human_bytes(os.path.getsize(local_input) if os.path.exists(local_input) else None)}"
                    )
                except Exception as e:
                    _cleanup(local_input)
                    return CompressResponse(success=False, errors=[f"Failed to download input: {e}"])
            else:
                local_input = input_video
                if not os.path.exists(local_input):
                    raise HTTPException(status_code=400, detail=f"Input file not found: {local_input}")
                await _track_temp_files(local_input)

            basename = os.path.splitext(os.path.basename(local_input))[0]
            output_filename = f"{basename}_compressed.mp4"
            output_path = os.path.join(SHARED_VOLUME_PATH, output_filename)
            await _track_temp_files(output_path)

            input_probe = await _probe_input_video(task_label, local_input)
            duration_seconds = input_probe.get("duration")
            duration = float(duration_seconds) if isinstance(duration_seconds, (int, float)) else None
            use_chunked = _should_chunk(req, duration_seconds)

            async with _running_task() as running_snapshot:
                queue_wait_seconds = time.monotonic() - queue_enter
                log.info(
                    f"[{task_label}] Starting compression "
                    f"(active={running_snapshot['active_tasks']}/{MAX_CONCURRENT}, "
                    f"queued={running_snapshot['queued_tasks']}, "
                    f"queue_wait={_format_seconds(queue_wait_seconds)}, "
                    f"duration={_format_seconds(duration)}, chunked={use_chunked})"
                )

                try:
                    if use_chunked:
                        log.info(
                            f"[{task_label}] Using keyframe chunked ab-av1 compression "
                            f"(target_chunk_seconds={req.chunk_duration_seconds or COMPRESSION_CHUNK_TARGET_SECONDS}, "
                            f"parallelism={req.chunk_parallelism or COMPRESSION_CHUNK_PARALLELISM})"
                        )
                        result = await _compress_chunked(
                            local_input,
                            output_path,
                            req,
                            encoder,
                            task_label,
                            requested_target_vmaf,
                            vmaf_targets,
                        )
                    elif direct_nvenc_cq:
                        result = await _run_direct_nvenc_encode(
                            task_label,
                            req,
                            encoder,
                            local_input,
                            output_path,
                            duration,
                            requested_target_vmaf,
                        )
                    else:
                        result = await _run_ab_av1_with_fallback(
                            task_label,
                            req,
                            encoder,
                            local_input,
                            output_path,
                            duration,
                            requested_target_vmaf,
                            vmaf_targets,
                        )
                except ChunkingUnavailable as e:
                    log.warning(f"[{task_label}] Chunking unavailable, falling back to single-pass compression: {e}")
                    use_chunked = False
                    if direct_nvenc_cq:
                        result = await _run_direct_nvenc_encode(
                            task_label,
                            req,
                            encoder,
                            local_input,
                            output_path,
                            duration,
                            requested_target_vmaf,
                        )
                    else:
                        result = await _run_ab_av1_with_fallback(
                            task_label,
                            req,
                            encoder,
                            local_input,
                            output_path,
                            duration,
                            requested_target_vmaf,
                            vmaf_targets,
                        )
                except Exception as e:
                    log.error(f"[{task_label}] Compression failed before output validation: {e}")
                    result = {
                        "returncode": 1,
                        "stdout_text": "",
                        "stderr_text": str(e),
                        "encode_seconds": 0.0,
                        "attempt_count": 0,
                        "effective_target_vmaf": requested_target_vmaf,
                    }

                returncode = int(result["returncode"])
                stdout_text = str(result["stdout_text"])
                stderr_text = str(result["stderr_text"])
                encode_seconds = float(result["encode_seconds"])
                attempt_count = int(result["attempt_count"])
                effective_target_vmaf = float(result["effective_target_vmaf"])

                if (
                    returncode == 0
                    and os.path.exists(output_path)
                    and _is_nvenc_encoder(encoder)
                    and not use_chunked
                    and AB_AV1_NVENC_FULL_VMAF_VERIFY
                ):
                    verification = await _verify_and_retry_nvenc_output(
                        task_label,
                        req,
                        encoder,
                        local_input,
                        output_path,
                        duration,
                        requested_target_vmaf,
                        result.get("selected_crf"),
                    )
                    actual_vmaf_value = verification["actual_vmaf"]
                    if isinstance(actual_vmaf_value, (int, float)):
                        actual_vmaf = float(actual_vmaf_value)
                    verification_seconds = float(verification["verification_seconds"])
                    retry_encode_seconds = float(verification["retry_encode_seconds"])
                    quality_retry_count = int(verification["retry_count"])
                    encode_seconds += retry_encode_seconds
                    attempt_count += quality_retry_count
                    verification_error = verification["error"]
                    if verification_error:
                        returncode = 1
                        stderr_text = str(verification_error)

        snapshot = await _queue_snapshot()
        stats = dict(active_tasks=snapshot["active_tasks"], queued_tasks=snapshot["queued_tasks"])

        input_size = input_probe.get("size_bytes")
        output_size = os.path.getsize(output_path) if os.path.exists(output_path) else None
        duration_value = input_probe.get("duration")
        duration = float(duration_value) if isinstance(duration_value, (int, float)) else None
        speed_text = ""
        if duration and encode_seconds and encode_seconds > 0:
            speed_text = f", effective_speed={duration / encode_seconds:.2f}x"
        actual_vmaf_text = f"{actual_vmaf:.6f}" if actual_vmaf is not None else "not_measured"

        if returncode is None:
            _cleanup(output_path)
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, errors=["compression did not start"], **stats)

        if returncode != 0:
            err_msg = (stderr_text or stdout_text).strip() or "compression failed without output"
            log.error(
                f"[{task_label}] ab-av1 failed (rc={returncode}, "
                f"attempts={attempt_count}, requested_target_vmaf={requested_target_vmaf:.2f}, "
                f"effective_min_vmaf_target={effective_target_vmaf:.2f}, "
                f"encode_elapsed={_format_seconds(encode_seconds)}{speed_text}): {err_msg}"
            )
            _cleanup(output_path)
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, errors=[err_msg], **stats)

        if not os.path.exists(output_path):
            log.error(f"[{task_label}] Output file not found: {output_path}")
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, errors=["Output file not created"], **stats)

        vbr_error = await _validate_vbr_output(req, output_path, task_label)
        if vbr_error:
            log.error(f"[{task_label}] {vbr_error}")
            _cleanup(output_path)
            if remote_mode:
                _cleanup(local_input)
            return CompressResponse(success=False, errors=[vbr_error], **stats)

        # --- Remote mode: upload result to S3, return URL ---
        if remote_mode:
            try:
                s3_key = f"processing/{task_label}/{output_filename}"
                upload_start = time.monotonic()
                output_url = _upload_to_s3(output_path, s3_key)
                upload_seconds = time.monotonic() - upload_start
                total_seconds = time.monotonic() - total_start
                ratio_text = ""
                if isinstance(input_size, int) and output_size:
                    ratio_text = f", size_ratio={output_size / input_size:.3f}"
                log.info(
                    f"[{task_label}] Compression timing summary: total={_format_seconds(total_seconds)}, "
                    f"download={_format_seconds(download_seconds)}, encode={_format_seconds(encode_seconds)}, "
                    f"verify={_format_seconds(verification_seconds)}, "
                    f"upload={_format_seconds(upload_seconds)}, input_duration={_format_seconds(duration)}, "
                    f"input_size={_human_bytes(input_size if isinstance(input_size, int) else None)}, "
                    f"output_size={_human_bytes(output_size)}"
                    f"{ratio_text}{speed_text}, attempts={attempt_count}, "
                    f"requested_target_vmaf={requested_target_vmaf:.2f}, "
                    f"effective_min_vmaf_target={effective_target_vmaf:.2f}, "
                    f"actual_vmaf={actual_vmaf_text}, quality_retries={quality_retry_count}"
                )
                log.info(f"[{task_label}] Compression complete (remote): {output_url[:80]}...")
                return CompressResponse(output_urls=[output_url], errors=[None], success=True, **stats)
            except Exception as e:
                log.error(f"[{task_label}] S3 upload failed: {e}")
                return CompressResponse(success=False, errors=[f"S3 upload failed: {e}"], **stats)
            finally:
                _cleanup(local_input, output_path)

        total_seconds = time.monotonic() - total_start
        ratio_text = ""
        if isinstance(input_size, int) and output_size:
            ratio_text = f", size_ratio={output_size / input_size:.3f}"
        log.info(
            f"[{task_label}] Compression timing summary: total={_format_seconds(total_seconds)}, "
            f"download={_format_seconds(download_seconds)}, encode={_format_seconds(encode_seconds)}, "
            f"verify={_format_seconds(verification_seconds)}, "
            f"upload={_format_seconds(upload_seconds)}, input_duration={_format_seconds(duration)}, "
            f"input_size={_human_bytes(input_size if isinstance(input_size, int) else None)}, "
            f"output_size={_human_bytes(output_size)}"
            f"{ratio_text}{speed_text}, attempts={attempt_count}, "
            f"requested_target_vmaf={requested_target_vmaf:.2f}, "
            f"effective_min_vmaf_target={effective_target_vmaf:.2f}, "
            f"actual_vmaf={actual_vmaf_text}, quality_retries={quality_retry_count}"
        )
        log.info(f"[{task_label}] Compression complete (local): {output_path}")
        return CompressResponse(output_paths=[output_path], errors=[None], success=True, **stats)
    finally:
        await _untrack_temp_files(local_input, output_path)


def _combine_compress_responses(responses: list[CompressResponse]) -> CompressResponse:
    output_paths = [response.output_paths[0] if response.output_paths else "" for response in responses]
    output_urls = [response.output_urls[0] if response.output_urls else "" for response in responses]
    errors = [response.errors[0] if response.errors else None for response in responses]
    success = all(response.success for response in responses)
    latest = responses[-1] if responses else None

    return CompressResponse(
        output_paths=output_paths,
        output_urls=output_urls,
        errors=errors,
        success=success,
        active_tasks=latest.active_tasks if latest else None,
        queued_tasks=latest.queued_tasks if latest else None,
    )


@app.post("/compress", response_model=CompressResponse)
async def compress(req: CompressRequest):
    input_videos = [video_path.strip() for video_path in req.video_paths]
    if any(not video_path for video_path in input_videos):
        raise HTTPException(status_code=400, detail="video_paths entries are required")

    base_task_id = req.task_id or uuid.uuid4().hex[:8]
    responses = await asyncio.gather(
        *[
            _compress_one(
                req,
                input_video,
                base_task_id if len(input_videos) == 1 else f"{base_task_id}-{index + 1}",
            )
            for index, input_video in enumerate(input_videos)
        ]
    )

    return responses[0] if len(responses) == 1 else _combine_compress_responses(responses)
