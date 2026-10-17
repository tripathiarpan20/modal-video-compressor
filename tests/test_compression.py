import asyncio

import pytest
from pydantic import ValidationError

from compression import app as compression


def _encoder_options(args: list[str]) -> list[str]:
    return [args[index + 1] for index, value in enumerate(args[:-1]) if value == "--enc"]


def _option_value(args: list[str], option: str) -> str:
    return args[args.index(option) + 1]


@pytest.mark.parametrize(
    "target_vmaf,expected_min_crf",
    [(85, "28"), (89, "22"), (93, "16")],
)
def test_nvenc_uses_quarter_step_targeted_crf_range(target_vmaf, expected_min_crf):
    req = compression.CompressRequest(video_paths=["input.mp4"], target_vmaf=target_vmaf)

    args = compression._build_ab_av1_args(
        req,
        "av1_nvenc",
        "input.mp4",
        "output.mp4",
        target_vmaf,
        60,
    )

    assert _option_value(args, "--min-crf") == expected_min_crf
    assert _option_value(args, "--max-crf") == "51"
    assert _option_value(args, "--crf-increment") == "0.25"


@pytest.mark.parametrize("codec,encoder", [("AV1", "av1_nvenc"), ("HEVC", "hevc_nvenc")])
def test_nvenc_auto_search_uses_one_full_video_probe_per_crf(codec, encoder):
    req = compression.CompressRequest(video_paths=["input.mp4"], codec=codec)

    args = compression._build_ab_av1_args(
        req,
        encoder,
        "input.mp4",
        "output.mp4",
        85,
        30,
    )

    assert _option_value(args, "--sample-duration") == "0s"
    assert "--sample-every" not in args


@pytest.mark.parametrize(
    "encoder,target_vmaf,expected_crf",
    [
        ("av1_nvenc", 85, 47.75),
        ("av1_nvenc", 89, 42.5),
        ("av1_nvenc", 93, 36.75),
        ("hevc_nvenc", 85, 46.25),
        ("hevc_nvenc", 89, 35.5),
        ("hevc_nvenc", 93, 30.0),
    ],
)
def test_direct_nvenc_uses_configurable_target_anchor(encoder, target_vmaf, expected_crf):
    assert compression._nvenc_initial_crf(encoder, target_vmaf) == expected_crf


def test_direct_nvenc_interpolates_between_target_anchors():
    assert compression._nvenc_initial_crf("av1_nvenc", 87) == 45.0


def test_direct_cq_defaults_to_nvenc_encoders_and_requires_verification(monkeypatch):
    assert compression._uses_direct_nvenc_cq("av1_nvenc") is True
    assert compression._uses_direct_nvenc_cq("hevc_nvenc") is True

    monkeypatch.setattr(compression, "AB_AV1_NVENC_FULL_VMAF_VERIFY", False)
    assert compression._uses_direct_nvenc_cq("av1_nvenc") is False
    assert compression._uses_direct_nvenc_cq("hevc_nvenc") is False


def test_adaptive_retry_uses_conservative_default_vmaf_slope():
    assert compression.AB_AV1_NVENC_VMAF_PER_CRF == pytest.approx(1.5)
    assert compression.AB_AV1_HEVC_NVENC_VMAF_PER_CRF == pytest.approx(0.18)


def test_hevc_adaptive_retry_uses_calibrated_local_vmaf_slope():
    assert compression._adaptive_retry_crf("hevc_nvenc", 30, 92.64, 93) == 28


def test_fixed_cq_retry_keeps_nvenc_quality_and_vbr_options():
    req = compression.CompressRequest(
        video_paths=["input.mp4"],
        codec="AV1",
        codec_mode="VBR",
        target_bitrate=8_000_000,
    )

    args = compression._build_ab_av1_encode_args(
        req,
        "av1_nvenc",
        "input.mp4",
        "output.mp4",
        47.5,
    )

    assert args[0] == "encode"
    assert _option_value(args, "--crf") == "47.5"
    assert "--min-vmaf" not in args
    assert "maxrate=8800000" in _encoder_options(args)
    assert "temporal-aq=1" in _encoder_options(args)


def test_hevc_direct_cq_uses_fast_quality_profile():
    req = compression.CompressRequest(
        video_paths=["input.mp4"],
        codec="HEVC",
        codec_mode="VBR",
        target_bitrate=10_000_000,
        target_vmaf=93,
    )

    args = compression._build_ab_av1_encode_args(
        req,
        "hevc_nvenc",
        "input.mp4",
        "output.mp4",
        compression._nvenc_initial_crf("hevc_nvenc", 93),
    )
    options = _encoder_options(args)

    assert _option_value(args, "--preset") == "p4"
    assert _option_value(args, "--crf") == "30"
    assert "tune=hq" in options
    assert "multipass=qres" in options
    assert "tune=uhq" not in options
    assert "multipass=fullres" not in options


def test_parses_selected_crf_and_plain_vmaf_score():
    output = "crf 47.5 VMAF 85.42\ncrf 47.75 VMAF 85.21\ncrf 48 VMAF 84.91"

    assert compression._parse_selected_crf("", output, 85.0) == 47.75
    assert compression._parse_vmaf_score("85.067966\n", "") == pytest.approx(85.067966)


def test_direct_nvenc_encode_starts_at_learned_cq(monkeypatch):
    req = compression.CompressRequest(
        video_paths=["input.mp4"],
        codec="AV1",
        target_vmaf=85,
    )
    commands = []

    async def run_attempt(_task_label, cmd, _output_path, _input_duration):
        commands.append(cmd)
        return 0, "", "", 2.0

    monkeypatch.setattr(compression, "_run_ab_av1_attempt", run_attempt)
    result = asyncio.run(
        compression._run_direct_nvenc_encode(
            "test",
            req,
            "av1_nvenc",
            "input.mp4",
            "output.mp4",
            30,
            85,
        )
    )

    assert result["returncode"] == 0
    assert result["selected_crf"] == 47.75
    assert len(commands) == 1
    assert commands[0][1] == "encode"
    assert _option_value(commands[0], "--crf") == "47.75"


def test_full_vmaf_miss_retries_only_one_quarter_step(monkeypatch, tmp_path):
    req = compression.CompressRequest(
        video_paths=["input.mp4"],
        codec="AV1",
        codec_mode="VBR",
        target_bitrate=8_000_000,
        target_vmaf=85,
    )
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"first encode")
    scores = iter([84.99, 85.10])
    commands = []

    async def measure(*_args):
        return next(scores), 1.0, None

    async def run_attempt(_task_label, cmd, retry_output_path, _input_duration):
        commands.append(cmd)
        with open(retry_output_path, "wb") as output:
            output.write(b"retry encode")
        return 0, "", "", 2.0

    monkeypatch.setattr(compression, "_measure_full_vmaf", measure)
    monkeypatch.setattr(compression, "_run_ab_av1_attempt", run_attempt)

    async def probe_bitrate(*_args):
        return 1_000_000

    monkeypatch.setattr(compression, "_probe_video_bitrate_bps", probe_bitrate)
    result = asyncio.run(
        compression._verify_and_retry_nvenc_output(
            "test",
            req,
            "av1_nvenc",
            "input.mp4",
            str(output_path),
            30,
            85,
            47.75,
        )
    )

    assert result["error"] is None
    assert result["actual_vmaf"] == pytest.approx(85.10)
    assert result["verification_seconds"] == 2.0
    assert result["retry_encode_seconds"] == 2.0
    assert result["retry_count"] == 1
    assert len(commands) == 1
    assert _option_value(commands[0], "--crf") == "47.5"


def test_hevc_vmaf_miss_at_rate_cap_stops_without_retry(monkeypatch, tmp_path):
    req = compression.CompressRequest(
        video_paths=["input.mp4"],
        codec="HEVC",
        codec_mode="VBR",
        target_bitrate=10_000_000,
        target_vmaf=93,
    )
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"rate capped encode")

    async def measure(*_args):
        return 91.47, 1.0, None

    async def probe_bitrate(*_args):
        return 10_540_000

    async def unexpected_retry(*_args):
        raise AssertionError("a saturated VBR encode must not retry at another CQ")

    monkeypatch.setattr(compression, "_measure_full_vmaf", measure)
    monkeypatch.setattr(compression, "_probe_video_bitrate_bps", probe_bitrate)
    monkeypatch.setattr(compression, "_run_ab_av1_attempt", unexpected_retry)
    result = asyncio.run(
        compression._verify_and_retry_nvenc_output(
            "test",
            req,
            "hevc_nvenc",
            "input.mp4",
            str(output_path),
            30,
            93,
            23,
        )
    )

    assert "futile CRF search was stopped" in result["error"]
    assert result["actual_vmaf"] == pytest.approx(91.47)
    assert result["retry_count"] == 0


def test_small_verified_overshoot_does_not_retry(monkeypatch, tmp_path):
    req = compression.CompressRequest(video_paths=["input.mp4"], target_vmaf=85)
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"verified encode")

    async def measure(*_args):
        return 85.06, 1.0, None

    async def unexpected_retry(*_args):
        raise AssertionError("a 0.06 VMAF overshoot should not trigger another encode")

    monkeypatch.setattr(compression, "_measure_full_vmaf", measure)
    monkeypatch.setattr(compression, "_run_ab_av1_attempt", unexpected_retry)
    result = asyncio.run(
        compression._verify_and_retry_nvenc_output(
            "test",
            req,
            "av1_nvenc",
            "input.mp4",
            str(output_path),
            30,
            85,
            47.75,
        )
    )

    assert result["error"] is None
    assert result["actual_vmaf"] == pytest.approx(85.06)
    assert result["retry_count"] == 0


def test_compression_retry_undershoot_retains_verified_first_encode(monkeypatch, tmp_path):
    req = compression.CompressRequest(video_paths=["input.mp4"], target_vmaf=85)
    output_path = tmp_path / "output.mp4"
    output_path.write_bytes(b"verified first encode")
    scores = iter([85.86, 84.90])
    commands = []

    async def measure(*_args):
        return next(scores), 1.0, None

    async def run_attempt(_task_label, cmd, retry_output_path, _input_duration):
        commands.append(cmd)
        with open(retry_output_path, "wb") as output:
            output.write(b"undershooting retry")
        return 0, "", "", 2.0

    monkeypatch.setattr(compression, "_measure_full_vmaf", measure)
    monkeypatch.setattr(compression, "_run_ab_av1_attempt", run_attempt)
    result = asyncio.run(
        compression._verify_and_retry_nvenc_output(
            "test",
            req,
            "av1_nvenc",
            "input.mp4",
            str(output_path),
            30,
            85,
            47.75,
        )
    )

    assert result["error"] is None
    assert result["actual_vmaf"] == pytest.approx(85.86)
    assert result["retry_count"] == 1
    assert _option_value(commands[0], "--crf") == "48.25"
    assert output_path.read_bytes() == b"verified first encode"


def test_nvenc_full_range_retry_restores_configured_minimum():
    req = compression.CompressRequest(video_paths=["input.mp4"], target_vmaf=93)

    args = compression._build_ab_av1_args(
        req,
        "av1_nvenc",
        "input.mp4",
        "output.mp4",
        93,
        60,
        full_crf_range=True,
    )

    assert _option_value(args, "--min-crf") == "1"
    assert _option_value(args, "--max-crf") == "51"
    assert _option_value(args, "--crf-increment") == "0.25"


def test_targeted_range_failure_retries_same_vmaf_with_full_range(monkeypatch):
    req = compression.CompressRequest(video_paths=["input.mp4"], target_vmaf=93)
    commands = []

    async def run_attempt(_task_label, cmd, _output_path, _input_duration):
        commands.append(cmd)
        if len(commands) == 1:
            return 1, "", "Failed to find a suitable crf", 1.0
        return 0, "", "", 1.0

    monkeypatch.setattr(compression, "_run_ab_av1_attempt", run_attempt)
    result = asyncio.run(
        compression._run_ab_av1_with_fallback(
            "test",
            req,
            "av1_nvenc",
            "input.mp4",
            "output.mp4",
            60,
            93,
            [93, 91],
        )
    )

    assert result["returncode"] == 0
    assert result["attempt_count"] == 2
    assert result["effective_target_vmaf"] == 93
    assert _option_value(commands[0], "--min-crf") == "16"
    assert _option_value(commands[1], "--min-crf") == "1"


@pytest.mark.parametrize("codec,encoder", [("AV1", "av1_nvenc"), ("HEVC", "hevc_nvenc")])
def test_nvenc_defaults_to_p4(codec, encoder):
    req = compression.CompressRequest(video_paths=["input.mp4"], codec=codec)

    args = compression._build_ab_av1_args(
        req,
        encoder,
        "input.mp4",
        "output.mp4",
        85,
        60,
    )

    assert _option_value(args, "--preset") == "p4"


def test_explicit_hevc_preset_overrides_default():
    req = compression.CompressRequest(
        video_paths=["input.mp4"],
        codec="HEVC",
        preset="p5",
    )

    args = compression._build_ab_av1_args(
        req,
        "hevc_nvenc",
        "input.mp4",
        "output.mp4",
        85,
        60,
    )

    assert _option_value(args, "--preset") == "p5"


def test_vbr_request_requires_target_bitrate():
    with pytest.raises(ValidationError, match="target_bitrate is required for VBR mode"):
        compression.CompressRequest(video_paths=["input.mp4"], codec_mode="vbr")


@pytest.mark.parametrize("target_mbps", [5, 8, 10])
def test_av1_vbr_adds_bframe_options_and_configured_bitrate_ceiling(target_mbps):
    target_bitrate = target_mbps * 1_000_000
    req = compression.CompressRequest(
        video_paths=["input.mp4"],
        codec="AV1",
        codec_mode="vbr",
        target_bitrate=target_bitrate,
        target_vmaf=85,
    )

    args = compression._build_ab_av1_args(
        req,
        "av1_nvenc",
        "input.mp4",
        "output.mp4",
        85,
        60,
    )

    assert req.codec_mode == "VBR"
    allowed_bitrate = int(target_bitrate * 1.10)
    assert _encoder_options(args) == [
        "rc=vbr",
        "b:v=0",
        "tune=uhq",
        "multipass=fullres",
        "rc-lookahead=24",
        "lookahead_level=3",
        "g=900",
        "tile-columns=1",
        "tile-rows=1",
        "qp_cb_offset=2",
        "qp_cr_offset=2",
        "bf=5",
        "b_ref_mode=middle",
        "temporal-aq=1",
        f"maxrate={allowed_bitrate}",
    ]


def test_hevc_vbr_keeps_crf_quality_options_and_adds_ceiling():
    req = compression.CompressRequest(
        video_paths=["input.mp4"],
        codec="HEVC",
        codec_mode="VBR",
        target_bitrate=5_000_000,
    )

    args = compression._build_ab_av1_args(
        req,
        "hevc_nvenc",
        "input.mp4",
        "output.mp4",
        85,
        60,
    )
    options = _encoder_options(args)

    assert options == [
        "rc=vbr",
        "b:v=0",
        "tune=hq",
        "multipass=qres",
        "rc-lookahead=24",
        "lookahead_level=3",
        "g=900",
        "qp_cb_offset=2",
        "qp_cr_offset=2",
        "bf=5",
        "b_ref_mode=middle",
        "temporal-aq=1",
        "maxrate=5500000",
    ]


def test_av1_crf_command_keeps_bframe_options_without_vbr_bitrate_cap():
    req = compression.CompressRequest(
        video_paths=["input.mp4"],
        codec="AV1",
        codec_mode="CRF",
    )

    args = compression._build_ab_av1_args(
        req,
        "av1_nvenc",
        "input.mp4",
        "output.mp4",
        85,
        60,
    )

    assert _encoder_options(args) == [
        "rc=vbr",
        "b:v=0",
        "tune=uhq",
        "multipass=fullres",
        "rc-lookahead=24",
        "lookahead_level=3",
        "g=900",
        "tile-columns=1",
        "tile-rows=1",
        "qp_cb_offset=2",
        "qp_cr_offset=2",
        "bf=5",
        "b_ref_mode=middle",
        "temporal-aq=1",
    ]


@pytest.mark.parametrize("target_mbps", [5, 8, 10])
def test_vbr_output_uses_configured_tolerance(monkeypatch, target_mbps):
    target_bitrate = target_mbps * 1_000_000
    allowed_bitrate = int(target_bitrate * 1.10)
    req = compression.CompressRequest(
        video_paths=["input.mp4"],
        codec_mode="VBR",
        target_bitrate=target_bitrate,
    )

    async def probe_pass(*_args):
        return allowed_bitrate

    monkeypatch.setattr(compression, "_probe_video_bitrate_bps", probe_pass)
    assert asyncio.run(compression._validate_vbr_output(req, "output.mp4", "test")) is None

    async def probe_fail(*_args):
        return allowed_bitrate + 1

    monkeypatch.setattr(compression, "_probe_video_bitrate_bps", probe_fail)
    error = asyncio.run(compression._validate_vbr_output(req, "output.mp4", "test"))
    assert error is not None
    assert f"maximum={target_mbps * 1.10:.2f} Mbps" in error
