import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from modal_app import validate_request

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize('path', ['../input.mp4', '/inputs/../secret.mp4', '/other/input.mp4', '/inputs'])
def test_input_path_cannot_escape_upload_directory(path):
    with pytest.raises(ValueError):
        validate_request(path, 'run-1', 89, 600, 4)


@pytest.mark.parametrize('run_id', ['../overwrite', '/absolute', '', 'with space'])
def test_run_id_cannot_escape_output_directory(run_id):
    with pytest.raises(ValueError):
        validate_request('/inputs/input.mp4', run_id, 89, 600, 4)


def test_sampled_profile_reaches_real_command_builder():
    # Fresh process applies the actual Modal image environment before importing
    # the worker, just as the remote image does. No GPU or Modal call is made.
    subprocess.run([sys.executable, '-c', '''
import os
from modal_app import WORKER_ENV
os.environ.update(WORKER_ENV)
from compression import app as c
for codec, encoder in [('AV1', 'av1_nvenc'), ('HEVC', 'hevc_nvenc')]:
    req = c.CompressRequest(video_paths=['input.mp4'], codec=codec, target_vmaf=89, chunked=True)
    for duration in (600, 276):
        args = c._build_ab_av1_args(req, encoder, 'input.mp4', 'output.mp4', 89, duration)
        assert '--sample-duration' not in args
        assert '--thorough' not in args
        assert args[args.index('--crf-increment') + 1] == '1'
        assert args[args.index('--sample-every') + 1] == '2m'
        assert 'model=version=vmaf_v0.6.1neg' in args
        assert '--vmaf-cuda' in args
        assert 'pool=harmonic_mean' in args
        assert c._vmaf_fallback_targets(89) == [89]
'''], cwd=ROOT, check=True)


def test_default_volume_does_not_require_local_environment():
    env = dict(os.environ)
    env.pop('COMPRESSION_VOLUME', None)
    subprocess.run([sys.executable, '-c',
                    'import modal_app; assert modal_app.VOLUME_NAME == "modal-video-compressor"'],
                   cwd=ROOT, env=env, check=True)


def test_published_report_passes_quality_and_matches_ratios():
    r = json.loads((ROOT / 'benchmarks/2026-10-17/report.json').read_text())
    assert r['quality_pass'] and r['measured_full_video_vmaf_neg'] >= r['target_vmaf_neg']
    assert r['compression_factor'] == pytest.approx(r['input_bytes'] / r['output_bytes'])
    assert r['processing_plus_verification_seconds'] == pytest.approx(
        r['processing_wall_seconds'] + r['final_verification_seconds'])
