#!/usr/bin/env bash
# Reproduce the published workload. Requires an authenticated Modal CLI.
set -euo pipefail
cd "$(dirname "$0")/.."
command -v modal >/dev/null
command -v curl >/dev/null

SOURCE_URL='https://tile.loc.gov/storage-services/service/mbrs/ntscrm/00021068/00021068.mp4'
EXPECTED_SHA256='ea51543b2c5a36391ea6b4dd6347655f3f9ecd8820ed3bc047092a7823ff1d6d'
RUN_ID="${RUN_ID:-$(date -u +%Y%m%dT%H%M%SZ)-$(python3 -c 'import secrets; print(secrets.token_hex(3))')}"
export COMPRESSION_VOLUME="${COMPRESSION_VOLUME:-modal-video-compressor-$RUN_ID}"
mkdir -p data "results/$RUN_ID"
printf '%s\n' "$COMPRESSION_VOLUME" > "results/$RUN_ID/volume-name.txt"
printf '%s\n' "$RUN_ID" > results/latest-run-id.txt

curl --fail --location --retry 5 --continue-at - "$SOURCE_URL" --output data/input.mp4
python3 - "$EXPECTED_SHA256" <<'PY'
import hashlib, sys
from pathlib import Path
h = hashlib.sha256()
with Path('data/input.mp4').open('rb') as f:
    for block in iter(lambda: f.read(8 * 1024 * 1024), b''):
        h.update(block)
if h.hexdigest() != sys.argv[1]:
    raise SystemExit('Input SHA-256 mismatch: do not compare this file with the published benchmark.')
print('Input checksum matches the published benchmark.')
PY

# Resolve/create lazily without needing a separate create command on reruns.
python3 - <<'PY'
import os, modal
volume = modal.Volume.from_name(os.environ['COMPRESSION_VOLUME'], create_if_missing=True)
with volume.batch_upload(force=True) as batch:
    batch.put_file('data/input.mp4', '/inputs/input.mp4')
PY

modal run modal_app.py \
  --target 89 --codec AV1 --chunk-seconds 600 --parallelism 4 \
  --input-path /inputs/input.mp4 --run-id "$RUN_ID" \
  --source-url "$SOURCE_URL" --expected-sha256 "$EXPECTED_SHA256" \
  2>&1 | tee "results/$RUN_ID/run.log"

modal volume get "$COMPRESSION_VOLUME" "/runs/$RUN_ID/input_compressed.mp4" "results/$RUN_ID/output.mp4"
python3 scripts/summarize.py "results/$RUN_ID/report.json" --baseline benchmarks/2026-10-17/report.json
printf '\nSaved results in results/%s\n' "$RUN_ID"
printf 'After checking the download, remove remote data with: modal volume delete %s\n' "$COMPRESSION_VOLUME"
