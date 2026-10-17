"""Summarize a saved report without Modal credentials or third-party dependencies."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('report', type=Path)
    parser.add_argument('--baseline', type=Path)
    args = parser.parse_args()
    r = json.loads(args.report.read_text())
    seconds = r['processing_wall_seconds']
    verified = seconds + r['final_verification_seconds']
    print(f"Processing: {seconds:.2f}s ({r['input_duration_seconds'] / seconds:.2f}x realtime)")
    print(f"Processing + verification: {verified:.2f}s")
    print(f"Compression: {r['input_bytes'] / r['output_bytes']:.2f}:1")
    print(f"Size reduction: {100 * (1 - r['output_bytes'] / r['input_bytes']):.2f}%")
    print(f"Full-video VMAF NEG: {r['measured_full_video_vmaf_neg']}")
    print(f"Quality passed: {r['quality_pass']}")
    if args.baseline:
        b = json.loads(args.baseline.read_text())
        print(f"Processing speedup over baseline: {b['processing_wall_seconds'] / seconds:.2f}x")
        if not r['quality_pass']:
            print('Quality failed: speedup is not a passing-quality comparison.')


if __name__ == '__main__':
    main()
