# Third-party components

The toolchain is built from these upstream projects. Consult each upstream source at the pinned revision for its license and notices; this repository does not replace those terms.

| Component | Source / version |
|---|---|
| ab-av1 | https://github.com/alexheretic/ab-av1 — `069c00fade496a4061170c334ee2a58e7d899c2f` |
| FFmpeg | https://github.com/FFmpeg/FFmpeg — `33b215d1554a14e87416a24f8e6034312e629af7` |
| libvmaf | https://github.com/Netflix/vmaf — `332dde62838d91d8b5216e9822de58851f2fd64f` |
| nv-codec-headers | https://github.com/FFmpeg/nv-codec-headers — `876af32a202d0de83bd1d36fe74ee0f7fcf86b0d` |
| dav1d | https://code.videolan.org/videolan/dav1d — `1.4.2` |
| SVT-AV1 | https://gitlab.com/AOMediaCodec/SVT-AV1 — `v2.3.0` |
| NVIDIA CUDA image | `nvidia/cuda:13.0.0-devel-ubuntu22.04` |
| Rust build image | `rust:1.88-slim-bookworm` |
| Modal SDK | `modal==1.5.0` |

The Dockerfile also installs distribution and Python packages. FFmpeg is built with `--enable-gpl` and `--enable-nonfree`; do not treat the built image as covered by a permissive project license. Build recipes fetch upstream components; binary image distribution is not part of this repository.

Local patches add CUDA VMAF integration, Rust 1.88 compatibility, and unshifted harmonic pooling over positive frame scores. Patch contents and hashes are available for review. CUDA image notices are emitted by the upstream image at startup.

The benchmark links to publicly accessible Library of Congress media. Public accessibility is not a license determination. Video bytes are downloaded by the user and are not redistributed in this repository.
