ARG AB_AV1_COMMIT=069c00fade496a4061170c334ee2a58e7d899c2f
ARG RUST_IMAGE=rust:1.88-slim-bookworm

FROM ${RUST_IMAGE} AS ab-av1-builder

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG AB_AV1_COMMIT

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    git \
    pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY ab-av1-rust-1.88-format-args.patch /tmp/ab-av1-rust-1.88-format-args.patch
COPY ab-av1-vmaf-cuda.patch /tmp/ab-av1-vmaf-cuda.patch

RUN git clone https://github.com/alexheretic/ab-av1.git /tmp/ab-av1 \
    && git -C /tmp/ab-av1 checkout "${AB_AV1_COMMIT}" \
    && git -C /tmp/ab-av1 apply /tmp/ab-av1-rust-1.88-format-args.patch \
    && git -C /tmp/ab-av1 apply /tmp/ab-av1-vmaf-cuda.patch \
    && cargo build --manifest-path /tmp/ab-av1/Cargo.toml --release \
    && rm -rf /root/.cargo/registry /root/.cargo/git


FROM nvidia/cuda:13.0.0-devel-ubuntu22.04

SHELL ["/bin/bash", "-o", "pipefail", "-c"]

ARG FFMPEG_COMMIT=33b215d1554a14e87416a24f8e6034312e629af7
ARG VMAF_COMMIT=332dde62838d91d8b5216e9822de58851f2fd64f
ARG NV_CODEC_HEADERS_COMMIT=876af32a202d0de83bd1d36fe74ee0f7fcf86b0d
ARG DAV1D_VERSION=1.4.2
ARG SVT_AV1_VERSION=v2.3.0

ENV PATH="/vmaf:/vmaf/libvmaf/build/tools:${PATH}"
ENV PKG_CONFIG_PATH="/usr/local/lib/pkgconfig:/usr/local/lib/x86_64-linux-gnu/pkgconfig:/usr/local/lib64/pkgconfig"
ENV AB_AV1_EXECUTION_MODE=local
ENV AB_AV1_BIN=/app/ab-av1
ENV FFMPEG_BIN=ffmpeg
ENV FFPROBE_BIN=ffprobe
ENV NVIDIA_VISIBLE_DEVICES=all
ENV NVIDIA_DRIVER_CAPABILITIES=video,compute,utility

RUN mkdir -p /app /tmp/video-compressor
COPY --from=ab-av1-builder /tmp/ab-av1/target/release/ab-av1 /app/ab-av1
RUN chmod +x /app/ab-av1 && /app/ab-av1 --version

RUN apt-get update && DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends \
    build-essential \
    ca-certificates \
    clang \
    cmake \
    curl \
    doxygen \
    g++ \
    git \
    libnuma-dev \
    libopus-dev \
    libvpx-dev \
    libx264-dev \
    libx265-dev \
    make \
    nasm \
    ninja-build \
    perl \
    pkg-config \
    python-is-python3 \
    python3 \
    python3-pip \
    python3-venv \
    unzip \
    wget \
    xxd \
    xz-utils \
    && rm -rf /var/lib/apt/lists/*

RUN pip3 install --no-cache-dir meson

COPY vmaf-harmonic-mean.patch /tmp/vmaf-harmonic-mean.patch

RUN wget "https://github.com/FFmpeg/nv-codec-headers/archive/${NV_CODEC_HEADERS_COMMIT}.zip" -O /tmp/nv-codec-headers.zip \
    && unzip /tmp/nv-codec-headers.zip -d /tmp \
    && make -C "/tmp/nv-codec-headers-${NV_CODEC_HEADERS_COMMIT}" \
    && make -C "/tmp/nv-codec-headers-${NV_CODEC_HEADERS_COMMIT}" install \
    && rm -rf /tmp/nv-codec-headers*

RUN set -eux \
    && git init /vmaf \
    && git -C /vmaf remote add origin https://github.com/Netflix/vmaf.git \
    && git -C /vmaf fetch --depth 1 origin "${VMAF_COMMIT}" \
    && git -C /vmaf checkout FETCH_HEAD \
    && git -C /vmaf apply /tmp/vmaf-harmonic-mean.patch \
    && make -C /vmaf clean \
    && make -C /vmaf ENABLE_NVCC=true \
    && make -C /vmaf install \
    && for d in /usr/local/lib /usr/local/lib/x86_64-linux-gnu /usr/local/lib64; do \
            if [ -d "$d" ]; then echo "$d"; fi; \
        done > /etc/ld.so.conf.d/00-vmaf-local.conf \
    && ldconfig \
    && pkg-config --print-errors --exists "libvmaf >= 2.0.0" \
    && printf '#include <libvmaf.h>\nint main(void) { void *p = (void *)vmaf_init; return p == 0; }\n' > /tmp/check-libvmaf.c \
    && cc /tmp/check-libvmaf.c $(pkg-config --cflags --libs libvmaf) -o /tmp/check-libvmaf \
    && rm -f /tmp/check-libvmaf.c /tmp/check-libvmaf

RUN git clone --depth 1 --branch "${DAV1D_VERSION}" https://code.videolan.org/videolan/dav1d.git /tmp/dav1d \
    && meson setup /tmp/dav1d/build /tmp/dav1d --default-library=static --buildtype release \
    && ninja -C /tmp/dav1d/build \
    && ninja -C /tmp/dav1d/build install \
    && ldconfig \
    && rm -rf /tmp/dav1d

RUN git clone --depth 1 --branch "${SVT_AV1_VERSION}" https://gitlab.com/AOMediaCodec/SVT-AV1.git /tmp/SVT-AV1 \
    && cmake -S /tmp/SVT-AV1 -B /tmp/SVT-AV1/build -G "Unix Makefiles" \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=ON \
    && make -C /tmp/SVT-AV1/build -j"$(nproc)" \
    && make -C /tmp/SVT-AV1/build install \
    && ldconfig \
    && rm -rf /tmp/SVT-AV1

RUN git init /tmp/ffmpeg \
    && git -C /tmp/ffmpeg remote add origin https://github.com/FFmpeg/FFmpeg.git \
    && git -C /tmp/ffmpeg fetch --depth 1 origin "${FFMPEG_COMMIT}" \
    && git -C /tmp/ffmpeg checkout FETCH_HEAD \
    && cd /tmp/ffmpeg \
    && PKG_CONFIG_PATH="${PKG_CONFIG_PATH}" ./configure \
        --enable-gpl \
        --enable-libdav1d \
        --enable-libopus \
        --enable-libsvtav1 \
        --enable-libvmaf \
        --enable-libvpx \
        --enable-libx264 \
        --enable-libx265 \
        --enable-nonfree \
        --enable-cuda-nvcc \
        --enable-nvdec \
        --enable-nvenc \
        --extra-cflags="-I/usr/local/cuda/include" \
        --extra-ldflags="-L/usr/local/cuda/lib64" \
        --disable-static \
        --enable-shared \
        --nvccflags="-gencode arch=compute_75,code=sm_75 -gencode arch=compute_75,code=compute_75 -O2" \
    && make -j"$(nproc)" \
    && make install \
    && ldconfig \
    && rm -rf /tmp/ffmpeg

RUN ffmpeg -hide_banner -encoders > /tmp/ffmpeg-encoders.txt \
    && grep -q "av1_nvenc" /tmp/ffmpeg-encoders.txt \
    && grep -q "hevc_nvenc" /tmp/ffmpeg-encoders.txt \
    && ffmpeg -hide_banner -filters > /tmp/ffmpeg-filters.txt \
    && grep -q "libvmaf" /tmp/ffmpeg-filters.txt \
    && grep -q "libvmaf_cuda" /tmp/ffmpeg-filters.txt \
    && rm -f /tmp/ffmpeg-encoders.txt /tmp/ffmpeg-filters.txt

# modal_app.py mounts the compression worker after building this toolchain.
WORKDIR /app
