#!/bin/bash
# 동영상에서 1초당 1프레임을 1920x1080 JPEG 로 추출한다.
# 사용: ./extract_frames.sh <video.mp4> <out_dir> [fps]
# ffmpeg 는 imageio-ffmpeg 가 번들한 바이너리를 쓴다 (pip install imageio-ffmpeg).
set -e
if [ -z "$2" ]; then echo "usage: $0 <video.mp4> <out_dir> [fps]"; exit 1; fi
VIDEO="$1"; OUT="$2"; FPS="${3:-1}"
FF=$(${PYTHON:-python} -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())")
mkdir -p "$OUT"
"$FF" -y -loglevel error -i "$VIDEO" -vf "fps=${FPS},scale=1920:1080" -q:v 2 "$OUT/f%05d.jpg"
echo "extracted: $(ls "$OUT" | wc -l) frames -> $OUT"
