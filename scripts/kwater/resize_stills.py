#!/usr/bin/env python
"""수직 촬영 사진(JPG)을 학습 해상도에 맞게 줄인다 (기본: 3장 중 1장, 가로 1320 px).
사용: python resize_stills.py --src <flight_dir> [--src ...] --out <out_root> [--every 3] [--width 1320]
  out_root/<flight_dir 이름>/<원본 stem>.jpg 로 저장한다."""
import argparse, glob, os
from concurrent.futures import ProcessPoolExecutor
from PIL import Image


def _work(t):
    src, dst, width = t
    if os.path.exists(dst):
        return
    im = Image.open(src); im.draft('RGB', (im.width // 4, im.height // 4)); im = im.convert('RGB')
    im = im.resize((width, round(im.height * width / im.width)), Image.LANCZOS); im.save(dst, quality=92)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--every', type=int, default=3); ap.add_argument('--width', type=int, default=1320); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    jobs = []
    for d in a.src:
        name = os.path.basename(os.path.normpath(d)).replace(' ', '')
        os.makedirs(os.path.join(a.out, name), exist_ok=True)
        for j in sorted(glob.glob(os.path.join(d, '*.JPG')) + glob.glob(os.path.join(d, '*.jpg')))[::a.every]:
            jobs.append((j, os.path.join(a.out, name, os.path.splitext(os.path.basename(j))[0] + '.jpg'), a.width))
    with ProcessPoolExecutor(a.workers) as ex:
        list(ex.map(_work, jobs, chunksize=8))
    print('done:', len(jobs), 'images ->', a.out)


if __name__ == '__main__':
    main()
