#!/usr/bin/env python
"""가상 라벨 데이터셋 생성 (사람 라벨 없음).
  물 영역   : SAM3 "water" 마스크 (sam3_masks.py 결과)
  녹조 수역 : 물 영역의 프레임 평균 녹색도 GI = G/(R+G+B) - B/(R+G+B) 가 임계(thr) 이상인 프레임의 물
  라벨 값   : 0 배경, 1 물, 2 녹조 수역(가상 라벨 클래스), 255 무시(임계 근처 프레임의 물, 물가 띠, 작은 성분)
출력 레이아웃은 bloomnet 의 aihub092 데이터셋 규약을 따른다:
  <out>/images/<split>/<scene>/<stem>.png (원본 JPEG 로의 심볼릭 링크), <out>/labels/<split>/<scene>/<stem>_labelids.png
사용 예:
  python make_pseudo_labels.py --images data/frames --sam3_out data/sam3_out --out data/pseudo \\
      --train video1 clear_m3m overcast_dam --val clear_m3m_alt150 --test video2 overcast_site \\
      --video_head video1=0.85 --thr 0.02
  --train/--val/--test 는 세트(입력 폴더 이름)의 부분 문자열 패턴이다. --video_head SET=0.85 는 그 세트의 앞 85% 프레임을 train,
  나머지를 val 로 나눈다. val 은 512² 조각(--val_crops)으로 저장하고 test 는 목록(test_list.json)만 만든다."""
import argparse, glob, json, os, re
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from PIL import Image
from scipy import ndimage as ndi

MIN_COMP, BAND = 400, 3


def greenness(rgb):
    x = rgb.astype(np.float32) + 1.0
    s = x.sum(-1)
    return x[..., 1] / s - x[..., 2] / s


def make_label(rgb, water, thr=0.02, margin=0.006, mode='frame'):
    lab = np.zeros(water.shape, np.uint8)
    comp, n = ndi.label(water)
    gi = greenness(rgb)
    stats = []
    for i in range(1, n + 1):
        m = comp == i
        if m.sum() < MIN_COMP:
            lab[m] = 255
            continue
        me = ndi.binary_erosion(m, iterations=BAND)
        if me.sum() < 50:
            me = m
        stats.append((m, float(gi[me].mean()), int(me.sum()), float(gi[me].sum())))
    if mode == 'frame' and stats:
        g = sum(x[3] for x in stats) / max(1, sum(x[2] for x in stats))
        for m, _, _, _ in stats:
            lab[m] = 255 if abs(g - thr) < margin else (2 if g >= thr else 1)
    else:
        for m, g, _, _ in stats:
            lab[m] = 255 if abs(g - thr) < margin else (2 if g >= thr else 1)
    band = ndi.binary_dilation(water, iterations=BAND) & ~ndi.binary_erosion(water, iterations=BAND)
    lab[band] = 255
    return lab


def load_water(sam3_out, set_name, stem):
    p = os.path.join(sam3_out, 'masks', set_name, f'{stem}_water.png')
    return (np.asarray(Image.open(p)) > 127) if os.path.exists(p) else None


def _process(job):
    s, st, sp, adm, img, out, thr, margin, crops = job
    w = load_water(job_ctx['sam3_out'], s, st)
    if w is None:
        return None
    rgb = np.asarray(Image.open(img).convert('RGB'))
    if rgb.shape[:2] != w.shape:
        w = np.asarray(Image.fromarray(w.astype(np.uint8) * 255).resize((rgb.shape[1], rgb.shape[0]), Image.NEAREST)) > 127
    lab = make_label(rgb, w, thr, margin)
    px = np.array([(lab == c).sum() for c in range(3)], np.int64)
    m = re.search(r'(\d{8})', s + st); date = m.group(1) if m else '20000101'
    seq = int(re.findall(r'\d+', st)[-1]) if re.findall(r'\d+', st) else 0
    scene = f'set{adm:02d}_' + re.sub(r'[^A-Za-z0-9_]+', '', s)[:24]
    if sp == 'test':
        return (sp, px, {'set': s, 'stem': st, 'image': img})
    if sp == 'train':
        stem = f'L09_{adm:05d}_0_{date}_N01_{seq:05d}'
        os.makedirs(f'{out}/images/train/{scene}', exist_ok=True); os.makedirs(f'{out}/labels/train/{scene}', exist_ok=True)
        lnk = f'{out}/images/train/{scene}/{stem}.png'
        if not os.path.lexists(lnk):
            os.symlink(os.path.abspath(img), lnk)   # JPEG 를 .png 이름으로 링크 (PIL 은 내용으로 포맷을 판별한다)
        Image.fromarray(lab).save(f'{out}/labels/train/{scene}/{stem}_labelids.png')
    else:
        H, W = lab.shape
        boxes = [(W // 4 - 256, H // 2 - 256), (3 * W // 4 - 256, H // 2 - 256)][:crops] if W >= 1600 else [(W // 2 - 256, H // 2 - 256)]
        os.makedirs(f'{out}/images/val/{scene}', exist_ok=True); os.makedirs(f'{out}/labels/val/{scene}', exist_ok=True)
        for bi, (x0, y0) in enumerate(boxes):
            x0 = max(0, min(x0, W - 512)); y0 = max(0, min(y0, H - 512))
            st2 = f'L09_{adm:05d}_0_{date}_N0{bi + 1}_{seq:05d}'
            Image.fromarray(rgb[y0:y0 + 512, x0:x0 + 512]).save(f'{out}/images/val/{scene}/{st2}.png')
            Image.fromarray(lab[y0:y0 + 512, x0:x0 + 512]).save(f'{out}/labels/val/{scene}/{st2}_labelids.png')
    return (sp, px, None)


job_ctx = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--images', required=True, help='<set>/<stem>.jpg 구조의 루트')
    ap.add_argument('--sam3_out', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--train', nargs='*', default=[]); ap.add_argument('--val', nargs='*', default=[]); ap.add_argument('--test', nargs='*', default=[])
    ap.add_argument('--exclude', nargs='*', default=[]); ap.add_argument('--video_head', nargs='*', default=[], help='SET=0.85: 앞 85%% train, 나머지 val')
    ap.add_argument('--thr', type=float, default=0.02); ap.add_argument('--margin', type=float, default=0.006)
    ap.add_argument('--val_crops', type=int, default=2); ap.add_argument('--workers', type=int, default=8)
    a = ap.parse_args()
    job_ctx['sam3_out'] = a.sam3_out
    sets = sorted(d for d in os.listdir(a.images) if os.path.isdir(os.path.join(a.images, d)) and not any(x in d for x in a.exclude))
    admin = {s: i + 1 for i, s in enumerate(sets)}
    head = {k: float(v) for k, v in (x.split('=') for x in a.video_head)}
    jobs = []
    for s in sets:
        files = sorted(glob.glob(os.path.join(a.images, s, '*.jpg')) + glob.glob(os.path.join(a.images, s, '*.JPG')))
        hk = next((k for k in head if k in s), None)
        for i, f in enumerate(files):
            st = os.path.splitext(os.path.basename(f))[0]
            if hk is not None:
                sp = 'train' if i < int(len(files) * head[hk]) else 'val'
            elif any(p in s for p in a.test):
                sp = 'test'
            elif any(p in s for p in a.val):
                sp = 'val'
            elif any(p in s for p in a.train):
                sp = 'train'
            else:
                continue
            jobs.append((s, st, sp, admin[s], f, a.out, a.thr, a.margin, a.val_crops))
    for d in ('images', 'labels'):
        for sp in ('train', 'val'):
            os.makedirs(f'{a.out}/{d}/{sp}', exist_ok=True)
    test_list, counts, px = [], {}, np.zeros(3, np.int64)
    with ProcessPoolExecutor(a.workers) as ex:
        for r in ex.map(_process, jobs, chunksize=4):
            if r is None:
                continue
            sp, p, t = r; px += p; counts[sp] = counts.get(sp, 0) + 1
            if t is not None:
                test_list.append(t)
    json.dump({'thr': a.thr, 'margin': a.margin, 'min_comp': MIN_COMP, 'band': BAND, 'counts': counts, 'test_n': len(test_list),
               'pixel_share': {'background': float(px[0] / px.sum()), 'water': float(px[1] / px.sum()), 'algae_water': float(px[2] / px.sum())}},
              open(f'{a.out}/meta.json', 'w'), ensure_ascii=False, indent=1)
    json.dump(test_list, open(f'{a.out}/test_list.json', 'w'), ensure_ascii=False, indent=0)
    print('built:', counts, 'test', len(test_list), 'pixel share bg/water/algae = %.3f/%.3f/%.3f' % tuple(px / px.sum()))


if __name__ == '__main__':
    main()
