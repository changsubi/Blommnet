#!/usr/bin/env python
"""가상 라벨과의 일치도 평가 (정답 라벨이 아니므로 '정확도'가 아니라 '일치도'다).
  3클래스 체크포인트 : 클래스별 IoU, mIoU, 녹조 수역 클래스 정밀도/재현율, 물 전체 IoU, 물 픽셀의 예측 분포
  12클래스(AI Hub) 체크포인트 : 가상 라벨 클래스별 예측 분포, 물 계열 클래스(배수로·연못) 의 물 IoU
사용: PYTHONPATH=<repo_root> python eval_pseudo.py --config <run>/config.resolved.yaml --ckpt <run>/best.pt \\
        --images data/frames --sam3_out data/sam3_out --sets video2 overcast_site --out eval.json [--vis_dir vis --vis_every 40] [--bench]"""
import argparse, json, os, sys, time, glob
import numpy as np, torch, torch.nn.functional as F
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from make_pseudo_labels import make_label, load_water
from bloomnet.config import load_config
from bloomnet.models.bloomnet import build_bloomnet
from bloomnet.constants import OUT_SEG, MODALITY_ORDER, FINE_CLASS_NAMES
from bloomnet.data.indices import normalize_imagenet
from bloomnet.engine.evaluator import model_inputs_from_batch

KW_NAMES = ['background', 'water', 'algae_water']
PAL3 = np.array([[0, 0, 0], [40, 90, 255], [0, 230, 60]], np.uint8)


def load_weights(model, ckpt):
    ck = torch.load(ckpt, map_location='cpu', weights_only=False)
    ema = ck.get('ema_state_dict')
    state, which = (ema['shadow'], 'ema') if isinstance(ema, dict) and 'shadow' in ema else (ck['model_state_dict'], 'model')
    own = model.state_dict()
    keep = {k: v for k, v in state.items() if k in own and tuple(own[k].shape) == tuple(v.shape)}
    model.load_state_dict(keep, strict=False)
    return {'which': which, 'loaded': len(keep), 'missing': len([k for k in own if k not in keep]), 'epoch': ck.get('epoch')}


@torch.no_grad()
def predict(model, cfg, rgb_u8, device):
    x = torch.from_numpy(np.ascontiguousarray(rgb_u8.transpose(2, 0, 1))).float() / 255.0
    x = normalize_imagenet(x)[None].to(device)
    h, w = x.shape[-2:]; ph, pw = (-h) % 32, (-w) % 32
    if ph or pw:
        x = F.pad(x, (0, pw, 0, ph))
    avail = torch.zeros(1, len(MODALITY_ORDER), device=device); avail[0, 0] = 1.0
    kw = model_inputs_from_batch({'rgb': x, 'avail': avail}, cfg)
    with torch.autocast(device.type, dtype=torch.bfloat16, enabled=device.type == 'cuda'):
        logits = model(**kw)[OUT_SEG].float()
    if logits.shape[-2:] != (h + ph, w + pw):
        logits = F.interpolate(logits, size=(h + ph, w + pw), mode='bilinear', align_corners=False)
    return logits[..., :h, :w].argmax(1)[0].to(torch.uint8).cpu().numpy()


def overlay(rgb, lab, pal, alpha=0.55):
    out = rgb.astype(np.float32).copy()
    for c in range(1, len(pal)):
        m = lab == c
        out[m] = (1 - alpha) * out[m] + alpha * pal[c]
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--images', required=True); ap.add_argument('--sam3_out', required=True)
    ap.add_argument('--sets', nargs='+', required=True, help='세트 이름의 부분 문자열')
    ap.add_argument('--thr', type=float, default=0.02); ap.add_argument('--margin', type=float, default=0.006)
    ap.add_argument('--every', type=int, default=1); ap.add_argument('--out', required=True)
    ap.add_argument('--vis_dir', default=None); ap.add_argument('--vis_every', type=int, default=40); ap.add_argument('--bench', action='store_true')
    a = ap.parse_args()
    cfg = load_config(a.config, base=None); device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = build_bloomnet(cfg); print('[ckpt]', load_weights(model, a.ckpt)); model.to(device).eval()
    K = int(cfg.data.num_classes); names = list(FINE_CLASS_NAMES) if K == len(FINE_CLASS_NAMES) else KW_NAMES
    items = []
    for s in sorted(os.listdir(a.images)):
        if not any(p in s for p in a.sets):
            continue
        for f in sorted(glob.glob(os.path.join(a.images, s, '*.jpg')) + glob.glob(os.path.join(a.images, s, '*.JPG')))[::a.every]:
            items.append((s, os.path.splitext(os.path.basename(f))[0], f))
    print('images:', len(items), flush=True)
    if a.vis_dir:
        os.makedirs(a.vis_dir, exist_ok=True)
    cm, t_inf = {}, []
    for i, (s, st, f) in enumerate(items):
        w = load_water(a.sam3_out, s, st)
        if w is None:
            continue
        rgb = np.asarray(Image.open(f).convert('RGB'))
        gt = make_label(rgb, w, a.thr, a.margin)
        t0 = time.perf_counter(); pred = predict(model, cfg, rgb, device); t_inf.append(time.perf_counter() - t0)
        m = gt != 255
        cm.setdefault(s, np.zeros((3, K), np.int64))
        np.add.at(cm[s], (gt[m].astype(np.int64), pred[m].astype(np.int64)), 1)
        if a.vis_dir and i % a.vis_every == 0:
            pal = PAL3 if K == 3 else np.array([[0, 0, 0]] + [list(np.random.RandomState(k).randint(60, 255, 3)) for k in range(1, K)], np.uint8)
            Image.fromarray(np.concatenate([rgb, overlay(rgb, gt, PAL3), overlay(rgb, pred, pal)], 1)).save(f'{a.vis_dir}/{s}_{st}.jpg', quality=88)
    cm['all'] = sum(cm.values())
    res = {'ckpt': a.ckpt, 'num_classes': K, 'names': names, 'label_names': KW_NAMES, 'thr': a.thr, 'n_images': len(items),
           'ms_per_frame_mean': 1000 * float(np.mean(t_inf[5:])) if len(t_inf) > 5 else None, 'groups': {}}
    for g, M in cm.items():
        o = {'n_label_pixels': M.sum(1).tolist()}
        if K == 3:
            inter = np.diag(M); union = M.sum(0) + M.sum(1) - inter; iou = inter / np.maximum(union, 1)
            o['iou'] = dict(zip(KW_NAMES, iou.round(4).tolist())); o['miou'] = float(iou[union > 0].mean())
            o['algae_precision'] = float(M[2, 2] / max(M[:, 2].sum(), 1)); o['algae_recall'] = float(M[2, 2] / max(M[2].sum(), 1))
            wsum = M[1:, :].sum(); o['pred_share_of_water'] = {KW_NAMES[j]: round(float(M[1:, j].sum() / max(wsum, 1)), 4) for j in range(3)}
            o['water_any_iou'] = float(M[1:, 1:].sum() / max(M[1:, :].sum() + M[:, 1:].sum() - M[1:, 1:].sum(), 1))
        else:
            share = M / np.maximum(M.sum(1, keepdims=True), 1)
            o['pred_share_by_label'] = {KW_NAMES[k]: {names[j]: round(float(share[k, j]), 4) for j in np.argsort(-share[k])[:5]} for k in range(3)}
            wl = [3, 11]; gt_w = M[1:, :].sum(); pred_w = M[:, wl].sum(); tp = M[1:, wl].sum()
            o['waterlike_vs_water'] = {'classes': [names[c] for c in wl], 'precision': float(tp / max(pred_w, 1)), 'recall': float(tp / max(gt_w, 1)), 'iou': float(tp / max(gt_w + pred_w - tp, 1))}
        res['groups'][g] = o
    if a.bench and device.type == 'cuda':
        x = np.zeros((1088, 1920, 3), np.uint8)
        for _ in range(5): predict(model, cfg, x, device)
        torch.cuda.synchronize(); t0 = time.perf_counter()
        for _ in range(30): predict(model, cfg, x, device)
        torch.cuda.synchronize(); res['bench_ms_1920x1088_bf16'] = 1000 * (time.perf_counter() - t0) / 30
    json.dump(res, open(a.out, 'w'), indent=1, ensure_ascii=False)
    for g, o in res['groups'].items():
        print(g, json.dumps(o, ensure_ascii=False))


if __name__ == '__main__':
    main()
