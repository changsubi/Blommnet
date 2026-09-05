#!/usr/bin/env python
"""SAM3 zero-shot 마스크 생성. 입력 폴더의 모든 JPG 에 대해 프롬프트별 마스크(0/255 PNG)와 반응 기록(JSONL)을 남긴다.
사용: python sam3_masks.py --inputs <dir1> [<dir2> ...] --out <out_dir> [--model facebook/sam3] [--prompts water "green water" algae "algal bloom"]
  결과: out_dir/masks/<입력 폴더 이름>/<stem>_<prompt>.png, out_dir/records.jsonl
  SAM3 가중치는 Hugging Face 'facebook/sam3'(승인 필요) 또는 로컬 경로. transformers>=5 필요."""
import argparse, glob, json, os, time
import numpy as np, torch
from PIL import Image
from transformers import Sam3Model, Sam3Processor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', nargs='+', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--model', default='facebook/sam3'); ap.add_argument('--prompts', nargs='+', default=['water', 'green water', 'algae', 'algal bloom'])
    ap.add_argument('--threshold', type=float, default=0.3); ap.add_argument('--max_side', type=int, default=1920)
    a = ap.parse_args()
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = Sam3Model.from_pretrained(a.model, torch_dtype=torch.bfloat16 if dev == 'cuda' else torch.float32).to(dev).eval()
    proc = Sam3Processor.from_pretrained(a.model)
    os.makedirs(a.out, exist_ok=True); rec = open(os.path.join(a.out, 'records.jsonl'), 'a'); t0 = time.time(); done = 0
    for d in a.inputs:
        name = os.path.basename(os.path.normpath(d)); md = os.path.join(a.out, 'masks', name); os.makedirs(md, exist_ok=True)
        for ip in sorted(glob.glob(os.path.join(d, '*.jpg')) + glob.glob(os.path.join(d, '*.JPG'))):
            stem = os.path.splitext(os.path.basename(ip))[0]
            if os.path.exists(os.path.join(md, f'{stem}_{a.prompts[0].replace(" ", "_")}.png')):
                continue
            im = Image.open(ip).convert('RGB'); im.thumbnail((a.max_side, a.max_side)); W, H = im.size
            r = {'set': name, 'stem': stem, 'w': W, 'h': H}
            for pr in a.prompts:
                inp = proc(images=im, text=pr, return_tensors='pt').to(dev)
                if dev == 'cuda':
                    inp['pixel_values'] = inp['pixel_values'].to(torch.bfloat16)
                with torch.no_grad():
                    out = model(**inp)
                res = proc.post_process_instance_segmentation(out, threshold=a.threshold, mask_threshold=0.5, target_sizes=[(H, W)])[0]
                m, s = res['masks'], res['scores']
                union = m.any(0).cpu().numpy() if len(m) else np.zeros((H, W), bool)
                r[pr] = {'n': int(len(m)), 'max_score': float(s.max()) if len(s) else 0.0, 'area': float(union.mean())}
                Image.fromarray((union * 255).astype(np.uint8)).save(os.path.join(md, f'{stem}_{pr.replace(" ", "_")}.png'))
            rec.write(json.dumps(r, ensure_ascii=False) + '\n'); rec.flush(); done += 1
            if done % 100 == 0:
                print(f'{done} images, {time.time() - t0:.0f}s', flush=True)
    print('DONE', done, f'{time.time() - t0:.0f}s')


if __name__ == '__main__':
    main()
