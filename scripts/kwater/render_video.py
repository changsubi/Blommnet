#!/usr/bin/env python
"""동영상 추론 결과 영상 생성.
  <name>_BloomNet.mp4          : BloomNet(3클래스) 오버레이, 1920x1080, 원본 fps
  <name>_BloomNet_vs_SAM3.mp4  : 좌 BloomNet / 우 SAM3 zero-shot 'green water' (N 프레임마다 갱신), 1920x540
사용: PYTHONPATH=<repo_root> python render_video.py --video in.mp4 --config <run>/config.resolved.yaml --ckpt <run>/best.pt --out_dir out [--no_sam3] [--sam3_model facebook/sam3]"""
import argparse, os, sys, json, time, re
import numpy as np, torch
from PIL import Image, ImageDraw, ImageFont
import imageio_ffmpeg
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from eval_pseudo import load_weights, predict, PAL3
from bloomnet.config import load_config
from bloomnet.models.bloomnet import build_bloomnet

FONT = ImageFont.load_default(size=28)


def label(im, text, xy=(12, 10)):
    d = ImageDraw.Draw(im); tw = d.textlength(text, font=FONT)
    d.rectangle([xy[0] - 6, xy[1] - 4, xy[0] + tw + 6, xy[1] + 34], fill=(0, 0, 0)); d.text(xy, text, fill=(255, 255, 255), font=FONT)
    return im


def overlay_np(rgb, lab, pal, alpha=0.5):
    out = rgb.astype(np.float32).copy()
    for c in range(1, len(pal)):
        m = lab == c
        if m.any():
            out[m] = (1 - alpha) * out[m] + alpha * pal[c]
    return out.astype(np.uint8)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--video', required=True); ap.add_argument('--config', required=True); ap.add_argument('--ckpt', required=True)
    ap.add_argument('--out_dir', required=True); ap.add_argument('--name', default=None); ap.add_argument('--max_frames', type=int, default=0)
    ap.add_argument('--sam3_every', type=int, default=5); ap.add_argument('--sam3_prompt', default='green water'); ap.add_argument('--sam3_model', default='facebook/sam3'); ap.add_argument('--no_sam3', action='store_true')
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    name = a.name or re.sub(r'\W+', '_', os.path.splitext(os.path.basename(a.video))[0]).strip('_')
    dev = torch.device('cuda')
    cfg = load_config(a.config, base=None); model = build_bloomnet(cfg); print('[ckpt]', load_weights(model, a.ckpt)); model.to(dev).eval()
    if not a.no_sam3:
        from transformers import Sam3Model, Sam3Processor
        sam = Sam3Model.from_pretrained(a.sam3_model, torch_dtype=torch.bfloat16).to(dev).eval(); proc = Sam3Processor.from_pretrained(a.sam3_model)
    reader = imageio_ffmpeg.read_frames(a.video, pix_fmt='rgb24', output_params=['-vf', 'scale=1920:1080'])
    meta = next(reader); fps = meta['fps']; W, H = 1920, 1080
    w1 = imageio_ffmpeg.write_frames(f'{a.out_dir}/{name}_BloomNet.mp4', (W, H), fps=fps, codec='libx264', bitrate='6M', pix_fmt_out='yuv420p', macro_block_size=1, output_params=['-preset', 'veryfast']); w1.send(None)
    if not a.no_sam3:
        w2 = imageio_ffmpeg.write_frames(f'{a.out_dir}/{name}_BloomNet_vs_SAM3.mp4', (W, H // 2), fps=fps, codec='libx264', bitrate='4M', pix_fmt_out='yuv420p', macro_block_size=1, output_params=['-preset', 'veryfast']); w2.send(None)
    stats, t_b, t_s, sam_mask, sam_score, n, t0 = [], [], [], np.zeros((H, W), bool), 0.0, 0, time.time()
    for fr in reader:
        rgb = np.frombuffer(fr, np.uint8).reshape(H, W, 3)
        torch.cuda.synchronize(); t1 = time.perf_counter(); pred = predict(model, cfg, rgb, dev); torch.cuda.synchronize(); t_b.append(time.perf_counter() - t1)
        frac = {'water': float((pred == 1).mean()), 'algae_water': float((pred == 2).mean())}
        im1 = Image.fromarray(overlay_np(rgb, pred, PAL3))
        label(im1, f'BloomNet fine-tuned on pseudo-labels (no ground truth) | green = predicted "algae water" class {100*frac["algae_water"]:.1f}%  blue = other water {100*frac["water"]:.1f}% | {1000*t_b[-1]:.0f} ms/frame')
        w1.send(np.asarray(im1))
        if not a.no_sam3:
            if n % a.sam3_every == 0:
                t2 = time.perf_counter()
                inp = proc(images=Image.fromarray(rgb), text=a.sam3_prompt, return_tensors='pt').to(dev); inp['pixel_values'] = inp['pixel_values'].to(torch.bfloat16)
                with torch.no_grad():
                    out = sam(**inp)
                res = proc.post_process_instance_segmentation(out, threshold=0.3, mask_threshold=0.5, target_sizes=[(H, W)])[0]
                sam_mask = res['masks'].any(0).cpu().numpy() if len(res['masks']) else np.zeros((H, W), bool)
                sam_score = float(res['scores'].max()) if len(res['scores']) else 0.0
                torch.cuda.synchronize(); t_s.append(time.perf_counter() - t2)
            im2 = Image.fromarray(overlay_np(rgb, sam_mask.astype(np.uint8) * 2, PAL3))
            label(im2, f"SAM3 zero-shot | prompt '{a.sam3_prompt}' {100*sam_mask.mean():.1f}% score {sam_score:.2f} | {1000*np.mean(t_s):.0f} ms/frame")
            side = Image.new('RGB', (W, H // 2)); side.paste(im1.resize((W // 2, H // 2)), (0, 0)); side.paste(im2.resize((W // 2, H // 2)), (W // 2, 0))
            w2.send(np.asarray(side)); frac['sam3'] = float(sam_mask.mean())
        stats.append(frac); n += 1
        if n % 300 == 0:
            print(f'{n} frames {time.time()-t0:.0f}s', flush=True)
        if a.max_frames and n >= a.max_frames:
            break
    w1.close()
    if not a.no_sam3:
        w2.close()
    json.dump({'video': a.video, 'frames': n, 'fps': fps, 'bloomnet_ms_mean': 1000 * float(np.mean(t_b)), 'sam3_ms_mean': 1000 * float(np.mean(t_s)) if t_s else None, 'per_frame': stats}, open(f'{a.out_dir}/{name}_stats.json', 'w'), indent=0)
    print('DONE', n, 'frames')


if __name__ == '__main__':
    main()
