"""Evaluate a trained author-reproduction checkpoint with the paper's metrics.

Reports TCC (Eq. 10, on CVD-simulated Lab), SSIM naturalness (Eq. 6/7) and
CD with lambda=0 (Eq. 11), per image and averaged per data group. Simulation
defaults to Machado et al. 2009 at severity 1.0 (the paper's dichromacy
setting); Brettel is available as a second-model sensitivity check.
Never tune on the test split: use --split validation for development.
"""
import argparse
import hashlib
import json
import sys
import time
from pathlib import Path

import torch
from PIL import Image

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import paths
from data import (seed_everything, read_manifest, group_paths, ImagePathDataset,
                  make_loader, write_dict_rows, mean_rows)
from metrics import batch_metrics, CD_SRGB_MAX
from models.author import make_model
from cvd_simulation import simulate_cvd


def save_panel(path, inp, out, axis, severity, simulator):
    panels = [('input', inp), ('input simulated', simulate_cvd(inp, axis, severity, simulator)),
              ('recolored', out), ('output simulated', simulate_cvd(out, axis, severity, simulator))]
    images = []
    for _, t in panels:
        array = (t.detach().cpu().clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype('uint8')
        images.append(Image.fromarray(array))
    w, h = images[0].size
    canvas = Image.new('RGB', (w * len(images), h), 'white')
    for i, im in enumerate(images):
        canvas.paste(im, (i * w, 0))
    path.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(path)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', type=Path, required=True, help='final.pt from train.py')
    p.add_argument('--axis', choices=['protan', 'deutan'], required=True)
    p.add_argument('--manifest', type=Path, default=paths.MANIFEST)
    p.add_argument('--natural-root', type=Path, default=paths.NATURAL_ROOT)
    p.add_argument('--synthetic-root', type=Path, default=paths.SYNTHETIC_ROOT)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--limit', type=int, default=0, help='Per-group image limit; 0 = all')
    p.add_argument('--simulator', choices=['machado', 'brettel'], default='machado')
    p.add_argument('--severity', type=float, default=1.0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--samples', type=int, default=8, help='Sample panels per group')
    a = p.parse_args()
    if (a.out / 'config.json').exists():
        raise ValueError('Use a fresh output directory')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.8)
    seed_everything(a.seed)
    device = torch.device('cuda')
    a.out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    ck = torch.load(a.checkpoint, map_location='cpu', weights_only=False)
    if ck['config']['axis'] != a.axis:
        raise ValueError(f"Checkpoint axis {ck['config']['axis']} != --axis {a.axis}")
    if ck['config']['manifest_sha256'] != sha:
        raise ValueError('Checkpoint was trained on a different manifest')
    if ck['step'] != ck['config']['pretrain_steps'] + ck['config']['finetune_steps']:
        raise ValueError('Checkpoint is not a completed run')
    model = make_model().to(device)
    model.load_state_dict(ck['model'])
    model.eval()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(manifest_sha256=sha, size=256,
                  metrics='paper TCC (Eq.10, simulated Lab), SSIM (Eq.6/7), CD lambda=0 (Eq.11)')
    (a.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    rows = read_manifest(a.manifest, a.natural_root, a.synthetic_root)
    all_rows = []
    eval_images = []
    for source in ['natural', 'synthetic']:
        axis_arg = a.axis if source == 'synthetic' else None
        try:
            group = group_paths(rows, source, a.split, axis_arg, a.limit, 2026)
        except ValueError as e:
            if not str(e).startswith('Empty data group:'):
                raise
            # v3 protocol: synthetic rows are train-only, so validation/test
            # legitimately has no synthetic group. Skip it, keep every other
            # empty-manifest mistake fatal.
            print(f'{source} {a.split} group empty -- skipped (v3: synthetic is train-only)', flush=True)
            continue
        root = a.natural_root if source == 'natural' else a.synthetic_root
        ds = ImagePathDataset(root, 256, group)
        loader = make_loader(ds, range(len(ds)), a.batch_size, device, a.workers, shuffle=False, seed=a.seed)
        eval_images.extend(dict(dataset=f'{source}_{a.split}', image=str(p)) for p in ds.paths)
        saved = 0
        offset = 0
        for inp, names in loader:
            inp = inp.to(device)
            torch.cuda.synchronize()
            started = time.perf_counter()
            out = (model(inp * 2 - 1) + 1) / 2
            torch.cuda.synchronize()
            ms = 1000 * (time.perf_counter() - started) / inp.shape[0]
            values = batch_metrics(inp, out, a.axis, a.severity, a.simulator, index_offset=offset, seed=a.seed)
            offset += inp.shape[0]
            for b, name in enumerate(names):
                row = dict(dataset=f'{source}_{a.split}', axis=a.axis, split=a.split,
                           simulator=a.simulator, severity=a.severity, image=name,
                           runtime_ms_per_image=ms)
                row.update({k: v[b] for k, v in values.items()})
                row['tcc_gain'] = row['tcc_output_sim'] - row['tcc_input_sim']
                row['cd_fidelity'] = 1.0 - row['cd_mean'] / CD_SRGB_MAX
                all_rows.append(row)
                if saved < a.samples:
                    save_panel(a.out / 'samples' / source / f'{a.axis}_s{a.severity:.3f}_{saved:02d}.png',
                               inp[b], out[b], a.axis, a.severity, a.simulator)
                    saved += 1
        print(source, a.split, len(group), 'images done', flush=True)
    write_dict_rows(a.out / 'evaluation_images.csv', eval_images)
    write_dict_rows(a.out / 'per_image_metrics.csv', all_rows)
    summary = mean_rows(all_rows, ['dataset', 'axis', 'simulator'])
    write_dict_rows(a.out / 'summary.csv', summary)
    lines = ['# 作者方法复现评估（论文指标）', '',
             f'模拟器 {a.simulator}，程度 {a.severity}；256×256；split={a.split}。',
             'TCC 在 CVD 模拟后的 Lab 空间计算（Eq.10）；SSIM 为自然度（Eq.6/7）；'
             f'色度保真 = 1 − λ=0 色度差CD/{CD_SRGB_MAX}（sRGB 色域色度平面直径，kornia 实测）。',
             '表格列统一为越高越好（2026-11-23 惯例）：TCC增益↑ SSIM↑ 色度保真↑；需联合判读。'
             f'CD 原值仍存于 per_image_metrics.csv 的 cd_mean 列（Eq.11 报告量，供溯源）。', '',
             '|数据|类型|模拟器|TCC输入|TCC输出|TCC增益↑|SSIM↑|色度保真↑|',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['dataset']}|{r['axis']}|{r['simulator']}|{r['tcc_input_sim']:.4f}|"
                     f"{r['tcc_output_sim']:.4f}|{r['tcc_gain']:.4f}|{r['ssim_naturalness']:.4f}|"
                     f"{r['cd_fidelity']:.4f}|")
    lines += ['', '逐图结果见 per_image_metrics.csv；样图见 samples/。',
              '单 seed、固定预算的适配复现，不等同论文表格数值。']
    (a.out / 'REPORT_zh.md').write_text('\n'.join(lines) + '\n')
    (a.out / 'COMPLETE.json').write_text(json.dumps({'rows': len(all_rows), 'manifest_sha256': sha}) + '\n')


if __name__ == '__main__':
    main()
