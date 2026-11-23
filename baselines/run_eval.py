"""Same-set baseline re-evaluation harness (docs/BASELINES_zh.md protocol).

Evaluates any registered baseline (baselines/methods/<name>.py exposing
``recolor(inp, axis, **params) -> torch.Tensor``) on the v3 natural
validation split with the paper's three metrics, producing the SAME output
schema as evaluate.py so baseline and method rows merge directly.

Numbers reported are OUR re-measurements only; never compare them against
the baselines' own tables (see docs/BASELINES_zh.md Sec.3 acceptance tiers).
"""
import argparse
import hashlib
import json
import sys
import time
from contextlib import nullcontext
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(BASE))

import paths
from data import (seed_everything, read_manifest, group_paths, ImagePathDataset,
                  make_loader, write_dict_rows, mean_rows)
from metrics import batch_metrics, CD_SRGB_MAX
from baselines.methods import load_method


def coerce_params(defaults, pairs, json_file):
    params = dict(defaults)
    if json_file:
        params.update(json.loads(Path(json_file).read_text()))
    for pair in pairs or []:
        key, _, raw = pair.partition('=')
        if key not in defaults:
            raise ValueError(f'Unknown param {key!r}; defaults are {sorted(defaults)}')
        current = defaults[key]
        if isinstance(current, bool):
            value = raw.lower() in ('1', 'true', 'yes')
        elif isinstance(current, int):
            value = int(raw)
        elif isinstance(current, float):
            value = float(raw)
        elif current is None or isinstance(current, str):
            # bool/int/float defaults can also be tuned from a CLI string when the
            # default itself is a string sentinel (e.g. alpha='auto'): try downcast.
            try:
                value = int(raw)
            except ValueError:
                try:
                    value = float(raw)
                except ValueError:
                    value = raw
        else:
            value = raw
        params[key] = value
    return params


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True, help='baselines/methods/<name>.py')
    p.add_argument('--param', action='append', help='k=v override of the method DEFAULTS, repeatable')
    p.add_argument('--params-json', type=Path, default=None)
    p.add_argument('--axes', nargs='+', default=['protan', 'deutan'], choices=['protan', 'deutan'])
    p.add_argument('--manifest', type=Path, default=paths.MANIFEST)
    p.add_argument('--natural-root', type=Path, default=paths.NATURAL_ROOT)
    p.add_argument('--split', choices=['validation', 'test'], default='validation')
    p.add_argument('--limit', type=int, default=0, help='Per-axis image limit; 0 = all (smoke: 200)')
    p.add_argument('--simulator', choices=['machado', 'brettel'], default='machado')
    p.add_argument('--severity', type=float, default=1.0)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--batch-size', type=int, default=8)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    a = p.parse_args()

    module = load_method(a.method)
    params = coerce_params(module.DEFAULTS, a.param, a.params_json)
    device = torch.device('cuda' if (a.device == 'auto' and torch.cuda.is_available())
                          else a.device if a.device != 'auto' else 'cpu')
    if (a.out / 'config.json').exists():
        raise ValueError('Use a fresh output directory (runs are immutable)')
    torch.set_num_threads(4)
    seed_everything(a.seed)
    a.out.mkdir(parents=True, exist_ok=True)
    sha = hashlib.sha256(a.manifest.read_bytes()).hexdigest()
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(manifest_sha256=sha, size=256, params=params, device=str(device),
                  metrics='paper TCC (Eq.10, simulated Lab), SSIM (Eq.6/7), CD lambda=0 (Eq.11)',
                  protocol='docs/BASELINES_zh.md')
    (a.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')

    rows_manifest = read_manifest(a.manifest, a.natural_root, paths.SYNTHETIC_ROOT)
    all_rows, eval_images = [], []
    tag = json.dumps(params, sort_keys=True)
    for axis in a.axes:
        group = group_paths(rows_manifest, 'natural', a.split, None, a.limit, 2026)
        ds = ImagePathDataset(a.natural_root, 256, group)
        loader = make_loader(ds, range(len(ds)), a.batch_size, device, a.workers,
                             shuffle=False, seed=a.seed)
        eval_images.extend(dict(dataset=f'natural_{a.split}', image=str(pth)) for pth in ds.paths)
        offset = 0
        with nullcontext() if not getattr(module, 'NO_GRAD', True) else torch.no_grad():
            for inp, names in loader:
                inp = inp.to(device)
                started = time.perf_counter()
                out = module.recolor(inp, axis, **params)
                if out.device != inp.device:
                    out = out.to(inp.device)
                ms = 1000 * (time.perf_counter() - started) / inp.shape[0]
                values = batch_metrics(inp, out, axis, a.severity, a.simulator,
                                       index_offset=offset, seed=a.seed)
                offset += inp.shape[0]
                for b, name in enumerate(names):
                    row = dict(dataset=f'natural_{a.split}', axis=axis, split=a.split,
                               simulator=a.simulator, severity=a.severity, image=name,
                               method=a.method, params=tag, runtime_ms_per_image=ms)
                    row.update({k: v[b] for k, v in values.items()})
                    row['tcc_gain'] = row['tcc_output_sim'] - row['tcc_input_sim']
                    row['cd_fidelity'] = 1.0 - row['cd_mean'] / CD_SRGB_MAX
                    all_rows.append(row)
            print(axis, a.split, len(group), 'images done', flush=True)

    write_dict_rows(a.out / 'evaluation_images.csv', eval_images)
    write_dict_rows(a.out / 'per_image_metrics.csv', all_rows)
    summary = mean_rows(all_rows, ['dataset', 'axis', 'method', 'simulator'])
    write_dict_rows(a.out / 'summary.csv', summary)
    lines = ['# 基线同集重测评估（论文指标，v3 协议）', '',
             f'方法 `{a.method}` 参数 `{tag}`；模拟器 {a.simulator}，程度 {a.severity}；'
             f'256×256；split={a.split}；limit={a.limit}。',
             '口径与 docs/BASELINES_zh.md 第 2 节一致：TCC 在 CVD 模拟后 Lab（Eq.10）；'
             f'SSIM 自然度（Eq.6/7）；色度保真 = 1 − λ=0 色度差CD/{CD_SRGB_MAX}。',
             '表格列统一越高越好：TCC增益↑ SSIM↑ 色度保真↑；需联合判读。',
             '本表全部为我方同集重测数字，与原文表格不可对数（欠定台账见方案文档）。', '',
             '|数据|类型|模拟器|TCC输入|TCC输出|TCC增益↑|SSIM↑|色度保真↑|',
             '|---|---|---|---:|---:|---:|---:|---:|']
    for r in summary:
        lines.append(f"|{r['dataset']}|{r['axis']}|{r['simulator']}|{r['tcc_input_sim']:.4f}|"
                     f"{r['tcc_output_sim']:.4f}|{r['tcc_gain']:.4f}|{r['ssim_naturalness']:.4f}|"
                     f"{r['cd_fidelity']:.4f}|")
    lines += ['', '逐图结果见 per_image_metrics.csv。']
    (a.out / 'REPORT_zh.md').write_text('\n'.join(lines) + '\n')
    (a.out / 'COMPLETE.json').write_text(json.dumps({'rows': len(all_rows), 'manifest_sha256': sha}) + '\n')


if __name__ == '__main__':
    main()
