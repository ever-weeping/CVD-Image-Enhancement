"""Own-method training on the verified author reproduction (Chen et al., NCAA 2024).

Default simulation convention is `linearized` (own method, promoted 2026-11-23;
see docs/METHOD_zh.md); pass `--sim-train author` for reproduction/ablation
controls. Unsupervised protocol: alpha=0 pretraining (contrast only), then
finetuning with the naturalness weight alpha (default 0.5). Adam lr=2e-4,
betas=(0.5,0.999), batch 4, 256x256, one axis-specific model per run.
Self-contained: data pipeline in data.py, network in models/author.py
(released author code under vendor/), loss in losses/author.py.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time
from collections import Counter

import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import trange

BASE = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE))

import paths
from data import (seed_everything, read_manifest, group_paths,
                  ImagePathDataset, make_loader)
from models.author import make_model
from losses.author import AuthorLoss


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--manifest', type=Path, default=paths.MANIFEST)
    p.add_argument('--natural-root', type=Path, default=paths.NATURAL_ROOT)
    p.add_argument('--synthetic-root', type=Path, default=paths.SYNTHETIC_ROOT)
    p.add_argument('--out', type=Path, default=None,
                   help='Run directory; default <RUN_ROOT>/author_<axis>_alpha<alpha>')
    p.add_argument('--axis', choices=['protan', 'deutan'], required=True)
    p.add_argument('--pretrain-steps', type=int, default=20000)
    p.add_argument('--finetune-steps', type=int, default=20000)
    p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--batch-size', type=int, default=4)
    p.add_argument('--workers', type=int, default=2)
    p.add_argument('--seed', type=int, default=7)
    p.add_argument('--save-every', type=int, default=1000)
    p.add_argument('--resume', action='store_true')
    p.add_argument('--init-from', type=Path, default=None,
                   help='Start from a shared alpha=0 pretrain checkpoint (model+optimizer+rng); '
                        'runs only the --finetune-steps phase at --alpha')
    p.add_argument('--sim-train', choices=['author', 'linearized'], default='linearized',
                   help='Training-time CVD simulation convention: author = matrix on encoded '
                        'RGB (faithful reproduction control), linearized = sRGB decode/encode around '
                        'the matrix (own method default, matches evaluate.py). Promoted 2026-11-23.')
    a = p.parse_args()
    if a.out is None:
        prefix = 'author' if a.sim_train == 'author' else 'method'
        a.out = paths.RUN_ROOT / f'{prefix}_{a.axis}_alpha{a.alpha:g}'
    if min(a.pretrain_steps, a.finetune_steps) < 0 or a.pretrain_steps + a.finetune_steps < 1:
        raise ValueError('Invalid steps')
    torch.set_num_threads(4)
    torch.cuda.set_per_process_memory_fraction(.80)
    seed_everything(a.seed)
    device = torch.device('cuda')
    rows = read_manifest(a.manifest, a.natural_root, a.synthetic_root)
    datasets = {
        'natural': ImagePathDataset(a.natural_root, 256, group_paths(rows, 'natural', 'train')),
        'synthetic': ImagePathDataset(a.synthetic_root, 256, group_paths(rows, 'synthetic', 'train', a.axis))}
    loaders = {k: make_loader(ds, range(len(ds)), a.batch_size, device, a.workers, shuffle=True, seed=a.seed)
               for k, ds in datasets.items()}
    iters = {k: iter(v) for k, v in loaders.items()}
    config = {k: str(v) if isinstance(v, Path) else v for k, v in vars(a).items()}
    config.update(manifest_sha256=hashlib.sha256(a.manifest.read_bytes()).hexdigest(),
                  author_commit='07271c5bc0f89068e8fc8cfa57cec0a9b8800549',
                  severity=1., size=256, synthetic_ratio=.5, lr=.0002, betas=[.5, .999],
                  dataset_counts={k: len(v) for k, v in datasets.items()},
                  protocol=f'{a.sim_train}-sim recoloring training; fixed manifest; '
                           'alpha=0 pretrain then alpha finetune')
    a.out.mkdir(parents=True, exist_ok=True)
    if (a.out / 'config.json').exists() and not a.resume:
        raise ValueError('Run exists; use a fresh directory or --resume')
    model = make_model().to(device)
    loss_fn = AuthorLoss(linearize=a.sim_train == 'linearized').to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=.0002, betas=(.5, .999))
    rng = random.Random(a.seed + 719)
    counts = Counter()
    start_step = 0
    if a.init_from is not None:
        if a.resume:
            raise ValueError('--init-from and --resume are mutually exclusive')
        if not a.init_from.is_file():
            raise ValueError(f'--init-from checkpoint missing: {a.init_from}')
        ck = torch.load(a.init_from, map_location=device, weights_only=False)
        for key in ['manifest_sha256', 'axis', 'batch_size', 'seed']:
            if ck['config'][key] != config[key]:
                raise ValueError(f'init-from mismatch: {key}')
        if ck['step'] != ck['config']['pretrain_steps']:
            raise ValueError('init-from checkpoint must sit exactly at the alpha=0 boundary')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_step = ck['step']
        rng.setstate(ck['source_rng'])
        counts.update(ck['sampling'])
        # Executed steps all lie past the boundary, so every step uses --alpha.
        a.pretrain_steps = start_step
        config['pretrain_steps'] = start_step
        config['init_from'] = str(a.init_from)
        config['init_from_sha256'] = hashlib.sha256(a.init_from.read_bytes()).hexdigest()
        config['init_note'] = ('Branch run from shared alpha=0 pretrain: weights/optimizer/rng '
                               'restored at the boundary; data order restarts from seed.')
    if a.resume:
        ck = torch.load(a.out / 'latest.pt', map_location=device, weights_only=False)
        for key in ['manifest_sha256', 'axis', 'pretrain_steps', 'finetune_steps', 'alpha', 'batch_size', 'seed']:
            if ck['config'][key] != config[key]:
                raise ValueError(f'Resume mismatch: {key}')
        model.load_state_dict(ck['model'])
        optimizer.load_state_dict(ck['optimizer'])
        start_step = ck['step']
        rng.setstate(ck['source_rng'])
        counts.update(ck['sampling'])
        # DataLoader permutation restarts on resume; explicitly recorded, not bitwise replay.
        config['resume_note'] = 'Weights/optimizer/step restored; data order restarts from seed.'
    (a.out / 'config.json').write_text(json.dumps(config, indent=2) + '\n')
    (a.out / 'fixed_manifest.csv').write_bytes(a.manifest.read_bytes())
    total = a.pretrain_steps + a.finetune_steps

    def save(step, filename):
        ck = dict(model=model.state_dict(), optimizer=optimizer.state_dict(), step=step,
                  config=config, source_rng=rng.getstate(), sampling=dict(counts))
        temp = a.out / (filename + '.tmp')
        torch.save(ck, temp)
        os.replace(temp, a.out / filename)

    started = time.monotonic()
    model.train()
    with SummaryWriter(str(a.out / 'tensorboard')) as writer:
        for step in trange(start_step + 1, total + 1, desc=a.axis, disable=not sys.stdout.isatty()):
            source = 'synthetic' if rng.random() < .5 else 'natural'
            try:
                inp, _ = next(iters[source])
            except StopIteration:
                iters[source] = iter(loaders[source])
                inp, _ = next(iters[source])
            inp = inp.to(device)
            alpha = 0. if step <= a.pretrain_steps else a.alpha
            optimizer.zero_grad(set_to_none=True)
            out = (model(inp * 2 - 1) + 1) / 2
            loss, logs = loss_fn(inp, out, a.axis, alpha)
            if not torch.isfinite(loss):
                raise FloatingPointError(f'Nonfinite loss at {step}')
            loss.backward()
            if any(not torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None):
                raise FloatingPointError(f'Nonfinite gradient at {step}')
            optimizer.step()
            counts[source] += len(inp)
            for key, val in logs.items():
                writer.add_scalar('train/' + key, val, step)
            writer.add_scalar('train/alpha', alpha, step)
            writer.add_scalar('train/synthetic_batch', source == 'synthetic', step)
            if step == 1 or step % 20 == 0 or step == total:
                elapsed = time.monotonic() - started
                status = dict(step=step, total=total, axis=a.axis, alpha=alpha, **logs,
                              elapsed_seconds=elapsed,
                              eta_seconds=elapsed / (step - start_step) * (total - step),
                              sampling=dict(counts))
                (a.out / 'progress.json').write_text(json.dumps(status, indent=2) + '\n')
                print(json.dumps(status), flush=True)
            if step % a.save_every == 0 or step == total or step == a.pretrain_steps:
                save(step, 'latest.pt')
            if step == a.pretrain_steps:
                save(step, 'pretrained_alpha0.pt')
    save(total, 'final.pt')
    (a.out / 'COMPLETE.json').write_text(
        json.dumps({'steps': total, 'elapsed_seconds': time.monotonic() - started}) + '\n')


if __name__ == '__main__':
    main()
