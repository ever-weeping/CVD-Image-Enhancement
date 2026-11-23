"""Explicit splits and axis-matched synthetic batches; no recursive re-splitting."""
import csv
import hashlib
import json
from pathlib import Path
import random
import shutil

import torch


def read_manifest(path, natural_root, synthetic_root):
    roots = {'natural': Path(natural_root).resolve()}
    if synthetic_root:
        roots['synthetic'] = Path(synthetic_root).resolve()
    with Path(path).open(newline='', encoding='utf-8') as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError('Empty manifest')
    hashes, files = {}, set()
    for row in rows:
        source, split, axis = row['source'], row['split'], row['axis']
        if source not in roots or split not in ('train', 'validation', 'test'):
            raise ValueError(f'Invalid source/split or missing root: {row}')
        if (source == 'natural' and axis != 'both') or (source == 'synthetic' and axis not in ('protan', 'deutan')):
            raise ValueError(f'Invalid axis: {row}')
        sha = row['sha256']
        if len(sha) != 64 or any(c not in '0123456789abcdef' for c in sha):
            raise ValueError('Manifest needs valid SHA256 values')
        image = Path(row['image'])
        resolved = (roots[source] / image).resolve()
        if image.is_absolute() or not resolved.is_relative_to(roots[source]) or not resolved.is_file():
            raise ValueError(f'Missing or unsafe image: {image}')
        if resolved in files:
            raise ValueError(f'Duplicate image path: {image}')
        if sha in hashes and hashes[sha] != split:
            raise ValueError(f'Content leakage across splits: {image}')
        if sha in hashes:
            raise ValueError(f'Duplicate content in manifest: {image}')
        files.add(resolved)
        hashes[sha] = split
        row['path'] = resolved
    return rows


def group_paths(rows, source, split, axis=None, limit=0, seed=2026):
    selected = [r['path'] for r in rows if r['source'] == source and r['split'] == split
                and (axis is None or r['axis'] == axis)]
    if not selected:
        raise ValueError(f'Empty data group: {source}/{split}/{axis}')
    selected.sort()
    if limit:
        random.Random(seed).shuffle(selected)
        selected = selected[:limit]
    return selected


def run_fixed(args):
    import cvd_recolor_experiments as exp
    from cvd_models import VARIANTS, make_model, parameter_count
    from runtime import resolve_device
    if args.steps < 1 or args.batch_size < 1:
        raise ValueError('Steps and batch size must be positive')
    if not 0 <= args.synthetic_ratio <= 1 or args.fixed_eval_limit < 0:
        raise ValueError('Invalid mixture ratio or evaluation limit')
    if args.stress_data:
        raise ValueError('Use synthetic held-out manifest groups; --stress-data is forbidden in fixed mode')
    if args.fixed_eval_split == 'test' and not args.eval_only_checkpoint:
        raise ValueError('Test split requires --eval-only-checkpoint; train/tune on validation first')
    if args.eval_only_checkpoint and (len(args.seeds) != 1 or len(args.variants) != 1):
        raise ValueError('Checkpoint evaluation needs exactly one seed and one variant')
    if (args.out / 'config.json').exists():
        raise ValueError('Output already contains a run; use a fresh --out')
    rows = read_manifest(args.fixed_manifest, args.data, args.synthetic_root)
    has_synthetic = any(r['source'] == 'synthetic' for r in rows)
    if args.synthetic_ratio > 0 and not has_synthetic:
        raise ValueError('Manifest has no synthetic data; set --synthetic-ratio 0 for natural-only control')
    device = resolve_device(args.device)
    natural_name = f'natural_{args.fixed_eval_split}'
    groups = [(natural_name, 'natural', None, args.axes)]
    if has_synthetic:
        groups += [(f'synthetic_{axis}_{args.fixed_eval_split}', 'synthetic', axis, [axis]) for axis in args.axes]
    eval_groups = []
    for name, source, axis, axes in groups:
        paths = group_paths(rows, source, args.fixed_eval_split, axis, args.fixed_eval_limit, args.split_seed)
        root = args.data if source == 'natural' else args.synthetic_root
        dataset = exp.ImagePathDataset(root, args.size, paths)
        eval_groups.append((name, dataset, axes))
    train_paths = group_paths(rows, 'natural', 'train')
    synthetic_paths = {axis: group_paths(rows, 'synthetic', 'train', axis) for axis in args.axes} if args.synthetic_ratio else {}
    args.out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(args.fixed_manifest, args.out / 'fixed_manifest.csv')
    cfg = {k: str(v.resolve()) if isinstance(v, Path) else v for k,v in vars(args).items()}
    cfg.update(manifest_sha256=hashlib.sha256(args.fixed_manifest.read_bytes()).hexdigest(),
               device_resolved=str(device), git_commit_at_start=exp.git_commit(), torch_version=torch.__version__,
               natural_train_count=len(train_paths), synthetic_train_counts={k:len(v) for k,v in synthetic_paths.items()},
               eval_counts={name:len(ds) for name,ds,_ in eval_groups},
               sampling='One source per batch; synthetic ratio is a probability, matching P/protan and D/deutan',
               legacy_count_flags='--train-images/--eval-images do not apply in fixed-manifest mode')
    (args.out/'config.json').write_text(json.dumps(cfg,indent=2)+'\n')
    exp.write_dict_rows(args.out/'evaluation_images.csv', [dict(dataset=name,image=str(p)) for name,ds,_ in eval_groups for p in ds.paths])
    print(f'Fixed split: train={len(train_paths)} synthetic={cfg["synthetic_train_counts"]} eval={cfg["eval_counts"]}',flush=True)
    results, history = [], []
    def evaluate(name, seed, model=None, transform=None, seconds=0., policy='correct'):
        for dataset_name, ds, axes in eval_groups:
            loader = exp.make_loader(ds,range(len(ds)),args.batch_size,device,args.num_workers,shuffle=False,seed=seed)
            protocols = [(args.train_simulator, sorted(set(args.eval_severities+args.interpolation_severities)))]
            if policy == 'correct':
                protocols.append((args.robustness_simulator,[1.]))
            for simulator, severities in protocols:
                results.extend(exp.evaluate_variant(name=name,seed=seed,loader=loader,dataset_name=dataset_name,
                    device=device,axes=axes,severities=severities,metric_simulator=simulator,
                    mask_simulator=args.train_simulator,model=model,transform=transform,
                    params=parameter_count(model) if model is not None else 0,train_seconds=seconds,
                    condition_policy=policy,no_progress=args.no_progress,
                    sample_root=args.out/'samples'/dataset_name if simulator==args.train_simulator and seed==args.seeds[0] else None,
                    sample_images=args.sample_images))
    for seed in args.seeds:
        for name, transform in [('identity',exp.identity_transform),('error_compensation',exp.error_compensation_transform)]:
            evaluate(name,seed,transform=transform)
        for name in args.variants:
            exp.seed_everything(seed)
            model = make_model(VARIANTS[name],args.width).to(device)
            seconds=0.
            if args.eval_only_checkpoint:
                ckpt=torch.load(args.eval_only_checkpoint,map_location=device,weights_only=True)
                if ckpt['variant']!=name or ckpt['seed']!=seed:
                    raise ValueError('Checkpoint variant/seed mismatch')
                if ckpt.get('manifest_sha256')!=cfg['manifest_sha256']:
                    raise ValueError('Checkpoint must use the same fixed manifest')
                if ckpt.get('width') != args.width:
                    raise ValueError('Checkpoint width differs from --width')
                model.load_state_dict(ckpt['model'])
            else:
                def loader_for(root,paths):
                    ds=exp.ImagePathDataset(root,args.size,paths)
                    return exp.make_loader(ds,range(len(ds)),args.batch_size,device,args.num_workers,shuffle=True,seed=seed)
                natural_loader=loader_for(args.data,train_paths)
                synthetic_loaders={axis:loader_for(args.synthetic_root,paths) for axis,paths in synthetic_paths.items()}
                h,seconds=exp.train_model(model,VARIANTS[name],natural_loader,device,args,seed,synthetic_loaders)
                history.extend(h)
                if args.save_checkpoints:
                    (args.out/'checkpoints').mkdir(exist_ok=True)
                    torch.save(dict(model=model.state_dict(),variant=name,seed=seed,width=args.width,
                                    manifest_sha256=cfg['manifest_sha256']),args.out/'checkpoints'/f'{name}_seed{seed}.pt')
            evaluate(name,seed,model=model,seconds=seconds)
            if args.condition_probes and name=='ours_full':
                for suffix,policy in [('axis_swap','axis_swap'),('fixed_s05','fixed_s05')]:
                    evaluate(f'{name}_{suffix}',seed,model=model,seconds=seconds,policy=policy)
            del model
    exp.write_dict_rows(args.out/'per_image_metrics.csv',results)
    exp.write_dict_rows(args.out/'train_history.csv',history)
    by_seed=exp.mean_rows(results,['dataset','seed','variant','axis','simulated_condition','simulator_severity','metric_simulator','mask_simulator'])
    exp.write_dict_rows(args.out/'summary_by_seed.csv',by_seed)
    conditions=exp.mean_rows(by_seed,['dataset','variant','axis','simulated_condition','simulator_severity','metric_simulator','mask_simulator'])
    exp.write_dict_rows(args.out/'summary_conditions.csv',conditions)
    overall=exp.mean_rows([r for r in by_seed if r['metric_simulator']==args.train_simulator],['dataset','variant'])
    exp.write_dict_rows(args.out/'summary_overall.csv',overall)
    stats=[]
    for name,_,_ in eval_groups:
        stats.extend(dict(dataset=name,**r) for r in exp.paired_statistics(results,args.bootstrap_samples,args.split_seed,name,args.train_simulator))
    exp.write_dict_rows(args.out/'paired_statistics.csv',stats)
    print(f'Wrote fixed-split results to {args.out.resolve()}',flush=True)
