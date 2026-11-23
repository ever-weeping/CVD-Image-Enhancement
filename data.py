"""Self-contained data pipeline for the author-method reproduction.

Vendored from the sibling CNN project (CVD-Image-Enhancement/code:
``fixed_data.py`` and the dataset/loader helpers of
``cvd_recolor_experiments.py``) so this repository no longer depends on it.
Only the pieces the reproduction needs are kept; conventions (center crop,
bicubic resize, [0,1] CHW tensors, seeded loaders, manifest leakage checks)
are preserved verbatim.
"""
import csv
import random
from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, Subset


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


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


class ImagePathDataset(Dataset):
    def __init__(self, root: Path, size: int, paths: Optional[Sequence[Path]] = None) -> None:
        self.root = root.resolve()
        self.size = size
        self.paths = list(paths) if paths is not None else sorted(
            path for path in self.root.rglob("*") if path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        if not self.paths:
            raise FileNotFoundError(f"No images found in {root}")

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        path = self.paths[index]
        with Image.open(path) as opened:
            image = opened.convert("RGB")
            width, height = image.size
            crop = min(width, height)
            left, top = (width - crop) // 2, (height - crop) // 2
            image = image.crop((left, top, left + crop, top + crop)).resize(
                (self.size, self.size), Image.Resampling.BICUBIC
            )
            array = np.asarray(image, dtype=np.float32) / 255.0
        try:
            name = str(path.resolve().relative_to(self.root))
        except ValueError:
            name = path.name
        return torch.from_numpy(array).permute(2, 0, 1), name


def make_loader(dataset, indices, batch_size, device, workers, *, shuffle, seed):
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        Subset(dataset, indices),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=device.type == "cuda",
        generator=generator,
    )


def write_dict_rows(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        return
    fields = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def mean_rows(rows, group_fields):
    """Group rows and average every numeric field; non-numeric fields dropped."""
    groups = {}
    for row in rows:
        groups.setdefault(tuple(row[f] for f in group_fields), []).append(row)
    output = []
    for key, members in groups.items():
        summary = dict(zip(group_fields, key))
        summary['n_images'] = len(members)
        for field in members[0]:
            if field in group_fields:
                continue
            try:
                values = np.asarray([float(m[field]) for m in members], dtype=float)
            except (TypeError, ValueError):
                continue
            summary[field] = float(values.mean())
        output.append(summary)
    return sorted(output, key=lambda r: tuple(str(r[f]) for f in group_fields))
