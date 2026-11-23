#!/usr/bin/env python3
"""Freeze natural splits and content-hash-based 90/5/5 artificial splits once."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
from pathlib import Path
import os


def build_manifest(natural_root, synthetic_root, output, seed=20261010, workers=4):
    natural_root=Path(natural_root).resolve();synthetic_root=Path(synthetic_root).resolve()
    output=Path(output)
    if output.exists():
        raise FileExistsError(f'Refusing to replace fixed manifest: {output}')
    items=[]
    suffixes={'.jpg','.jpeg','.png'}
    for split in ['train','validation','test']:
        folder=natural_root/split
        paths=sorted(p for p in folder.rglob('*') if p.is_file() and p.suffix.lower() in suffixes)
        if not paths:raise ValueError(f'Empty fixed natural split: {folder}')
        items.extend(('natural','both',split,p) for p in paths)
    for axis,folder in [('protan','Color_cvd_P_experiment_100000'),('deutan','Color_cvd_D_experiment_100000')]:
        paths=sorted(p for p in (synthetic_root/folder).rglob('*') if p.is_file() and p.suffix.lower() in suffixes)
        if not paths:raise ValueError(f'No author images: {synthetic_root/folder}')
        items.extend(('synthetic',axis,None,p) for p in paths)
    def hashed(item):
        h=hashlib.sha256()
        with item[3].open('rb') as f:
            for chunk in iter(lambda:f.read(1024*1024),b''):h.update(chunk)
        return item,h.hexdigest()
    rows=[];seen={};duplicates=0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for start in range(0,len(items),512):
            for (source,axis,split,path),sha in pool.map(hashed,items[start:start+512]):
                if sha in seen:
                    if source=='natural' and seen[sha]!=split:
                        raise ValueError(f'Natural content overlaps fixed splits: {path}')
                    duplicates+=1
                    continue
                if source=='synthetic':
                    bucket=int(hashlib.sha256(f'{seed}:{sha}'.encode()).hexdigest()[:16],16)%100
                    split='train' if bucket<90 else ('validation' if bucket<95 else 'test')
                seen[sha]=split
                root=natural_root if source=='natural' else synthetic_root
                rows.append(dict(source=source,axis=axis,split=split,image=str(path.relative_to(root)),sha256=sha))
            print(f'Hashed {min(start+512,len(items))}/{len(items)}',flush=True)
    counts=Counter((r['source'],r['axis'],r['split']) for r in rows)
    for axis in ['protan','deutan']:
        for split in ['train','validation','test']:
            if not counts['synthetic',axis,split]:
                raise ValueError(f'Empty synthetic {axis}/{split}; need more unique images')
    output.parent.mkdir(parents=True,exist_ok=True)
    temporary=output.with_suffix(output.suffix+'.tmp')
    with temporary.open('x',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['source','axis','split','image','sha256'])
        writer.writeheader();writer.writerows(rows)
    # Atomic publication without replacing an existing manifest.
    os.link(temporary,output);temporary.unlink()
    metadata=dict(seed=seed,synthetic_split='SHA256(seed:content_sha256) first 64 bits modulo 100: <90 train, <95 validation, else test',
                  duplicates_removed=duplicates,counts={'/'.join(k):v for k,v in counts.items()},
                  sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
                  note='Natural folder assignments preserved; global exact-file dedup, natural takes priority; no perceptual dedup')
    output.with_suffix('.meta.json').write_text(json.dumps(metadata,indent=2)+'\n')
    print(json.dumps(metadata,indent=2),flush=True)
    return rows


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--natural-root',type=Path,required=True)
    parser.add_argument('--synthetic-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--seed',type=int,default=20261010)
    parser.add_argument('--workers',type=int,default=4)
    args=parser.parse_args()
    build_manifest(args.natural_root,args.synthetic_root,args.output,args.seed,args.workers)
