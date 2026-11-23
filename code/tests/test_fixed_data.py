import argparse
import csv
import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from torch.utils.data import DataLoader,TensorDataset
from PIL import Image
import cvd_recolor_experiments as exp
from cvd_models import VARIANTS,make_model
from fixed_data import read_manifest,group_paths
from scripts.prepare_fixed_manifest import build_manifest

class FixedDataTests(unittest.TestCase):
    def test_content_leak_and_unsafe_paths_rejected(self):
        with tempfile.TemporaryDirectory() as temp:
            r=Path(temp);(r/'a.png').write_bytes(b'a');(r/'b.png').write_bytes(b'a');manifest=r/'m.csv'
            def write(rows):
                with manifest.open('w',newline='') as f:
                    w=csv.DictWriter(f,fieldnames=['source','axis','split','image','sha256']);w.writeheader();w.writerows(rows)
            base=dict(source='natural',axis='both',split='train',image='a.png',sha256=hashlib.sha256(b'a').hexdigest())
            write([base,dict(base,split='test',image='b.png')])
            with self.assertRaisesRegex(ValueError,'leakage'):read_manifest(manifest,r,None)
            write([dict(base,image='../outside.png')])
            with self.assertRaisesRegex(ValueError,'unsafe'):read_manifest(manifest,r,None)

    def test_prepare_preserves_splits_and_deduplicates(self):
        with tempfile.TemporaryDirectory() as temp:
            r=Path(temp);n=r/'natural';s=r/'synthetic'
            for i,split in enumerate(['train','validation','test']):
                (n/split).mkdir(parents=True);Image.new('RGB',(8,8),(i+1,0,0)).save(n/split/'same.png')
            for axis,folder in enumerate(['Color_cvd_P_experiment_100000','Color_cvd_D_experiment_100000']):
                (s/folder).mkdir(parents=True)
                for i in range(180):Image.new('RGB',(8,8),(i,axis+20,30)).save(s/folder/f'{i}.png')
            (s/'Color_cvd_D_experiment_100000'/'duplicate.png').write_bytes((n/'test'/'same.png').read_bytes())
            with patch('builtins.print'):build_manifest(n,s,r/'m.csv',workers=2)
            rows=read_manifest(r/'m.csv',n,s);self.assertEqual(len(rows),363)
            for split in ['train','validation','test']:
                self.assertEqual(len(group_paths(rows,'natural',split)),1)
                for axis in ['protan','deutan']:self.assertTrue(group_paths(rows,'synthetic',split,axis))
            with self.assertRaises(FileExistsError):build_manifest(n,s,r/'m.csv')

    def test_mixed_batches_match_axis_and_reproduce(self):
        with tempfile.TemporaryDirectory() as temp:
            args=argparse.Namespace(lr=4e-4,weight_decay=1e-4,axes=['protan','deutan'],train_severities=[1.],
                steps=20,out=Path(temp),no_tensorboard=True,no_progress=True,synthetic_ratio=.5,
                train_simulator='machado',w_structure=1.,w_fidelity=1.5,w_cvd=1.8,w_decouple=.03,
                grad_clip=1.,log_every=1,checkpoint_every=10,width=4)
            def loader(v):return DataLoader(TensorDataset(torch.full((4,3,16,16),v),torch.arange(4)),batch_size=2)
            original=exp.confusion_mask;observed=[]
            def check(inp,axis,*a,**kw):
                value=round(inp.mean().item(),1)
                if value!=.1:self.assertEqual(value, .2 if axis=='protan' else .3)
                observed.append((value,axis));return original(inp,axis,*a,**kw)
            trajectories=[]
            for _ in range(2):
                exp.seed_everything(7);model=make_model(VARIANTS['baseline'],4)
                with patch.object(exp,'confusion_mask',side_effect=check),patch.object(exp.tqdm,'write'):
                    history,_=exp.train_model(model,VARIANTS['baseline'],loader(.1),torch.device('cpu'),args,7,
                                             {'protan':loader(.2),'deutan':loader(.3)})
                trajectories.append([(r['axis'],r['synthetic_batch']) for r in history])
                checkpoint=torch.load(Path(temp)/'checkpoints/baseline_seed7_latest.pt',weights_only=True)
                self.assertEqual(checkpoint['step'],20)
                self.assertEqual(checkpoint['variant'],'baseline')
                for name,value in model.state_dict().items():
                    self.assertTrue(torch.equal(value,checkpoint['model'][name]))
            self.assertEqual(*trajectories);self.assertEqual({v for v,_ in observed},{.1,.2,.3})
