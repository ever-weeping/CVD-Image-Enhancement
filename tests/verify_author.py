"""Numerical parity with released loss routines, then real full-resolution GPU step."""
import json
import sys
import time
from pathlib import Path

import torch

BASE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(BASE))
sys.path.insert(0, str(BASE / 'vendor'))
from models.author import make_model
from losses.author import AuthorLoss, normalized_lab, author_simulate
from contrast import calculate_contrast_oneimg_l1, global_contrast_img_l1
from cvd_function import cvd_simulation_tensors

torch.set_num_threads(4)
torch.cuda.set_per_process_memory_fraction(.8)
torch.manual_seed(7)
x = torch.rand(1, 3, 256, 256, device='cuda')
y = torch.rand_like(x, requires_grad=True)
criterion = AuthorLoss().cuda()
for axis, idx in [('protan', 0), ('deutan', 1)]:
    assert torch.allclose(author_simulate(y, axis), cvd_simulation_tensors(y, idx, 100), atol=2e-7)
    torch.manual_seed(91)
    loss, logs = criterion(x, y, axis, .5)
    grad = torch.autograd.grad(loss, y)[0]
    a, b = normalized_lab(x), normalized_lab(author_simulate(y, axis))
    local = (calculate_contrast_oneimg_l1(a, 5) - calculate_contrast_oneimg_l1(b, 5)).abs().mean()
    torch.manual_seed(91)
    g1, g2 = global_contrast_img_l1(a, b, 3000)
    glob = (g1 - g2).abs().mean()
    ssim = criterion.ssim(a * 2 - 1, normalized_lab(y) * 2 - 1)
    official = .5 * (local + glob) + .5 * ssim
    official_grad = torch.autograd.grad(official, y)[0]
    assert torch.allclose(loss, official, atol=1e-6), (loss, official)
    assert torch.allclose(grad, official_grad, atol=1e-6, rtol=1e-4)
    print(axis, 'official loss and gradient parity OK', logs, flush=True)
del y, grad, official_grad, loss, official, a, b, local, glob, ssim
model = make_model().cuda()
optim = torch.optim.Adam(model.parameters(), lr=.0002, betas=(.5, .999))
x = torch.rand(4, 3, 256, 256, device='cuda')
for step in range(3):
    begin = time.monotonic()
    optim.zero_grad(set_to_none=True)
    y = (model(x * 2 - 1) + 1) / 2
    loss, logs = criterion(x, y, 'deutan', 0 if step == 0 else .5)
    loss.backward()
    assert torch.isfinite(loss)
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    optim.step()
    torch.cuda.synchronize()
    print(json.dumps(dict(step=step, seconds=time.monotonic() - begin,
                          max_memory_gb=torch.cuda.max_memory_allocated() / 1e9,
                          params=sum(p.numel() for p in model.parameters()), **logs)), flush=True)
