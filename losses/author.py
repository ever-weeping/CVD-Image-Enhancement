"""Released author loss; numerical conventions preserved.

``linearize=True`` switches the training-time CVD simulation from the author's
encoded-RGB convention to the physically correct sRGB-linearized Machado
simulation (same matrix family as evaluate.py). Default False keeps the
faithful author reproduction; the flag exists solely for the linearization
ablation decided 2026-11-23.
"""
import torch
import torch.nn.functional as F
from kornia.color import rgb_to_lab
from losses.ssim import SSIMLoss
from models.author import make_model
from cvd_function import cvd_matrix
from cvd_simulation import simulate_machado_cvd

def author_simulate(rgb, axis):
    # Official implementation applies the matrix directly to encoded RGB, no gamma conversion.
    matrix = rgb.new_tensor(cvd_matrix(0 if axis == 'protan' else 1, 100)).reshape(3, 3)
    return torch.einsum('bchw,dc->bdhw', rgb, matrix).clamp(0, 1)


def normalized_lab(rgb):
    return rgb_to_lab(rgb) / rgb.new_tensor([100, 128, 128]).view(1, 3, 1, 1)


class AuthorLoss(torch.nn.Module):
    def __init__(self, points=3000, linearize=False):
        super().__init__()
        self.points = points
        self.linearize = linearize
        self.ssim = SSIMLoss()

    def forward(self, original, output, axis, alpha):
        target = normalized_lab(original)
        if self.linearize:
            simulated = normalized_lab(simulate_machado_cvd(output, axis, 1.0, linearize=True))
        else:
            simulated = normalized_lab(author_simulate(output, axis))
        # Same 120 offsets and mean reduction as official contrast.py. Stack once instead
        # of repeatedly concatenating progressively larger tensors.
        h, w = target.shape[-2:]
        tc, sc = target[:, :, 5:h-5, 5:w-5], simulated[:, :, 5:h-5, 5:w-5]
        local = []
        for dy in range(-5, 6):
            for dx in range(-5, 6):
                if dy == dx == 0:
                    continue
                td = (tc - target[:, :, 5+dy:h-5+dy, 5+dx:w-5+dx]).abs().sum(1)
                sd = (sc - simulated[:, :, 5+dy:h-5+dy, 5+dx:w-5+dx]).abs().sum(1)
                local.append((td - sd).abs().mean())
        local_loss = torch.stack(local).mean()
        # Official coordinates sampled on CPU, shared across the batch.
        y, x, y2, x2 = [torch.randint(0, h, (self.points,)) for _ in range(4)]
        td = (target[:, :, x, y] - target[:, :, x2, y2]).abs().sum(1)
        sd = (simulated[:, :, x, y] - simulated[:, :, x2, y2]).abs().sum(1)
        global_loss = F.l1_loss(td, sd)
        natural = self.ssim(target * 2 - 1, normalized_lab(output) * 2 - 1)
        loss = (1-alpha) * (local_loss + global_loss) + alpha * natural
        return loss, {'loss':loss.detach().item(), 'local':local_loss.detach().item(),
                      'global':global_loss.detach().item(), 'ssim':natural.detach().item()}
