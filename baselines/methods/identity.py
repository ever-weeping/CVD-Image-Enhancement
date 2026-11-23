"""Identity "method" -- harness calibration, not a baseline.

Passes the input through unchanged. Expected behaviour on the unified
metrics (protocol v3): tcc_gain == 0 exactly (same images both sides),
ssim == 1 exactly, cd_mean == 0 exactly. Any deviation means the harness,
the metric seeding or the batch plumbing is broken -- run this first
whenever run_eval.py or metrics.py changes.
"""
import torch

DEFAULTS = {}
PARAM_DOC = {}


def recolor(inp: torch.Tensor, axis: str, **params) -> torch.Tensor:
    assert axis in ('protan', 'deutan')
    return inp.clone()


def selftest() -> None:
    from metrics import batch_metrics
    torch.manual_seed(0)
    inp = torch.rand(2, 3, 64, 64)
    out = recolor(inp, 'protan')
    assert torch.equal(inp, out)
    rows = batch_metrics(inp, out, 'protan', 1.0, 'machado', index_offset=0, seed=7)
    assert all(abs(v) < 1e-6 for v in rows['cd_mean']), rows['cd_mean']
    assert all(abs(v - 1.0) < 1e-4 for v in rows['ssim_naturalness']), rows['ssim_naturalness']
    for a, b in zip(rows['tcc_input_sim'], rows['tcc_output_sim']):
        assert abs(a - b) < 1e-6, (a, b)
    print('identity selftest: PASS')


if __name__ == '__main__':
    selftest()
