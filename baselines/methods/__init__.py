"""Baseline method registry (dynamic import).

Each module <name>.py must expose:
    DEFAULTS: dict[str, scalar]        # every tunable knob with its default
    PARAM_DOC: dict[str, str]          # one-line meaning incl. 台账/欠定 provenance
    recolor(inp: torch.Tensor, axis: str, **params) -> torch.Tensor
        batch-aware, sRGB [0,1] CHW in and out, same device/dtype as input;
        axis in {'protan','deutan'}; severity is fixed to 1.0 by protocol.
    selftest() -> None                 # tier-A invariants, small CPU images,
                                       # raises AssertionError with a message.
Optional: NO_GRAD = False (methods doing per-image gradient optimization
    manage their own autograd; the harness then does not disable grad).
"""
import importlib


def load_method(name: str):
    if '/' in name or name.startswith('_'):
        raise ValueError(f'Invalid method name: {name!r}')
    return importlib.import_module(f'baselines.methods.{name}')
