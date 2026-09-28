import unittest

import numpy as np
import torch

from cvd_metrics import batch_metrics, confusion_mask, differentiable_losses
from cvd_models import (
    VARIANTS,
    DisentangledRecolorNet,
    make_model,
    matched_baseline_width,
    parameter_count,
)
from cvd_simulation import condition_vector


class MetricAndModelTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(5)
        self.image = torch.rand(2, 3, 16, 16)

    def test_identity_metrics_have_expected_limits(self) -> None:
        mask = confusion_mask(self.image, "deutan", 1.0, "machado")
        values = batch_metrics(self.image, self.image, self.image, mask, "deutan", 1.0, "machado")
        self.assertTrue(np.allclose(values["delta_e00_mean"], 0.0, atol=1e-6))
        self.assertTrue(np.allclose(values["ssim_luma"], 1.0, atol=1e-5))
        self.assertTrue(np.allclose(values["cvd_contrast_ratio"], 1.0, atol=1e-5))
        self.assertTrue(np.allclose(values["gamut_preclip_rate"], 0.0))

    def test_parameter_matching_is_close(self) -> None:
        width = 12
        full = parameter_count(DisentangledRecolorNet(width))
        baseline = parameter_count(make_model(VARIANTS["baseline"], width))
        self.assertLess(abs(full - baseline) / full, 0.12)
        self.assertEqual(matched_baseline_width(width) % 4, 0)

    def test_ablation_returns_only_active_features(self) -> None:
        model = make_model(VARIANTS["ours_no_confusion"], 8)
        mask = confusion_mask(self.image, "protan", 0.5, "fast")
        condition = condition_vector("protan", 0.5, 2, dtype=self.image.dtype, device=self.image.device)
        output, features, raw = model(self.image, mask, condition)
        self.assertNotIn("F_confusion", features)
        self.assertIn("F_structure", features)
        self.assertIn("F_fidelity", features)
        self.assertEqual(output.shape, self.image.shape)
        self.assertEqual(raw.shape, self.image.shape)

    def test_no_mask_ablation_is_spatial_mask_invariant(self) -> None:
        model = make_model(VARIANTS["ours_no_mask"], 8).eval()
        condition = condition_vector("deutan", 0.75, 2, dtype=self.image.dtype, device=self.image.device)
        with torch.no_grad():
            output_zero, _, _ = model(self.image, torch.zeros(2, 1, 16, 16), condition)
            output_one, _, _ = model(self.image, torch.ones(2, 1, 16, 16), condition)
        self.assertTrue(torch.equal(output_zero, output_one))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_full_training_gradient_is_finite_on_mps(self) -> None:
        image = self.image.to("mps")
        model = make_model(VARIANTS["ours_full"], 8).to("mps")
        mask = confusion_mask(image, "protan", 0.75, "machado").detach()
        condition = condition_vector("protan", 0.75, 2, dtype=image.dtype, device=image.device)
        output, features, _ = model(image, mask, condition)
        from cvd_models import feature_decouple_loss

        loss, _ = differentiable_losses(
            image,
            output,
            mask,
            "protan",
            0.75,
            "machado",
            feature_decouple_loss(features),
        )
        loss.backward()
        self.assertTrue(all(
            parameter.grad is None or torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
        ))


if __name__ == "__main__":
    unittest.main()
