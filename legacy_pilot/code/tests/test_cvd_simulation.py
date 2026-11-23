import unittest

import torch

from cvd_simulation import (
    condition_vector,
    cvd_condition_label,
    simulate_cvd,
)


class CvdSimulationTests(unittest.TestCase):
    def setUp(self) -> None:
        torch.manual_seed(3)
        self.image = torch.rand(2, 3, 8, 8)

    def test_terminology_boundary(self) -> None:
        self.assertEqual(cvd_condition_label("protan", 0.5), "protanomaly")
        self.assertEqual(cvd_condition_label("protanopia", 1.0), "protanopia")
        self.assertEqual(cvd_condition_label("deutan", 0.25), "deuteranomaly")
        self.assertEqual(cvd_condition_label("deuteranomaly", 1.0), "deuteranopia")

    def test_zero_severity_is_identity(self) -> None:
        for simulator in ("machado", "fast", "vienot", "brettel"):
            output = simulate_cvd(self.image, "deutan", 0.0, simulator)
            self.assertTrue(torch.allclose(output, self.image, atol=2e-6), simulator)

    def test_fast_simulator_responds_to_severity(self) -> None:
        mild = simulate_cvd(self.image, "protan", 0.25, "fast")
        endpoint = simulate_cvd(self.image, "protan", 1.0, "fast")
        self.assertGreater(float((mild - endpoint).abs().mean()), 1e-3)

    def test_all_backends_are_differentiable_and_bounded(self) -> None:
        for simulator in ("machado", "fast", "vienot", "brettel"):
            image = self.image.clone().requires_grad_(True)
            output = simulate_cvd(image, "protan", 0.7, simulator)
            self.assertEqual(output.shape, image.shape)
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)
            output.mean().backward()
            self.assertIsNotNone(image.grad)

    def test_condition_vector(self) -> None:
        vector = condition_vector("deutan", 0.75, 2, dtype=torch.float32, device=torch.device("cpu"))
        expected = torch.tensor([[0.0, 1.0, 0.75], [0.0, 1.0, 0.75]])
        self.assertTrue(torch.equal(vector, expected))

    @unittest.skipUnless(torch.backends.mps.is_available(), "MPS is unavailable")
    def test_cpu_mps_numerical_parity(self) -> None:
        cpu = simulate_cvd(self.image, "protan", 1.0, "machado")
        mps = simulate_cvd(self.image.to("mps"), "protan", 1.0, "machado").cpu()
        self.assertTrue(torch.allclose(cpu, mps, atol=5e-5, rtol=5e-5))


if __name__ == "__main__":
    unittest.main()
