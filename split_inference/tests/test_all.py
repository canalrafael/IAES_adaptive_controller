"""
Unit and Integration Tests for Original SPViT Components
=========================================================
Tests the exact original controller functions and classes from the SPViT repo:
1. ARIMA class from ARIMA.py
2. calculate_flops_MSA, calculate_flops_FC, compute_DL, compute_eta from AdaptiveSplit.py
3. ViT fine-grained head/neuron split execution
"""

import unittest
import torch

from split_inference.controller import (
    ARIMA,
    calculate_flops_MSA,
    calculate_flops_FC,
    compute_DL,
    compute_eta,
)
from split_inference.model.vit_split import ViT_split, PartitionConfig


class TestOriginalSPViTComponents(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = ViT_split(
            image_size=64, patch_size=16, num_classes=5, dim=64, depth=2, heads=4, mlp_dim=128
        ).to(self.device)
        self.model.eval()
        self.dummy_input = torch.randn(2, 3, 64, 64)

    def test_original_arima_model(self):
        """Verifies exact original ARIMA class difference and fit."""
        arima = ARIMA(p=3, d=1, q=0)
        data = [10.0, 11.0, 12.0, 13.0, 14.0]
        diff = arima.difference(data, 1)
        self.assertEqual(diff, [1.0, 1.0, 1.0, 1.0])

        arima.fit(data)
        preds = arima.forecast()
        self.assertEqual(len(preds), len(data))

    def test_original_adaptive_split_math(self):
        """Verifies exact FLOPs and delay calculation functions from AdaptiveSplit.py."""
        # calculate_flops_MSA: 4 * heads * H * W * C^2 + 2 * heads * (H * W)^2
        fl_msa = calculate_flops_MSA(height=14, width=14, channels=384, num_heads=8)
        self.assertGreater(fl_msa, 0)

        # calculate_flops_FC: H * W * (output_neurons + 1) * input_neurons
        fl_fc = calculate_flops_FC(height=14, width=14, num_output_neurons=1536, num_input_neurons=384)
        self.assertGreater(fl_fc, 0)

        # compute_DL
        params = [1000.0, 2000.0]
        bw = [100.0, 200.0]
        flops_d = [1e6, 2e6]
        dl = compute_DL(params, bw, fl_msa, fl_fc, flops_d)
        self.assertEqual(len(dl), 2)
        self.assertGreater(dl[0], 0.0)

        # compute_eta
        eta = compute_eta(dl, lambda_value=0.01)
        self.assertIsInstance(eta, float)

    def test_vit_split_numerical_equivalence(self):
        """Validates that collaborative forward_split produces identical output to full forward pass."""
        with torch.no_grad():
            y_full = self.model(self.dummy_input)

            configs = [
                PartitionConfig(heads_edge=2, heads_server=2, neurons_edge=64, neurons_server=64),
                PartitionConfig(heads_edge=1, heads_server=3, neurons_edge=32, neurons_server=96),
                PartitionConfig(heads_edge=3, heads_server=1, neurons_edge=96, neurons_server=32),
            ]

            for cfg in configs:
                y_split = self.model.forward_with_partition(self.dummy_input, cfg)
                max_diff = (y_full - y_split).abs().max().item()
                self.assertLess(
                    max_diff,
                    1e-4,
                    f"Numerical mismatch with partition {cfg}: max diff = {max_diff}",
                )


if __name__ == "__main__":
    unittest.main()
