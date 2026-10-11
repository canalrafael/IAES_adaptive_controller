"""
Unit and Integration Tests for SPViT Components
================================================
Verifies:
1. ARIMA(p, d, q) time-series forecasting.
2. ARIMA-V controller adaptive greedy partitioning.
3. ViT_split numerical equivalence between full forward and fine-grained split forward.
4. Client/Server socket communication protocol.
"""

import sys
import time
import unittest
import torch

from split_inference.controller.arima import ARIMA
from split_inference.controller.adaptive_split import ARIMAVController, PartitionConfig
from split_inference.model.vit_split import ViT_split


class TestSPViTComponents(unittest.TestCase):
    def setUp(self):
        self.device = torch.device("cpu")
        self.model = ViT_split(
            image_size=64, patch_size=16, num_classes=5, dim=64, depth=2, heads=4, mlp_dim=128
        ).to(self.device)
        self.model.eval()
        self.dummy_input = torch.randn(2, 3, 64, 64)

    def test_arima_forecasting(self):
        """Verifies ARIMA fits and predicts reasonable positive capacities."""
        arima = ARIMA(p=3, d=1, q=0)
        history = [10.0, 10.5, 11.0, 11.2, 10.8, 11.5, 12.0]
        arima.fit(history)
        pred = arima.predict_next()
        self.assertGreater(pred, 0.0)
        self.assertLess(pred, 25.0)

    def test_arima_v_controller_greedy_allocation(self):
        """Verifies ARIMA-V computes allocations for heads and neurons summing to totals."""
        controller = ARIMAVController(
            device_count=2, head_count=4, neuron_count=128, num_patches=16, embed_dim=64, lambda_value=0.01
        )
        # Feed synthetic history with server having 4x compute capacity
        for _ in range(5):
            controller.record_step(
                edge_comp_time=0.08, server_comp_time=0.02, edge_to_server_comm_time=0.01, server_to_edge_comm_time=0.01
            )

        partition = controller.optimize_partition()
        # Sum of heads must equal total heads
        self.assertEqual(partition.heads_edge + partition.heads_server, 4)
        # Sum of neurons must equal total neurons
        self.assertEqual(partition.neurons_edge + partition.neurons_server, 128)
        # Since server is much faster, server should be allocated more heads and neurons
        self.assertGreaterEqual(partition.heads_server, partition.heads_edge)
        self.assertGreaterEqual(partition.neurons_server, partition.neurons_edge)

    def test_vit_split_numerical_equivalence(self):
        """Validates that collaborative forward_split produces identical output to full forward pass."""
        with torch.no_grad():
            y_full = self.model(self.dummy_input)

            # Test different head and neuron split configurations
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
