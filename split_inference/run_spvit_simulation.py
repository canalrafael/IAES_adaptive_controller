"""
SPViT Standalone Simulation & Controller Benchmark
==================================================
Simulates dynamic edge-server collaborative inference with varying network
bandwidth and edge compute pressure, validating the online adaptation of
the ARIMA-V controller without requiring a physical network socket.
"""

import argparse
import random
import time
import numpy as np
import torch

from split_inference.model.vit_split import ViT_split
from split_inference.controller.adaptive_split import ARIMAVController, PartitionConfig


def run_simulation(minibatches: int = 40, lambda_val: float = 0.02) -> None:
    print("=" * 80)
    print("SPViT: ADAPTIVE VISION TRANSFORMER OFFLOADING SIMULATION (ARIMA-V)")
    print("=" * 80)

    # Initialize ViT Model & ARIMA-V Controller
    heads = 6
    mlp_dim = 1536
    dim = 384
    patches = 196

    model = ViT_split(
        image_size=224, patch_size=16, num_classes=10, dim=dim, depth=6, heads=heads, mlp_dim=mlp_dim
    )
    model.eval()

    controller = ARIMAVController(
        device_count=2,
        head_count=heads,
        neuron_count=mlp_dim,
        num_patches=patches,
        embed_dim=dim,
        lambda_value=lambda_val,
    )

    print(f"Total Attention Heads: {heads} | Total MLP Neurons: {mlp_dim} | Lambda: {lambda_val}")
    print(f"{'Batch':<6} | {'Condition':<15} | {'Edge Comp':<11} | {'Serv Comp':<11} | {'Comm Lat':<10} | {'Heads (E/S)':<12} | {'Neurons (E/S)':<14}")
    print("-" * 88)

    # Environments: Phase 1: High BW -> Phase 2: Bandwidth drop -> Phase 3: Server compute bottleneck
    for n in range(1, minibatches + 1):
        if n <= 12:
            condition = "Nominal LAN"
            base_edge_speed = 1.0  # multiplier
            base_bw_mbps = 50.0
            base_server_speed = 3.5
        elif n <= 25:
            condition = "Low Bandwidth"
            base_edge_speed = 1.0
            base_bw_mbps = 4.0  # bandwidth drop
            base_server_speed = 3.5
        else:
            condition = "Server Contention"
            base_edge_speed = 1.0
            base_bw_mbps = 20.0
            base_server_speed = 0.8  # server slowed down

        # 1. Controller determines partition
        partition = controller.optimize_partition()

        # 2. Simulate hardware latency according to allocated workload
        fl_h = controller.flops_per_head
        fl_m = controller.flops_per_neuron

        edge_flops = partition.heads_edge * fl_h + partition.neurons_edge * fl_m
        server_flops = partition.heads_server * fl_h + partition.neurons_server * fl_m

        edge_comp = (edge_flops / (1e8 * base_edge_speed)) * (1.0 + random.uniform(-0.05, 0.05))
        server_comp = (server_flops / (1e8 * base_server_speed)) * (1.0 + random.uniform(-0.05, 0.05))

        tx_bytes = controller.compute_transfer_bytes(partition)[0]
        comm_lat = (tx_bytes / (base_bw_mbps * 1e6 / 8.0)) * (1.0 + random.uniform(-0.05, 0.05))

        # 3. Feed observed timings to ARIMA-V controller
        controller.record_step(
            edge_comp_time=edge_comp,
            server_comp_time=server_comp,
            edge_to_server_comm_time=comm_lat / 2.0,
            server_to_edge_comm_time=comm_lat / 2.0,
        )

        heads_str = f"{partition.heads_edge}/{partition.heads_server}"
        neurons_str = f"{partition.neurons_edge}/{partition.neurons_server}"

        print(
            f"{n:<6} | {condition:<15} | {edge_comp * 1000:<11.2f} | {server_comp * 1000:<11.2f} | "
            f"{comm_lat * 1000:<10.2f} | {heads_str:<12} | {neurons_str:<14}"
        )

    print("\n[Simulation Complete] The ARIMA-V controller dynamically redistributed attention heads")
    print("and MLP neurons based on real-time channel capacity and device compute capability.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPViT Standalone Simulation")
    parser.add_argument("--minibatches", type=int, default=35, help="Number of minibatches to simulate")
    parser.add_argument("--lambda-val", type=float, default=0.02, help="Disparity threshold lambda")
    args = parser.parse_args()

    run_simulation(minibatches=args.minibatches, lambda_val=args.lambda_val)
