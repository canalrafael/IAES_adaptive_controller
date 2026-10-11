"""
SPViT Standalone Simulation & Controller Benchmark
==================================================
Simulates dynamic edge-server collaborative inference with varying network
bandwidth and edge compute pressure, exercising the original SPViT controller.
"""

import argparse
import random
import time
import numpy as np
import torch

from split_inference.model.vit_split import ViT_split, PartitionConfig
from split_inference.client import SPViTControllerRunner


def run_simulation(minibatches: int = 30, lambda_val: float = 0.02) -> None:
    print("=" * 80)
    print("SPViT: ADAPTIVE VISION TRANSFORMER SIMULATION (ORIGINAL CONTROLLER)")
    print("=" * 80)

    heads = 6
    mlp_dim = 1536
    dim = 384
    patches = 196

    model = ViT_split(
        image_size=224, patch_size=16, num_classes=10, dim=dim, depth=6, heads=heads, mlp_dim=mlp_dim
    )
    model.eval()

    controller = SPViTControllerRunner(
        device_count=2,
        head_count=heads,
        neuron_count=mlp_dim,
        img_h=14,
        img_w=14,
        channels=dim,
        lambda_val=lambda_val,
    )

    print(f"Total Attention Heads: {heads} | Total MLP Neurons: {mlp_dim} | Lambda: {lambda_val}")
    print(f"{'Batch':<6} | {'Condition':<15} | {'Edge Comp':<11} | {'Serv Comp':<11} | {'Comm Lat':<10} | {'Heads (E/S)':<12} | {'Neurons (E/S)':<14}")
    print("-" * 88)

    for n in range(1, minibatches + 1):
        if n <= 10:
            condition = "Nominal LAN"
            base_edge_speed = 1.0
            base_bw_mbps = 50.0
            base_server_speed = 3.5
        elif n <= 20:
            condition = "Low Bandwidth"
            base_edge_speed = 1.0
            base_bw_mbps = 4.0
            base_server_speed = 3.5
        else:
            condition = "Server Contention"
            base_edge_speed = 1.0
            base_bw_mbps = 20.0
            base_server_speed = 0.8

        partition = controller.optimize_partition()

        fl_h = controller.flops_h
        fl_m = controller.flops_n

        edge_flops = partition.heads_edge * fl_h + partition.neurons_edge * fl_m
        server_flops = partition.heads_server * fl_h + partition.neurons_server * fl_m

        edge_comp = (edge_flops / (1e8 * base_edge_speed)) * (1.0 + random.uniform(-0.05, 0.05))
        server_comp = (server_flops / (1e8 * base_server_speed)) * (1.0 + random.uniform(-0.05, 0.05))

        tx_bytes = 14 * 14 * dim * 4
        comm_lat = (tx_bytes / (base_bw_mbps * 1e6 / 8.0)) * (1.0 + random.uniform(-0.05, 0.05))

        controller.record_step(
            edge_comp=edge_comp,
            server_comp=server_comp,
            edge_comm=comm_lat / 2.0,
            server_comm=comm_lat / 2.0,
        )

        heads_str = f"{partition.heads_edge}/{partition.heads_server}"
        neurons_str = f"{partition.neurons_edge}/{partition.neurons_server}"

        print(
            f"{n:<6} | {condition:<15} | {edge_comp * 1000:<11.2f} | {server_comp * 1000:<11.2f} | "
            f"{comm_lat * 1000:<10.2f} | {heads_str:<12} | {neurons_str:<14}"
        )

    print("\n[Simulation Complete] Tested original SPViT controller with dynamic offloading.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPViT Standalone Simulation")
    parser.add_argument("--minibatches", type=int, default=30, help="Number of minibatches to simulate")
    parser.add_argument("--lambda-val", type=float, default=0.02, help="Disparity threshold lambda")
    args = parser.parse_args()

    run_simulation(minibatches=args.minibatches, lambda_val=args.lambda_val)
