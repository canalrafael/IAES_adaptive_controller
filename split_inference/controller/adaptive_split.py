"""
SPViT Adaptive Splitting and Offloading Controller (ARIMA-V)
============================================================
Implements Algorithm 1 (ARIMA-V) from the paper:
"SPViT: Accelerate Vision Transformer Inference on Mobile Devices via Adaptive Splitting and Offloading"
(IEEE Transactions on Mobile Computing, 2025).

Features:
- Head-Width & Neuron-Width resource prediction
- ARIMA-based capacity forecasting (computation FLOPS and network BW)
- Disparity-driven greedy reallocation (eta-threshold optimization)
"""

from typing import Dict, List, Tuple
import numpy as np
from .arima import ARIMA


class PartitionConfig:
    """Represents a fine-grained or block-level ViT partition configuration."""

    def __init__(
        self,
        heads_edge: int,
        heads_server: int,
        neurons_edge: int,
        neurons_server: int,
        split_block_idx: int = 0,
    ):
        self.heads_edge = heads_edge
        self.heads_server = heads_server
        self.neurons_edge = neurons_edge
        self.neurons_server = neurons_server
        self.split_block_idx = split_block_idx

    def to_dict(self) -> Dict[str, int]:
        return {
            "heads_edge": self.heads_edge,
            "heads_server": self.heads_server,
            "neurons_edge": self.neurons_edge,
            "neurons_server": self.neurons_server,
            "split_block_idx": self.split_block_idx,
        }

    def __repr__(self) -> str:
        return (
            f"PartitionConfig(heads: [{self.heads_edge}e, {self.heads_server}s], "
            f"neurons: [{self.neurons_edge}e, {self.neurons_server}s], "
            f"block: {self.split_block_idx})"
        )


class ARIMAVController:
    """
    ARIMA-V Adaptive Splitting & Offloading Controller for SPViT.
    """

    def __init__(
        self,
        device_count: int = 2,
        head_count: int = 8,
        neuron_count: int = 2048,
        num_patches: int = 196,
        embed_dim: int = 512,
        lambda_value: float = 0.02,
        p: int = 5,
        d: int = 1,
        q: int = 0,
    ):
        self.device_count = device_count
        self.head_count = head_count
        self.neuron_count = neuron_count
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.lambda_value = lambda_value

        # Historical metrics per device
        self.seq_flops: List[List[float]] = [[] for _ in range(device_count)]
        self.seq_bw: List[List[float]] = [[] for _ in range(device_count)]

        # ARIMA models
        self.arima_flops = [ARIMA(p=p, d=d, q=q) for _ in range(device_count)]
        self.arima_bw = [ARIMA(p=p, d=d, q=q) for _ in range(device_count)]

        # Initial balanced or edge-favored partition
        init_heads_edge = head_count // 2
        init_neurons_edge = neuron_count // 2
        self.current_partition = PartitionConfig(
            heads_edge=init_heads_edge,
            heads_server=head_count - init_heads_edge,
            neurons_edge=init_neurons_edge,
            neurons_server=neuron_count - init_neurons_edge,
        )

        # Baseline per-head and per-neuron FLOPs
        self.flops_per_head = self.calculate_flops_per_head()
        self.flops_per_neuron = self.calculate_flops_per_neuron()

    def calculate_flops_per_head(self) -> float:
        """
        Calculates theoretical FLOPs for 1 attention head in an MSA layer.
        FLOPs_h = 4 * N * (D^2 / H) + 2 * N^2 * (D / H)
        """
        N = self.num_patches
        D = self.embed_dim
        H = self.head_count
        return (4.0 * N * (D**2) / H) + (2.0 * (N**2) * D / H)

    def calculate_flops_per_neuron(self) -> float:
        """
        Calculates theoretical FLOPs for 1 intermediate neuron in the FC/FFN layer.
        FLOPs_n = 2 * N * D (forward projection + backward/contraction)
        """
        N = self.num_patches
        D = self.embed_dim
        return 2.0 * N * D

    def compute_transfer_bytes(self, partition: PartitionConfig) -> List[float]:
        """
        Computes data transmission volume (Params_d in bytes) for edge and server.
        Device 0 (Edge) sends intermediate heads/tokens to Device 1 (Server).
        """
        N = self.num_patches
        D = self.embed_dim
        bytes_per_elem = 4  # float32

        # In Head-Width splitting, if heads are distributed across devices,
        # intermediate attention vectors or gathered tokens are transmitted.
        if partition.heads_edge > 0 and partition.heads_server > 0:
            edge_tx = float(N * (D * partition.heads_edge // self.head_count) * bytes_per_elem)
        else:
            edge_tx = float(N * D * bytes_per_elem)

        server_tx = float(N * D * bytes_per_elem)  # return payload
        return [edge_tx, server_tx]

    def record_step(
        self,
        edge_comp_time: float,
        server_comp_time: float,
        edge_to_server_comm_time: float,
        server_to_edge_comm_time: float,
    ) -> None:
        """
        Records actual execution metrics (te, ts, tr) from the latest minibatch.
        """
        transfer_bytes = self.compute_transfer_bytes(self.current_partition)

        # Edge (d=0)
        edge_comp_time = max(1e-5, edge_comp_time)
        edge_comm_time = max(1e-5, edge_to_server_comm_time)
        edge_flops_done = (
            self.current_partition.heads_edge * self.flops_per_head
            + self.current_partition.neurons_edge * self.flops_per_neuron
        )
        edge_flops_capacity = edge_flops_done / edge_comp_time
        edge_bw = transfer_bytes[0] / edge_comm_time

        self.seq_flops[0].append(edge_flops_capacity)
        self.seq_bw[0].append(edge_bw)

        # Server (d=1)
        server_comp_time = max(1e-5, server_comp_time)
        server_comm_time = max(1e-5, server_to_edge_comm_time)
        server_flops_done = (
            self.current_partition.heads_server * self.flops_per_head
            + self.current_partition.neurons_server * self.flops_per_neuron
        )
        server_flops_capacity = server_flops_done / server_comp_time
        server_bw = transfer_bytes[1] / server_comm_time

        self.seq_flops[1].append(server_flops_capacity)
        self.seq_bw[1].append(server_bw)

    def optimize_partition(self) -> PartitionConfig:
        """
        Executes Algorithm 1 (ARIMA-V) to predict capacities and greedily
        allocate heads and neurons to minimize total latency disparity.
        """
        pre_flops = []
        pre_bw = []

        # 1. Fit ARIMA models and forecast for each device
        for d in range(self.device_count):
            if len(self.seq_flops[d]) >= 2:
                self.arima_flops[d].fit(self.seq_flops[d])
                self.arima_bw[d].fit(self.seq_bw[d])
                pre_flops.append(self.arima_flops[d].predict_next())
                pre_bw.append(self.arima_bw[d].predict_next())
            else:
                # Default initial capacities
                pre_flops.append(1e8 if d == 0 else 1e9)
                pre_bw.append(1e7)  # 10 MB/s

        # 2. Evaluate current partition delay DL_d
        tx_bytes = self.compute_transfer_bytes(self.current_partition)
        fl_a = [
            self.current_partition.heads_edge * self.flops_per_head,
            self.current_partition.heads_server * self.flops_per_head,
        ]
        fl_f = [
            self.current_partition.neurons_edge * self.flops_per_neuron,
            self.current_partition.neurons_server * self.flops_per_neuron,
        ]

        DL = [
            (tx_bytes[d] / pre_bw[d]) + (fl_a[d] / pre_flops[d]) + (fl_f[d] / pre_flops[d])
            for d in range(self.device_count)
        ]

        # 3. Compute disparity eta = abs(max(DL) - min(DL)) - lambda
        eta = (max(DL) - min(DL)) - self.lambda_value

        # 4. If disparity exceeds threshold, perform greedy reallocation
        if eta > 0:
            alloc_heads = [0 for _ in range(self.device_count)]
            alloc_neurons = [0 for _ in range(self.device_count)]

            # Base communication delay before allocation
            DL_sim = [tx_bytes[d] / pre_bw[d] for d in range(self.device_count)]

            # Greedy Head Allocation
            for _ in range(self.head_count):
                best_d = int(np.argmin(DL_sim))
                alloc_heads[best_d] += 1
                DL_sim[best_d] += self.flops_per_head / pre_flops[best_d]

            # Greedy Neuron Allocation
            for _ in range(self.neuron_count):
                best_d = int(np.argmin(DL_sim))
                alloc_neurons[best_d] += 1
                DL_sim[best_d] += self.flops_per_neuron / pre_flops[best_d]

            self.current_partition = PartitionConfig(
                heads_edge=alloc_heads[0],
                heads_server=alloc_heads[1],
                neurons_edge=alloc_neurons[0],
                neurons_server=alloc_neurons[1],
            )

        return self.current_partition
