"""
SPViT Edge Client using Original SPViT Controller Functions
============================================================
Runs on mobile/edge device (e.g. Raspberry Pi):
- Employs original ARIMA class and AdaptiveSplit functions (calculate_flops_MSA,
  calculate_flops_FC, compute_DL, compute_eta)
- Prepares ViT image tokens and executes edge partition
- Offloads collaborative partition to server and collects timing feedback
"""

import argparse
import socket
import time
from typing import Dict, List, Optional
import numpy as np
import torch

from split_inference.model.vit_split import ViT_split, PartitionConfig
from split_inference.controller import (
    ARIMA,
    calculate_flops_MSA,
    calculate_flops_FC,
    compute_DL,
    compute_eta,
)
from split_inference.network.protocol import (
    recv_message,
    send_message,
    MSG_INFERENCE,
    MSG_QUIT,
)


class SPViTControllerRunner:
    """
    Online runner that coordinates the original SPViT ARIMA and AdaptiveSplit functions.
    """

    def __init__(
        self,
        device_count: int = 2,
        head_count: int = 6,
        neuron_count: int = 1536,
        img_h: int = 14,
        img_w: int = 14,
        channels: int = 384,
        lambda_val: float = 0.02,
    ):
        self.device_count = device_count
        self.head_count = head_count
        self.neuron_count = neuron_count
        self.img_h = img_h
        self.img_w = img_w
        self.channels = channels
        self.lambda_val = lambda_val

        # Lists to store capacity history as in original AdaptiveSplit
        self.seq_flops = [[] for _ in range(device_count)]
        self.seq_bw = [[] for _ in range(device_count)]
        self.arima_model = ARIMA(p=5, d=1, q=0)

        self.current_partition = PartitionConfig(
            heads_edge=head_count // 2,
            heads_server=head_count - head_count // 2,
            neurons_edge=neuron_count // 2,
            neurons_server=neuron_count - neuron_count // 2,
        )

        # Baseline single-unit FLOPs using original formulas
        self.flops_h = calculate_flops_MSA(img_h, img_w, channels, 1)
        self.flops_n = calculate_flops_FC(img_h, img_w, neuron_count, 1)

    def record_step(
        self,
        edge_comp: float,
        server_comp: float,
        edge_comm: float,
        server_comm: float,
    ) -> None:
        params_d = [
            float(self.img_h * self.img_w * self.channels * 4),
            float(self.img_h * self.img_w * self.channels * 4),
        ]
        bw_edge = params_d[0] / max(1e-5, edge_comm)
        bw_server = params_d[1] / max(1e-5, server_comm)

        fl_a_edge = self.current_partition.heads_edge * self.flops_h
        fl_f_edge = self.current_partition.neurons_edge * self.flops_n
        flops_edge = (fl_a_edge + fl_f_edge) / max(1e-5, edge_comp)

        fl_a_server = self.current_partition.heads_server * self.flops_h
        fl_f_server = self.current_partition.neurons_server * self.flops_n
        flops_server = (fl_a_server + fl_f_server) / max(1e-5, server_comp)

        self.seq_flops[0].append(flops_edge)
        self.seq_bw[0].append(bw_edge)
        self.seq_flops[1].append(flops_server)
        self.seq_bw[1].append(bw_server)

    def optimize_partition(self) -> PartitionConfig:
        params_d = [
            float(self.img_h * self.img_w * self.channels * 4),
            float(self.img_h * self.img_w * self.channels * 4),
        ]

        # Use original ARIMA.fit and predict
        pre_flops_d = []
        pre_bw_d = []
        for d in range(self.device_count):
            if len(self.seq_flops[d]) >= 2:
                self.arima_model.fit(self.seq_flops[d])
                preds = self.arima_model.forecast()
                pre_flops_d.append(max(1e5, float(preds[-1]) if preds else 1e8))

                self.arima_model.fit(self.seq_bw[d])
                preds_bw = self.arima_model.forecast()
                pre_bw_d.append(max(1e5, float(preds_bw[-1]) if preds_bw else 1e7))
            else:
                pre_flops_d.append(1e8 if d == 0 else 5e8)
                pre_bw_d.append(1e7)

        # Compute DL via original compute_DL
        fl_a = self.head_count * self.flops_h
        fl_f = self.neuron_count * self.flops_n
        dl = compute_DL(params_d, pre_bw_d, fl_a, fl_f, pre_flops_d)

        # Compute eta via original compute_eta
        eta = compute_eta(dl, self.lambda_val)

        # Adjust DL based on eta (original AdaptiveSplit greedy loop)
        if eta > 0:
            alloc_heads = [0 for _ in range(self.device_count)]
            alloc_neurons = [0 for _ in range(self.device_count)]
            dl_work = [params_d[d] / pre_bw_d[d] for d in range(self.device_count)]

            for _ in range(self.head_count):
                best_d = int(np.argmin(dl_work))
                alloc_heads[best_d] += 1
                dl_work[best_d] += self.flops_h / pre_flops_d[best_d]

            for _ in range(self.neuron_count):
                best_d = int(np.argmin(dl_work))
                alloc_neurons[best_d] += 1
                dl_work[best_d] += self.flops_n / pre_flops_d[best_d]

            self.current_partition = PartitionConfig(
                heads_edge=alloc_heads[0],
                heads_server=alloc_heads[1],
                neurons_edge=alloc_neurons[0],
                neurons_server=alloc_neurons[1],
            )

        return self.current_partition


class SPViTClient:
    """SPViT Edge Client connecting to server."""

    def __init__(
        self,
        server_host: str = "127.0.0.1",
        server_port: int = 5005,
        device_str: str = "auto",
        lambda_val: float = 0.02,
    ):
        self.server_host = server_host
        self.server_port = server_port
        self.device = torch.device(
            "cuda" if (device_str == "auto" and torch.cuda.is_available()) else "cpu"
        )

        self.image_size = 224
        self.patch_size = 16
        self.dim = 384
        self.depth = 6
        self.heads = 6
        self.mlp_dim = 1536
        self.num_classes = 10

        self.model = ViT_split(
            image_size=self.image_size,
            patch_size=self.patch_size,
            num_classes=self.num_classes,
            dim=self.dim,
            depth=self.depth,
            heads=self.heads,
            mlp_dim=self.mlp_dim,
        ).to(self.device)
        self.model.eval()

        self.controller = SPViTControllerRunner(
            device_count=2,
            head_count=self.heads,
            neuron_count=self.mlp_dim,
            img_h=self.image_size // self.patch_size,
            img_w=self.image_size // self.patch_size,
            channels=self.dim,
            lambda_val=lambda_val,
        )

        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        print(f"[SPViT Client] Connecting to {self.server_host}:{self.server_port}...")
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.connect((self.server_host, self.server_port))
        print("[SPViT Client] Connected successfully.")

    def close(self) -> None:
        if self.sock:
            try:
                send_message(self.sock, msg_type=MSG_QUIT)
            except Exception:
                pass
            self.sock.close()
            self.sock = None

    def infer_minibatch(self, img_batch: torch.Tensor) -> Dict[str, float]:
        assert self.sock is not None, "Client socket is not connected"

        partition = self.controller.optimize_partition()
        t_total_start = time.time()

        t_edge_start = time.time()
        with torch.no_grad():
            img_tensor = img_batch.to(self.device)
            tokens = self.model.prepare_tokens(img_tensor)
        edge_comp = time.time() - t_edge_start

        meta = {"partition": partition.to_dict(), "batch_size": img_batch.shape[0]}
        t_send_start = time.time()
        send_message(self.sock, tensor=tokens, meta=meta, msg_type=MSG_INFERENCE)

        msg_type, resp_meta, logits, tr_recv, n_bytes = recv_message(self.sock)
        t_total_end = time.time()

        rtt = t_total_end - t_send_start
        server_comp = float(resp_meta.get("server_comp_time", 0.0))
        edge_comm = float(resp_meta.get("edge_comm_latency", rtt / 2.0))
        server_comm = max(0.0, rtt - server_comp - edge_comm)

        self.controller.record_step(edge_comp, server_comp, edge_comm, server_comm)

        return {
            "total_latency_ms": (t_total_end - t_total_start) * 1000.0,
            "edge_comp_ms": edge_comp * 1000.0,
            "server_comp_ms": server_comp * 1000.0,
            "comm_latency_ms": (rtt - server_comp) * 1000.0,
            "heads_edge": partition.heads_edge,
            "heads_server": partition.heads_server,
            "neurons_edge": partition.neurons_edge,
            "neurons_server": partition.neurons_server,
        }


def run_client(
    host: str = "127.0.0.1",
    port: int = 5005,
    minibatches: int = 25,
    batch_size: int = 1,
    lambda_val: float = 0.02,
) -> None:
    client = SPViTClient(server_host=host, server_port=port, lambda_val=lambda_val)
    client.connect()

    print(f"\n[SPViT Client] Running {minibatches} minibatches with Original SPViT Controller...")
    print(f"{'Batch':<6} | {'Total(ms)':<10} | {'Edge(ms)':<9} | {'Serv(ms)':<9} | {'Comm(ms)':<9} | {'Heads (E/S)':<12} | {'Neurons (E/S)':<14}")
    print("-" * 80)

    try:
        for i in range(1, minibatches + 1):
            dummy_img = torch.randn(batch_size, 3, 224, 224)
            m = client.infer_minibatch(dummy_img)
            heads_str = f"{m['heads_edge']}/{m['heads_server']}"
            neurons_str = f"{m['neurons_edge']}/{m['neurons_server']}"
            print(
                f"{i:<6} | {m['total_latency_ms']:<10.2f} | {m['edge_comp_ms']:<9.2f} | "
                f"{m['server_comp_ms']:<9.2f} | {m['comm_latency_ms']:<9.2f} | "
                f"{heads_str:<12} | {neurons_str:<14}"
            )
            time.sleep(0.05)
    finally:
        client.close()
        print("\n[SPViT Client] Finished.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPViT Edge Client")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="Server host")
    parser.add_argument("--port", type=int, default=5005, help="Server port")
    parser.add_argument("--minibatches", type=int, default=25, help="Number of minibatches")
    parser.add_argument("--batch-size", type=int, default=1, help="Minibatch size")
    parser.add_argument("--lambda-val", type=float, default=0.02, help="ARIMA-V lambda threshold")
    args = parser.parse_args()

    run_client(
        host=args.host,
        port=args.port,
        minibatches=args.minibatches,
        batch_size=args.batch_size,
        lambda_val=args.lambda_val,
    )
