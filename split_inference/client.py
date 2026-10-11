"""
SPViT Edge Client with Online ARIMA-V Adaptive Controller
=========================================================
Runs on mobile/edge device (e.g. Raspberry Pi):
- Prepares ViT image tokens and executes edge partition
- Employs ARIMA-V controller to dynamically optimize head and neuron allocation
- Offloads collaborative partition to server and collects timing feedback
"""

import argparse
import socket
import time
from typing import Dict, List, Optional
import torch

from split_inference.model.vit_split import ViT_split
from split_inference.controller.adaptive_split import ARIMAVController, PartitionConfig
from split_inference.network.protocol import (
    recv_message,
    send_message,
    MSG_INFERENCE,
    MSG_QUIT,
)


class SPViTClient:
    """SPViT Edge Client running online ARIMA-V partitioning."""

    def __init__(
        self,
        server_host: str = "127.0.0.1",
        server_port: int = 5005,
        device_str: str = "auto",
        lambda_val: float = 0.02,
    ):
        self.server_host = server_host
        self.server_port = server_port

        if device_str == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device_str)

        # ViT Model specification
        self.image_size = 224
        self.patch_size = 16
        self.dim = 384
        self.depth = 6
        self.heads = 6
        self.mlp_dim = 1536
        self.num_classes = 10
        self.num_patches = (self.image_size // self.patch_size) ** 2

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

        # SPViT ARIMA-V Controller
        self.controller = ARIMAVController(
            device_count=2,
            head_count=self.heads,
            neuron_count=self.mlp_dim,
            num_patches=self.num_patches,
            embed_dim=self.dim,
            lambda_value=lambda_val,
        )

        self.sock: Optional[socket.socket] = None

    def connect(self) -> None:
        print(f"[SPViT Client] Connecting to server {self.server_host}:{self.server_port}...")
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
        """
        Executes one collaborative inference minibatch with adaptive partition.
        """
        assert self.sock is not None, "Client socket is not connected"

        # 1. Controller determines partition for this minibatch
        partition = self.controller.optimize_partition()

        t_total_start = time.time()

        # 2. Edge computation: prepare tokens
        t_edge_comp_start = time.time()
        with torch.no_grad():
            img_tensor = img_batch.to(self.device)
            tokens = self.model.prepare_tokens(img_tensor)

        t_edge_comp_end = time.time()
        edge_comp_time = t_edge_comp_end - t_edge_comp_start

        # 3. Offload to server with metadata
        meta = {
            "partition": partition.to_dict(),
            "batch_size": img_batch.shape[0],
        }

        t_send_start = time.time()
        send_message(self.sock, tensor=tokens, meta=meta, msg_type=MSG_INFERENCE)

        # 4. Await response
        msg_type, resp_meta, logits, tr_recv, n_bytes = recv_message(self.sock)
        t_total_end = time.time()

        round_trip_time = t_total_end - t_send_start
        server_comp_time = float(resp_meta.get("server_comp_time", 0.0))
        edge_to_server_comm = float(resp_meta.get("edge_comm_latency", round_trip_time / 2.0))
        server_to_edge_comm = max(0.0, round_trip_time - server_comp_time - edge_to_server_comm)

        total_latency = t_total_end - t_total_start

        # 5. Record step metrics into ARIMA-V controller
        self.controller.record_step(
            edge_comp_time=edge_comp_time,
            server_comp_time=server_comp_time,
            edge_to_server_comm_time=edge_to_server_comm,
            server_to_edge_comm_time=server_to_edge_comm,
        )

        return {
            "total_latency_ms": total_latency * 1000.0,
            "edge_comp_ms": edge_comp_time * 1000.0,
            "server_comp_ms": server_comp_time * 1000.0,
            "comm_latency_ms": (round_trip_time - server_comp_time) * 1000.0,
            "heads_edge": partition.heads_edge,
            "heads_server": partition.heads_server,
            "neurons_edge": partition.neurons_edge,
            "neurons_server": partition.neurons_server,
        }


def run_client(
    host: str = "127.0.0.1",
    port: int = 5005,
    minibatches: int = 30,
    batch_size: int = 1,
    lambda_val: float = 0.02,
) -> None:
    client = SPViTClient(server_host=host, server_port=port, lambda_val=lambda_val)
    client.connect()

    print(f"\n[SPViT Client] Running {minibatches} minibatches with ARIMA-V Adaptive Controller...")
    print(f"{'Batch':<6} | {'Total(ms)':<10} | {'Edge(ms)':<9} | {'Serv(ms)':<9} | {'Comm(ms)':<9} | {'Heads (E/S)':<12} | {'Neurons (E/S)':<14}")
    print("-" * 80)

    try:
        for i in range(1, minibatches + 1):
            dummy_img = torch.randn(batch_size, 3, 224, 224)
            metrics = client.infer_minibatch(dummy_img)

            heads_str = f"{metrics['heads_edge']}/{metrics['heads_server']}"
            neurons_str = f"{metrics['neurons_edge']}/{metrics['neurons_server']}"

            print(
                f"{i:<6} | {metrics['total_latency_ms']:<10.2f} | {metrics['edge_comp_ms']:<9.2f} | "
                f"{metrics['server_comp_ms']:<9.2f} | {metrics['comm_latency_ms']:<9.2f} | "
                f"{heads_str:<12} | {neurons_str:<14}"
            )
            time.sleep(0.05)
    finally:
        client.close()
        print("\n[SPViT Client] Finished inference run.")


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
