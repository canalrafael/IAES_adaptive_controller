"""
SPViT Server for Collaborative Vision Transformer Inference
============================================================
Listens for edge client connections and processes server-side ViT partitions:
- Executes server-allocated attention heads and MLP neurons
- Measures execution latency and returns timing metrics for ARIMA modeling
"""

import argparse
import socket
import time
import torch

from split_inference.model.vit_split import ViT_split, PartitionConfig
from split_inference.network.protocol import (
    recv_message,
    send_message,
    MSG_INFERENCE,
    MSG_PREDICTION,
    MSG_QUIT,
)


def run_server(host: str = "0.0.0.0", port: int = 5005, device_str: str = "auto") -> None:
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    print(f"[SPViT Server] Initializing on {host}:{port} using device: {device}")
    model = ViT_split(image_size=224, patch_size=16, num_classes=10, dim=384, depth=6, heads=6, mlp_dim=1536)
    model.to(device)
    model.eval()

    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server_sock.bind((host, port))
    server_sock.listen(5)
    print(f"[SPViT Server] Listening on {host}:{port}...")

    try:
        while True:
            conn, client_addr = server_sock.accept()
            print(f"[SPViT Server] Client connected from {client_addr}")
            try:
                with torch.no_grad():
                    while True:
                        msg_type, meta, tensor, tr_recv, n_bytes = recv_message(conn)
                        if msg_type == MSG_QUIT:
                            print("[SPViT Server] Received QUIT message from client.")
                            break

                        ts_edge_sent = meta.get("ts_send", tr_recv)
                        edge_comm_latency = max(0.0, tr_recv - ts_edge_sent)

                        part_dict = meta.get("partition", {})
                        partition = PartitionConfig(
                            heads_edge=part_dict.get("heads_edge", 3),
                            heads_server=part_dict.get("heads_server", 3),
                            neurons_edge=part_dict.get("neurons_edge", 768),
                            neurons_server=part_dict.get("neurons_server", 768),
                        )

                        # Server compute
                        t_comp_start = time.time()
                        if tensor is not None:
                            x_in = tensor.to(device)
                            # Execute server blocks
                            for block in model.blocks:
                                out_s_attn = block.attn.compute_heads_server(x_in, partition.heads_edge)
                                out_e_attn = block.attn.compute_heads_edge(x_in, partition.heads_edge)
                                x_attn = block.attn.combine_heads_and_project(x_in, out_e_attn, out_s_attn)

                                out_e_ff = block.ff.forward_edge(x_attn, partition.neurons_edge)
                                out_s_ff = block.ff.forward_server(x_attn, partition.neurons_edge)
                                x_in = x_attn + (out_e_ff + out_s_ff)

                            out_norm = model.norm(x_in)
                            logits = model.mlp_head(out_norm[:, 0])
                        else:
                            logits = torch.zeros(1, 10, device=device)

                        t_comp_end = time.time()
                        server_comp_time = t_comp_end - t_comp_start

                        resp_meta = {
                            "server_comp_time": server_comp_time,
                            "edge_comm_latency": edge_comm_latency,
                        }
                        send_message(conn, logits, resp_meta, msg_type=MSG_PREDICTION)

            except (ConnectionError, BrokenPipeError, ConnectionResetError) as e:
                print(f"[SPViT Server] Client disconnected: {e}")
            finally:
                conn.close()

    except KeyboardInterrupt:
        print("\n[SPViT Server] Shutting down.")
    finally:
        server_sock.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SPViT Inference Server")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="Binding host")
    parser.add_argument("--port", type=int, default=5005, help="Port to listen on")
    parser.add_argument("--device", type=str, default="auto", help="Compute device (cuda/cpu/auto)")
    args = parser.parse_args()

    run_server(host=args.host, port=args.port, device_str=args.device)
