"""
TCP Streaming Protocol for SPViT Collaborative Inference
=========================================================
Implements framed socket communication for transferring intermediate
Vision Transformer tokens/activations between the Edge Client and Server,
with precise timestamp tagging (te, ts, tr) for ARIMA capacity estimation.
"""

import io
import json
import socket
import struct
import time
from typing import Any, Dict, Optional, Tuple
import torch

MAGIC_HEADER = b"SPVT"
MAGIC_RESP = b"SPVR"
MSG_INFERENCE = 1
MSG_PREDICTION = 2
MSG_QUIT = 3


def send_exact(sock: socket.socket, data: bytes) -> None:
    """Ensures all bytes in buffer are completely transmitted."""
    total_sent = 0
    while total_sent < len(data):
        sent = sock.send(data[total_sent:])
        if sent == 0:
            raise ConnectionError("Socket connection closed during send.")
        total_sent += sent


def recv_exact(sock: socket.socket, num_bytes: int) -> bytes:
    """Blocks until exactly num_bytes are received."""
    chunks = []
    bytes_recd = 0
    while bytes_recd < num_bytes:
        chunk = sock.recv(min(num_bytes - bytes_recd, 65536))
        if not chunk:
            raise ConnectionError("Socket connection closed during recv.")
        chunks.append(chunk)
        bytes_recd += len(chunk)
    return b"".join(chunks)


def send_message(
    sock: socket.socket,
    tensor: Optional[torch.Tensor] = None,
    meta: Optional[Dict[str, Any]] = None,
    msg_type: int = MSG_INFERENCE,
) -> int:
    """
    Sends framed message containing optional tensor and JSON metadata.
    Header format: [Magic(4B)][MsgType(1B)][MetaLen(4B)][TensorLen(4B)]
    """
    meta_dict = meta or {}
    meta_dict["ts_send"] = time.time()
    meta_bytes = json.dumps(meta_dict).encode("utf-8")

    tensor_bytes = b""
    if tensor is not None:
        buf = io.BytesIO()
        torch.save(tensor.cpu(), buf)
        tensor_bytes = buf.getvalue()

    header = MAGIC_HEADER + struct.pack("!BII", msg_type, len(meta_bytes), len(tensor_bytes))
    payload = header + meta_bytes + tensor_bytes
    send_exact(sock, payload)
    return len(payload)


def recv_message(
    sock: socket.socket,
) -> Tuple[int, Dict[str, Any], Optional[torch.Tensor], float, int]:
    """
    Receives framed message.
    Returns: (msg_type, meta_dict, tensor, tr_recv_time, total_bytes)
    """
    tr_recv = time.time()
    header = recv_exact(sock, 13)
    magic = header[:4]
    if magic != MAGIC_HEADER and magic != MAGIC_RESP:
        raise ValueError(f"Invalid SPViT protocol magic: {magic}")

    msg_type, meta_len, tensor_len = struct.unpack("!BII", header[4:13])

    meta_bytes = recv_exact(sock, meta_len) if meta_len > 0 else b"{}"
    meta_dict = json.loads(meta_bytes.decode("utf-8"))

    tensor = None
    if tensor_len > 0:
        tensor_bytes = recv_exact(sock, tensor_len)
        buf = io.BytesIO(tensor_bytes)
        tensor = torch.load(buf, weights_only=True)

    total_bytes = 13 + meta_len + tensor_len
    return msg_type, meta_dict, tensor, tr_recv, total_bytes
