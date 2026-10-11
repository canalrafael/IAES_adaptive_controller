# SPViT: Adaptive Vision Transformer Inference via Splitting & Offloading

This branch (`adaptive-controller_SPViT`) implements the architecture and controller from the paper:

> **"SPViT: Accelerate Vision Transformer Inference on Mobile Devices via Adaptive Splitting and Offloading"**  
> *Sifan Zhao, Tongtong Liu, Hai Jin, Dezhong Yao*  
> IEEE Transactions on Mobile Computing (TMC), Vol. 24, No. 10, October 2025.  
> DOI: [10.1109/TMC.2025.3562721](https://doi.org/10.1109/TMC.2025.3562721)

---

## Key Differences from Other Branches

| Branch | Paper / Topic | Model | Controller |
| :--- | :--- | :--- | :--- |
| `main` / `adaptive-controller` | **Autodidactic Neurosurgeon (ANS)** | Partitioned ResNet-18 (CNN) | **muLinUCB** (Contextual Bandit with Ridge Regression) |
| `adaptive-controller_SPViT` *(this branch)* | **SPViT (IEEE TMC 2025)** | Partitionable Vision Transformer (ViT) | **ARIMA-V** (Auto-Regressive Capacity Forecasting + Greedy Reallocation) |

---

## Architecture Overview

1. **Partitionable ViT (`split_inference.model.vit_split`):**
   * **Head-Width Splitting:** Partitions Multi-Head Self-Attention (MSA) heads between edge device ($h_{edge}$) and server ($h_{server}$).
   * **Neuron-Width Splitting:** Partitions Feed-Forward Network (FFN/FC) neurons between edge device ($m_{edge}$) and server ($m_{server}$).
   * Zero external dependencies: pure PyTorch implementation compatible with ARM64 (Raspberry Pi) and PC.

2. **ARIMA-V Adaptive Controller (`split_inference.controller.adaptive_split`):**
   * Implements **Algorithm 1 (ARIMA-V)** from the IEEE TMC paper.
   * Maintains history of computing capacity ($\text{FLOPS}_d$) and communication bandwidth ($\text{BW}_d$) for each device.
   * Uses ARIMA time-series forecasting to predict $(n+1)$-th minibatch capacities.
   * If latency disparity exceeds threshold $\eta > 0$, greedily reallocates attention heads and MLP neurons to minimize total latency.

3. **Client / Server Pipeline (`split_inference.client` & `split_inference.server`):**
   * Framed TCP streaming protocol with timestamps ($t_e, t_s, t_r$) for precise round-trip and computation latency profiling.

---

## Usage

### 1. Run the Standalone Simulation
Simulates changing channel conditions (Nominal LAN $\to$ Low Bandwidth $\to$ Server Contention) and visualizes the ARIMA-V controller reallocating heads and neurons in real time:

```bash
python -m split_inference.run_spvit_simulation --minibatches 30 --lambda-val 0.02
```

### 2. Run the Client / Server on Hardware or LAN

**On Server (PC / Cloud GPU):**
```bash
python -m split_inference.server --host 0.0.0.0 --port 5005
```

**On Edge Client (Raspberry Pi / Mobile device):**
```bash
python -m split_inference.client --host <SERVER_IP> --port 5005 --minibatches 50
```

### 3. Run Unit Tests
```bash
python -m unittest split_inference.tests.test_all
```
