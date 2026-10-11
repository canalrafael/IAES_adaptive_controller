# SPViT: Adaptive Vision Transformer Inference via Splitting & Offloading

This branch (`adaptive-controller_SPViT`) uses the **original, unmodified controller** from the official SPViT paper and repository:

> **"SPViT: Accelerate Vision Transformer Inference on Mobile Devices via Adaptive Splitting and Offloading"**  
> *Sifan Zhao, Tongtong Liu, Hai Jin, Dezhong Yao*  
> IEEE Transactions on Mobile Computing (TMC), Vol. 24, No. 10, October 2025.  
> Official Code: [https://github.com/CGCL-codes/SPViT](https://github.com/CGCL-codes/SPViT)

---

## Unmodified Original Controller Components

In [`split_inference/controller/`](file:///c:/Users/canal/Desktop/Doutorado%20-%20PPGCC/adaptive%20split%20point%20controller/IAES/split_inference/controller):
1. **[`ARIMA.py`](file:///c:/Users/canal/Desktop/Doutorado%20-%20PPGCC/adaptive%20split%20point%20controller/IAES/split_inference/controller/ARIMA.py):**
   * The original `ARIMA` class implementing difference, inverse difference, fitting, and forecasting.
2. **[`AdaptiveSplit.py`](file:///c:/Users/canal/Desktop/Doutorado%20-%20PPGCC/adaptive%20split%20point%20controller/IAES/split_inference/controller/AdaptiveSplit.py):**
   * The original functions:
     * `adaptive_vit_inference_offloading`
     * `calculate_flops_MSA`
     * `calculate_flops_FC`
     * `compute_DL`
     * `compute_eta`

---

## Model & Execution Pipeline

* **Vision Transformer ([`split_inference.model.vit_split`](file:///c:/Users/canal/Desktop/Doutorado%20-%20PPGCC/adaptive%20split%20point%20controller/IAES/split_inference/model/vit_split.py)):**
  Partitionable ViT supporting Head-Width splitting in Multi-Head Self-Attention and Neuron-Width splitting in Feed-Forward layers.
* **Edge Client & Server ([`split_inference.client`](file:///c:/Users/canal/Desktop/Doutorado%20-%20PPGCC/adaptive%20split%20point%20controller/IAES/split_inference/client.py) & [`split_inference.server`](file:///c:/Users/canal/Desktop/Doutorado%20-%20PPGCC/adaptive%20split%20point%20controller/IAES/split_inference/server.py)):**
  Framed TCP streaming socket protocol with timestamp tagging ($t_e, t_s, t_r$) for online latency measurement and feedback into the ARIMA controller.

---

## Usage

### 1. Run the Standalone Simulation
```bash
python -m split_inference.run_spvit_simulation --minibatches 30
```

### 2. Run the Client / Server on Hardware or LAN

**On Server:**
```bash
python -m split_inference.server --host 0.0.0.0 --port 5005
```

**On Edge Client:**
```bash
python -m split_inference.client --host <SERVER_IP> --port 5005 --minibatches 30
```

### 3. Run Unit Tests
```bash
python -m unittest split_inference.tests.test_all
```
