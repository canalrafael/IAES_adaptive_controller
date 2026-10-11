# Project Checkpoint & Session Status

**Date:** 2026-10-11  
**Active Branch:** `adaptive-controller_SPViT`  
**Repository:** `https://github.com/canalrafael/IAES_adaptive_controller.git`  
**Workspace:** `c:\Users\canal\Desktop\Doutorado - PPGCC\adaptive split point controller\IAES`

---

## 1. Branch Architecture & Topic Separation

| Branch | Paper / Topic | Target Architecture | Controller | Remote Commit |
| :--- | :--- | :--- | :--- | :--- |
| **`adaptive-controller_SPViT`** *(current)* | **SPViT** (IEEE TMC 2025) | Fine-grained partitionable **Vision Transformer (ViT)** | **Original SPViT Controller** (`ARIMA.py` & `AdaptiveSplit.py`) | `c1c2347` (Pushed) |
| **`adaptive-controller`** | **Autodidactic Neurosurgeon (ANS)** (WWW 2021) | Sequential partition **ResNet-18 (CNN)** | **muLinUCB** Contextual Bandit with Ridge Regression | `914bf91` (Pushed) |

---

## 2. Summary of Work Completed Today

### A. SPViT Branch Setup (`adaptive-controller_SPViT`)
1. **Created & Synced Branch:**
   * Created new branch `adaptive-controller_SPViT` from `adaptive-controller` and pushed upstream to `origin`.
2. **Dedicated Pure SPViT Architecture:**
   * Removed all ANS / muLinUCB / ResNet scripts from `split_inference/` to keep this branch 100% focused on SPViT.
3. **Exact Original Controller Integration:**
   * Brought in the exact original files from the official SPViT repository ([`CGCL-codes/SPViT`](https://github.com/CGCL-codes/SPViT)) without any changes:
     * [`split_inference/controller/ARIMA.py`](split_inference/controller/ARIMA.py): Original ARIMA class with `difference()`, `fit()`, and `forecast()`.
     * [`split_inference/controller/AdaptiveSplit.py`](split_inference/controller/AdaptiveSplit.py): Original functions `calculate_flops_MSA`, `calculate_flops_FC`, `compute_DL`, `compute_eta`, and `adaptive_vit_inference_offloading`.
4. **Partitionable ViT Model (`split_inference/model/vit_split.py`):**
   * Implemented pure PyTorch (zero external `einops` dependency) Vision Transformer supporting:
     * **Head-Width Splitting:** Partitions Multi-Head Attention heads between edge ($h_e$) and server ($h_s$).
     * **Neuron-Width Splitting:** Partitions Feed-Forward projection neurons between edge ($m_e$) and server ($m_s$) with linear recombining: $Y = Y_{edge} + Y_{server}$.
5. **Client / Server Socket Streaming (`client.py` & `server.py`):**
   * Framed TCP streaming socket protocol with timestamp metadata ($t_e, t_s, t_r$) to profile computing capacity and bandwidth online.
6. **Testing & Simulation:**
   * Unit tests ([`split_inference/tests/test_all.py`](split_inference/tests/test_all.py)) passing all tests:
     * Exact original `ARIMA` time-series forecasting.
     * Exact original `AdaptiveSplit` math functions.
     * Numerical equivalence ($< 10^{-4}$) between full forward pass and collaborative split forward pass.
   * Standalone simulation ([`split_inference/run_spvit_simulation.py`](split_inference/run_spvit_simulation.py)) running dynamic offloading across simulated LAN, low bandwidth, and server contention phases.
7. **Documentation:**
   * Created [`README_SPViT.md`](README_SPViT.md) detailing architecture, equations, and usage.

### B. Previous Milestone on `adaptive-controller` (ANS / muLinUCB)
* Applied and pushed fixes to [`split_inference/controller/muLinUCB.py`](split_inference/controller/muLinUCB.py) (action indexing, $C_x$ bound, $A = \beta \cdot I$, scalar extraction).
* Real physical hardware validation on Raspberry Pi 4 + PC confirmed 100% decision manipulation rate and 2.43x control-induced latency amplification (+114.95 ms overhead).

---

## 3. Quick-Start Commands for Next Session

### Working on SPViT (`adaptive-controller_SPViT`):
```powershell
# 1. Run unit tests:
python -m unittest split_inference.tests.test_all

# 2. Run standalone adaptive simulation:
python -m split_inference.run_spvit_simulation --minibatches 30

# 3. Run Server (e.g. on PC / GPU):
python -m split_inference.server --port 5005

# 4. Run Edge Client (e.g. on Raspberry Pi or local test):
python -m split_inference.client --host 127.0.0.1 --port 5005 --minibatches 30
```

### Switching Back to ANS / muLinUCB (`adaptive-controller`):
```powershell
git checkout adaptive-controller
```

---

## 4. Key Documentation & Artifacts

* [`README_SPViT.md`](README_SPViT.md): Full documentation for the SPViT branch and its original controller.
* [`split_inference/controller/ARIMA.py`](split_inference/controller/ARIMA.py): Exact original SPViT ARIMA model.
* [`split_inference/controller/AdaptiveSplit.py`](split_inference/controller/AdaptiveSplit.py): Exact original SPViT offloading logic.
* [`split_inference/model/vit_split.py`](split_inference/model/vit_split.py): Fine-grained partitionable Vision Transformer.
* [`PRESENTATION_SLIDES.md`](PRESENTATION_SLIDES.md): 16-slide Google Slides presentation on ANS muLinUCB contention vulnerability.
* [`results/hardware_gonogo_results.csv`](results/hardware_gonogo_results.csv): Real hardware experimental dataset.
