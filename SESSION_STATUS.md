# Project Checkpoint & Session Status

**Date:** 2026-10-12  
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

## 2. Summary of Work Completed (Latest Session: 2026-10-12)

### A. Publication-Quality Controller Figures & Rigorous Code Verification
1. **Generated High-Resolution Figures:**
   * Script: [`split_inference/generate_controller_figures.py`](split_inference/generate_controller_figures.py)
   * Rendered at 300 DPI PNG and publication vector PDF in [`results/figures/`](results/figures/):
     * [`results/figures/mulinucb_controller_explained.png`](results/figures/mulinucb_controller_explained.png) & [`.pdf`](results/figures/mulinucb_controller_explained.pdf): Sequential layer partitioning & $\mu$LinUCB contextual bandit closed loop.
     * [`results/figures/spvit_controller_explained.png`](results/figures/spvit_controller_explained.png) & [`.pdf`](results/figures/spvit_controller_explained.pdf): Intra-block parallel ViT splitting & ARIMA-driven dynamic load balancing.
2. **Code-Level Mathematical & Algorithmic Audit:**
   * **$\mu$LinUCB (ANS, commit `914bf91`):** Verified that the 7 context features in `fillThetaContext` are `conv_macs`, `conv_layers`, `fc_macs`, `fc_layers`, `act_macs`, `act_layers`, and `mid_data_size`. The weight vector $\hat{\boldsymbol{\theta}} = \mathbf{A}^{-1}\mathbf{b}$ models remote compute speed and link transmission rate ($1/B$).
   * **SPViT (IEEE TMC 2025):** Verified exact mathematical functions in [`AdaptiveSplit.py`](split_inference/controller/AdaptiveSplit.py) and [`ARIMA.py`](split_inference/controller/ARIMA.py) (`calculate_flops_MSA`, `calculate_flops_FC`, `compute_DL`, `compute_eta`, and greedy head/neuron repartitioning).
   * **Channel Representation:** Kept "Wireless Channel" in Figure 1 to preserve original paper fidelity, while formally documenting the physical Gigabit Ethernet link with raw TCP streaming in the LaTeX manuscript.

### B. Periodic Dynamics Analysis of $\mu$LinUCB
Audited [`split_inference/controller/muLinUCB.py`](split_inference/controller/muLinUCB.py) for temporal and periodic sensitivity:
* **Periodic Forced Sampling:** The controller triggers a forced probe every $N_{\text{sample}} = \lceil \text{frameNum}^{0.25} \rceil$ frames ($t \pmod{N_{\text{sample}}} == 0$, initially every 4 frames: $t \in \{4, 8, 12, \dots\}$). It explicitly forbids the purely local execution arm (`estimate_delay[0:-1]`), compelling an offloading decision to sample channel dynamics.
* **Periodic Key-Frame Confidence Weighting:** The confidence interval width $\sqrt{(1 - L) \mathbf{x}^T \mathbf{A}^{-1} \mathbf{x}}$ oscillates periodically with video GOP patterns: $L_{\text{key}} = 0.8$ ($\sqrt{1-L} \approx 0.447$, conservative exploitation) vs. $L_{\text{nonkey}} = 0.2$ ($\sqrt{1-L} \approx 0.894$, aggressive exploration).
* **Epoch-Based Doubling Trick:** Sampling period expands geometrically across epochs ($N_{\text{sample}} \in \{4, 5, 6, \dots\}$ at $t \in \{200, 400, 800\}$).
* **Adversarial Implication:** Deterministic periodicity makes the bandit temporally predictable, exposing it to phase-locked adversarial contention attacks.

### C. Comprehensive LaTeX Chapter & Documentation
* Fully integrated LaTeX code with figures, equations, bibliography, and comparative analysis.
* Dedicated paragraphs added to LaTeX:
  * `\paragraph{Testbed Implementation and Experimental Differences.}` (Gigabit Ethernet, custom TCP binary protocol, adversarial memory/cache contention, ARM bare-metal profiling).
  * `\paragraph{Periodic Dynamics and Deterministic Exploration Schedules.}` (Formal mathematical explanation of forced exploration periodicity and vulnerability).

---

## 3. Quick-Start Commands & Recovery for Next Session

### Re-generating or Modifying Figures:
```powershell
python split_inference/generate_controller_figures.py
```

### Running SPViT Architecture & Simulation (`adaptive-controller_SPViT`):
```powershell
# Run unit tests (ARIMA, FLOPs, Numerical Equivalence)
python -m unittest split_inference.tests.test_all

# Run standalone dynamic offloading simulation
python -m split_inference.run_spvit_simulation --minibatches 30

# Run PC server
python -m split_inference.server --port 5005

# Run Raspberry Pi / edge client
python -m split_inference.client --host 127.0.0.1 --port 5005 --minibatches 30
```

### Switching to ANS / muLinUCB ResNet-18 (`adaptive-controller`):
```powershell
git checkout adaptive-controller
```

---

## 4. Key Files & Artifacts

* [`results/figures/mulinucb_controller_explained.pdf`](results/figures/mulinucb_controller_explained.pdf) & [`.png`](results/figures/mulinucb_controller_explained.png): Figure 1 (ANS $\mu$LinUCB).
* [`results/figures/spvit_controller_explained.pdf`](results/figures/spvit_controller_explained.pdf) & [`.png`](results/figures/spvit_controller_explained.png): Figure 2 (SPViT).
* [`split_inference/generate_controller_figures.py`](split_inference/generate_controller_figures.py): Generation script.
* [`split_inference/controller/AdaptiveSplit.py`](split_inference/controller/AdaptiveSplit.py) & [`ARIMA.py`](split_inference/controller/ARIMA.py): Pure SPViT controller.
* [`split_inference/model/vit_split.py`](split_inference/model/vit_split.py): Fine-grained partitionable Vision Transformer.
* [`PRESENTATION_SLIDES.md`](PRESENTATION_SLIDES.md): 16-slide presentation on $\mu$LinUCB memory contention vulnerability.
