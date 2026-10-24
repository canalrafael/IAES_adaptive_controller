# Project Checkpoint & Session Status

**Date:** 2026-10-24  
**Active Branch:** `adaptive-controller` (repository: `canalrafael/IAES_adaptive_controller`)  
**Workspace:** `c:\Users\canal\Desktop\Doutorado - PPGCC\adaptive split point controller\IAES`  
**Guide File:** [`HOW_TO_RUN_EXPERIMENTS.txt`](HOW_TO_RUN_EXPERIMENTS.txt)

---

## 1. Current State & Milestones Achieved

- **Milestone 1:** **Stage 1 Hardware Go/No-Go Experiment [PASSED]**
  - **Hypothesis Validated on Real Physical Hardware:**
    Transient memory bus contention during learning manipulates `muLinUCB` to pick suboptimal split points ($k^a \in \{1, 2\}$ instead of nominal $k^0 = 4$).
    - **Decision Manipulation Rate:** **100%** (5/5 trials)
    - **Physical Contention Impact (`Delta_phys`):** **+21.00 ms**
    - **Persistent Control Overhead (`Delta_ctrl`):** **+90.53 ms** (~4.3x amplification)
  - **Controller Integrity:**
    The `muLinUCB` controller (`split_inference/controller/muLinUCB.py`) remains **100% original and unmodified** from WWW 2021.

- **Milestone 2:** **Stage 2 Full Hardware Campaign Sweep [COMPLETED]**
  - **Date Executed:** 2026-10-24
  - **Scale:** 10 experimental cells $\times$ 20 trials = **200 physical trials total**
    - Models: ResNet-18 and MobileNetV2
    - Contention Intensity: 1, 2, 4, 6, 8 memory contention worker threads
  - **Raw Results File:** `results/hardware_results_20261024_104909.csv` (saved on Raspberry Pi 4)

- **Milestone 3:** **Network Protocol & PyTorch Backwards-Compatibility [RESOLVED]**
  - Resolved `torch.load(..., weights_only=True)` incompatibility on PyTorch < 1.13.
  - Implemented `_safe_load_tensor()` in `split_inference/network/protocol.py` that probes `weights_only` compatibility once on import and safely seeks `buf.seek(0)` to prevent stream corruption.
  - Added robust exception logging in `split_inference/network/server.py`.

- **Milestone 4:** **Hardware Visual Analytics & Publication Figures [READY]**
  - Formatted raw hardware results into [`results/hardware_gonogo_results.csv`](results/hardware_gonogo_results.csv).
  - Created plot generator [`split_inference/plot_hardware_results.py`](split_inference/plot_hardware_results.py).
  - High-res PNG and PDF figures in [`results/figures/`](results/figures/).

- **Milestone 5:** **Official Autodidactic Neurosurgeon (ANS) Repository Integration [COMPLETED]**
  - Imported entire official `https://github.com/letian-zhang/ANS` repository as dedicated [`ans/`](ans/) source package.
  - Confirmed byte-for-byte parity with official WWW 2021 code.

---

## 2. Hardware & Network Setup

| Component | Hardware | Role & Operating System | IP Address |
|---|---|---|---|
| **Edge Device** | Raspberry Pi 4 | Client / Split Inference Frontend (Raspberry Pi OS 64-bit) | `192.168.1.222` |
| **Edge Server** | Linux VM (Ubuntu 18.04) | Backend Server for DNN Suffix Execution (Python 3.6 + PyTorch 1.10) | `192.168.1.21` |
| **Alternative Server** | Host PC (Windows 11) | Backend GPU/CPU Server (Python 3.12 + PyTorch) | `192.168.1.19` |
| **Network** | Ethernet | Direct Gigabit LAN via Router / Switch | Port `5005` |

---

## 3. Quick-Start Commands to Resume Next Session

### Step 1: Copy Results CSV from Raspberry Pi to Linux VM / PC
From your Linux VM terminal:
```bash
cd ~/Desktop/git/IAES_adaptive_controller
scp canal@192.168.1.222:~/IAES_adaptive_controller/results/hardware_results_20261024_104909.csv ./results/
```

### Step 2: Push Results to Git
```bash
git add results/
git commit -m "Save full Stage 2 hardware sweep results (200 trials)"
git push origin adaptive-controller
```

### Step 3: Generate Plots and Statistical Analysis
```bash
python3 -m split_inference.analyze_results
python3 -m split_inference.plot_hardware_results
```

---

## 4. Key Artifacts & Documentation

- [`HOW_TO_RUN_EXPERIMENTS.txt`](HOW_TO_RUN_EXPERIMENTS.txt): Complete step-by-step cheat sheet for running experiments across VM and Pi.
- [`PRESENTATION_SLIDES.md`](PRESENTATION_SLIDES.md): 16-slide presentation text with plain-text formulas and speaking notes.
- [`results/figures/`](results/figures/): High-resolution PNG and PDF publication figures.
- [`results/hardware_results_20261024_104909.csv`](results/): Raw data from the 200-trial Stage 2 hardware campaign.
- `SESSION_STATUS.md`: This checkpoint file.
