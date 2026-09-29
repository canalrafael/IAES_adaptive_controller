# Project Checkpoint & Session Status

**Date:** 2026-09-28  
**Active Branch:** `adaptive-controller` (repository: `canalrafael/IAES_adaptive_controller`)  
**Workspace:** `c:\Users\canal\Desktop\Doutorado - PPGCC\adaptive split point controller\IAES`

---

## 1. Current State & Milestones Achieved

- **Milestone 1:** **Stage 1 Hardware Go/No-Go Experiment [PASSED]**
  - **Hypothesis Validated on Real Physical Hardware:**
    Transient memory bus contention during learning manipulates `muLinUCB` to pick suboptimal split `k^a = 2` instead of `k^0 = 4`.
    - **Decision Manipulation Rate:** **100%** (5/5 trials)
    - **Physical Contention Impact (`Delta_phys`):** **+47.74 ms**
    - **Persistent Control Overhead (`Delta_ctrl`):** **+114.95 ms** (~2.43x amplification)
  - **Controller Integrity:**
    The `muLinUCB` controller (`split_inference/controller/muLinUCB.py`) remains **100% original and unmodified** from WWW 2021.

- **Milestone 2:** **Hardware Visual Analytics & Publication Figures [COMPLETED]**
  - Formatted raw hardware results into [`results/hardware_gonogo_results.csv`](results/hardware_gonogo_results.csv).
  - Created plot generator [`split_inference/plot_hardware_results.py`](split_inference/plot_hardware_results.py).
  - Generated 4 publication-quality figures in both high-res PNG and vector PDF formats under [`results/figures/`](results/figures/):
    1. `hardware_delta_comparison.png` / `.pdf`: Per-trial `Delta_phys` vs `Delta_ctrl` with 2.4x badge callouts.
    2. `hardware_amplification_summary.png` / `.pdf`: Mean ± std aggregated comparison and amplification ratios.
    3. `hardware_phase_trajectory.png` / `.pdf`: 3-phase lifecycle trajectory (Nominal -> Attack -> Post-Attack lock-in).
    4. `hardware_split_decision_map.png` / `.pdf`: Split decision manipulation matrix and ResNet-18 partition placement.

- **Milestone 3:** **30-Minute Presentation Script & Slide Deck [COMPLETED]**
  - Created [`PRESENTATION_SLIDES.md`](PRESENTATION_SLIDES.md) containing 16 clean, non-fluffy slides ready for Google Slides.
  - Formatted in plain text (no LaTeX syntax) with concise bullet points and complete "What to say" speaker notes.
  - Covers:
    - Background & why ANS was chosen (architectural fit & open-source availability).
    - Proof of online learning (explanation of matrix A, vector b, ridge regression).
    - Microarchitectural contention mechanics and the attribution error causing split shifts.
    - Physical hardware setup, TCP binary protocol, and experimental results.

---

## 2. Hardware & Network Setup

| Component | Hardware | Role & Operating System | IP Address |
|---|---|---|---|
| **Edge Device** | Raspberry Pi 4 | Client / Split Inference Frontend (Raspberry Pi OS 64-bit) | `192.168.1.222` |
| **Server** | Host PC | Edge Server / Backend GPU/CPU (Windows 11) | `192.168.1.19` |
| **Network** | Ethernet | Direct Gigabit LAN via Router | Port `5005` |

### Environment on Raspberry Pi 4:
- Path: `~/IAES_adaptive_controller`
- Virtual environment: `~/IAES_adaptive_controller/venv` (Python 3.13)
- PyTorch: `2.14.0+cu130` (ARM64 via piwheels)

---

## 3. Quick-Start Commands to Resume Tomorrow

### Step 1: Start Server (on Windows PC)
From the repository root (`IAES`):
```powershell
python -m split_inference.run_hardware_server --port 5005
```

### Step 2: SSH into Raspberry Pi 4
From your terminal / Ubuntu VM:
```bash
ssh canal@192.168.1.222
cd ~/IAES_adaptive_controller
source venv/bin/activate
git pull
```

### Step 3: Run Planned Next Step (Stage 2 Parameter Sweep)
```bash
# Run sweep across contention intensity (1 to 8 workers) and models:
python3 -m split_inference.run_stage2_hardware --host 192.168.1.19 --port 5005 --trials 20
```

---

## 4. Key Artifacts & Documentation

- [`PRESENTATION_SLIDES.md`](PRESENTATION_SLIDES.md): Complete 16-slide presentation text with plain-text formulas and speaking notes.
- [`results/figures/`](results/figures/): High-resolution PNG and PDF figures from the real hardware experiment.
- [`results/hardware_gonogo_results.csv`](results/hardware_gonogo_results.csv): Raw data from the 5 hardware trials.
- [`results/experiment_results_summary.txt`](results/experiment_results_summary.txt): Full plain-text experiment record.
- `SESSION_STATUS.md`: This checkpoint file for seamless pickup.
