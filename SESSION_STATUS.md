# Project Checkpoint & Session Status

**Date:** 2026-09-26  
**Active Branch:** `adaptive-controller` (repository: `canalrafael/IAES_adaptive_controller`)  
**Workspace:** `c:\Users\canal\Desktop\Doutorado - PPGCC\adaptive split point controller\IAES`

---

## 1. Current State & Milestone Achieved

- **Milestone:** **Stage 1 Hardware Go/No-Go Experiment [PASSED]**
- **Core Hypothesis Confirmed:**
  Microarchitectural contention during profiling fools the `muLinUCB` split-inference controller into selecting a suboptimal split point ($k^a = 2$ instead of $k^0 = 4$).
  - **Decision Manipulation Rate:** **100%** (5/5 trials)
  - **Direct Contention Slowdown ($\Delta_{\text{phys}}$):** **+47.74 ms**
  - **Persistent Control Overhead ($\Delta_{\text{ctrl}}$):** **+114.95 ms** (~2.4× amplification)
- **Controller Integrity:**
  The `muLinUCB` controller code (`split_inference/controller/muLinUCB.py`) remains **100% original and unmodified**.

---

## 2. Hardware & Network Setup

| Component | Hardware | Role & Operating System | IP Address |
|---|---|---|---|
| **Edge Device** | Raspberry Pi 4 | Client / Split Inference Frontend (Raspberry Pi OS 64-bit) | `192.168.1.222` |
| **Server** | Host PC | Edge Server / Backend GPU/CPU (Windows 11) | `192.168.1.19` |
| **Network** | Ethernet | Direct Gigabit LAN via Router | Port `5005` |

### Environment Setup on Raspberry Pi 4:
- Virtual environment: `~/IAES_adaptive_controller/venv` (Python 3.13)
- PyTorch: `2.14.0+cu130` (ARM64 via piwheels)
- Repository path: `~/IAES_adaptive_controller`

---

## 3. Quick-Start Commands to Resume in Next Interaction

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
```

### Step 3: Run Planned Next Step (Stage 2 Parameter Sweep)
```bash
# Quick validation or full sweep across contention workers (1 to 8):
python3 -m split_inference.run_stage2_hardware --host 192.168.1.19 --port 5005 --trials 20
```

---

## 4. Key Artifacts & Documentation Generated

- [`results/experiment_results_summary.txt`](file:///c:/Users/canal/Desktop/Doutorado%20-%20PPGCC/adaptive%20split%20point%20controller/IAES/results/experiment_results_summary.txt): Full plain-text experiment record with per-trial table and metrics.
- [`results/raw_results_20260919_*.csv`](file:///c:/Users/canal/Desktop/Doutorado%20-%20PPGCC/adaptive%20split%20point%20controller/IAES/results/): Historical and current trial raw metrics.
- `SESSION_STATUS.md`: This checkpoint file for seamless pickup.
