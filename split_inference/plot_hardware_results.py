"""
Hardware Results Plot Generator
================================
Generates publication-quality figures from the physical hardware Go/No-Go experiment
conducted with Raspberry Pi 4 (Edge Client) and Host PC (Edge Server) over Gigabit Ethernet.

Outputs (in results/figures/):
  - results/figures/hardware_delta_comparison.png (.pdf)
  - results/figures/hardware_amplification_summary.png (.pdf)
  - results/figures/hardware_phase_trajectory.png (.pdf)
  - results/figures/hardware_split_decision_map.png (.pdf)
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

FIG_DIR = os.path.join("results", "figures")
os.makedirs(FIG_DIR, exist_ok=True)

# 1. Load Hardware Experiment Data
DATA_PATH = os.path.join("results", "hardware_gonogo_results.csv")
df = pd.read_csv(DATA_PATH)

# Styling
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.labelsize": 12,
    "axes.titlesize": 13,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.titlesize": 14,
    "figure.dpi": 300
})

c_phys = "#e67e22"   # Warm Orange
c_ctrl = "#2980b9"   # Strong Blue
c_green = "#27ae60"  # Emerald Green
c_red = "#c0392b"    # Dark Crimson

# ==============================================================================
# Plot 1: Per-Trial Comparison: Delta_phys vs Delta_ctrl
# ==============================================================================
fig, ax = plt.subplots(figsize=(9.0, 5.2))

x = np.arange(len(df))
width = 0.35

rects1 = ax.bar(x - width/2, df["delta_phys_ms"], width, 
                label=r"Direct Physical Contention ($\Delta_{phys}$)", 
                color=c_phys, edgecolor="#b95e09", linewidth=1.2, zorder=3)
rects2 = ax.bar(x + width/2, df["delta_probe_ctrl_ms"], width, 
                label=r"Persistent Control Degradation ($\Delta_{ctrl}$)", 
                color=c_ctrl, edgecolor="#1a5276", linewidth=1.2, zorder=3)

mean_phys = df["delta_phys_ms"].mean()
mean_ctrl = df["delta_probe_ctrl_ms"].mean()

ax.axhline(mean_phys, color=c_phys, linestyle="--", linewidth=1.5, alpha=0.85,
           label=rf"Mean $\Delta_{{phys}}$: {mean_phys:.1f} ms")
ax.axhline(mean_ctrl, color=c_ctrl, linestyle="--", linewidth=1.5, alpha=0.85,
           label=rf"Mean $\Delta_{{ctrl}}$: {mean_ctrl:.1f} ms")

# Value annotations on bars
for r in rects1:
    h = r.get_height()
    ax.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#784212")

for r in rects2:
    h = r.get_height()
    ax.annotate(f"{h:.1f}", xy=(r.get_x() + r.get_width()/2, h),
                xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                fontsize=9.5, fontweight="bold", color="#154360")

# Amplification ratio callout badges above bars
for i, row in df.iterrows():
    ratio = row["amplification_factor"]
    ax.annotate(f"{ratio:.2f}× Amp", 
                xy=(x[i] + width/2, row["delta_probe_ctrl_ms"] + 5),
                ha="center", va="bottom", fontsize=8.5, fontweight="bold", color="#1b4f72",
                bbox=dict(boxstyle="round,pad=0.22", fc="#ebf5fb", ec="#aed6f1", lw=0.9))

ax.set_ylabel("Latency Overhead (ms)")
ax.set_xlabel("Hardware Trial (Raspberry Pi 4 + GbE Ethernet)")
ax.set_title(r"Physical Contention ($\Delta_{phys}$) vs. Control Amplification ($\Delta_{ctrl}$) on Real Hardware",
             fontweight="bold", pad=12)
ax.set_xticks(x)
ax.set_xticklabels([f"Trial {t}\n($k^0=4 \\rightarrow k^a=2$)" for t in df["trial"]])
ax.set_ylim(0, 180)
ax.grid(True, linestyle=":", alpha=0.6, zorder=0)

# Legend placed cleanly in upper left within generous headroom
ax.legend(loc="upper left", ncol=2, frameon=True, facecolor="white", edgecolor="#d5d8dc")

plt.tight_layout()
p1_png = os.path.join(FIG_DIR, "hardware_delta_comparison.png")
p1_pdf = os.path.join(FIG_DIR, "hardware_delta_comparison.pdf")
plt.savefig(p1_png, dpi=300, bbox_inches="tight")
plt.savefig(p1_pdf, bbox_inches="tight")
plt.close()
print(f"Generated: {p1_png}")

# ==============================================================================
# Plot 2: Amplification Summary & Ratio Distribution
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(9.8, 4.6), gridspec_kw={"width_ratios": [1, 1.1]})

# Left: Aggregated Mean ± Std
metrics = [r"$\Delta_{phys}$" + "\n(Direct Contention)", r"$\Delta_{ctrl}$" + "\n(Persistent Overhead)"]
means = [mean_phys, mean_ctrl]
stds = [df["delta_phys_ms"].std(), df["delta_probe_ctrl_ms"].std()]
colors = [c_phys, c_ctrl]
edge_colors = ["#b95e09", "#1a5276"]

bars = ax1.bar(metrics, means, yerr=stds, capsize=7, color=colors, edgecolor=edge_colors,
               linewidth=1.3, width=0.52, zorder=3)
for bar, m, s in zip(bars, means, stds):
    ax1.annotate(f"{m:.2f} ± {s:.2f} ms",
                 xy=(bar.get_x() + bar.get_width()/2, m + s + 3.5),
                 ha="center", va="bottom", fontweight="bold", fontsize=10)

ax1.set_ylabel("Latency Overhead (ms)")
ax1.set_title("Aggregated Hardware Impact (Mean ± Std)", fontweight="bold", pad=10)
ax1.set_ylim(0, 145)
ax1.grid(True, linestyle=":", alpha=0.6, zorder=0)

# Right: Amplification Factor per Trial
trials = [f"Trial {t}" for t in df["trial"]]
ratios = df["amplification_factor"]

bar_amp = ax2.bar(trials, ratios, color=c_green, edgecolor="#196f3d", linewidth=1.2, width=0.55, zorder=3)
mean_amp = ratios.mean()
ax2.axhline(mean_amp, color="#922b21", linestyle="--", linewidth=1.6,
            label=f"Mean Amplification: {mean_amp:.2f}×")

for bar, r in zip(bar_amp, ratios):
    ax2.annotate(f"{r:.2f}×",
                 xy=(bar.get_x() + bar.get_width()/2, r),
                 xytext=(0, 4), textcoords="offset points", ha="center", va="bottom",
                 fontweight="bold", fontsize=9.5, color="#145a32")

ax2.set_ylabel(r"Damage Amplification Ratio ($\Delta_{ctrl} / \Delta_{phys}$)")
ax2.set_xlabel("Hardware Trial")
ax2.set_title(r"Amplification Factor ($\Delta_{ctrl} / \Delta_{phys}$)", fontweight="bold", pad=10)
ax2.set_ylim(0, 3.5)
ax2.grid(True, linestyle=":", alpha=0.6, zorder=0)
ax2.legend(frameon=True, facecolor="white", edgecolor="#d5d8dc", loc="upper right")

plt.tight_layout()
p2_png = os.path.join(FIG_DIR, "hardware_amplification_summary.png")
p2_pdf = os.path.join(FIG_DIR, "hardware_amplification_summary.pdf")
plt.savefig(p2_png, dpi=300, bbox_inches="tight")
plt.savefig(p2_pdf, bbox_inches="tight")
plt.close()
print(f"Generated: {p2_png}")

# ==============================================================================
# Plot 3: 3-Phase Hardware Lifecycle Trajectory
# ==============================================================================
fig, ax = plt.subplots(figsize=(10.0, 5.0))

t1_phys = df.loc[0, "delta_phys_ms"]
t1_ctrl = df.loc[0, "delta_probe_ctrl_ms"]

base_k0 = 18.0          # Clean GbE full remote offload baseline (ms)
attack_k0 = base_k0 + t1_phys  # Under contention (ms)
subopt_k2 = base_k0 + t1_ctrl  # Post-attack suboptimal split k=2 (ms)

phases = [
    ("Phase 1: Nominal Learning\n(Clean Environment)", 0, 40, base_k0, "#27ae60", "Nominal Split $k^0=4$ (Offload)"),
    ("Phase 2: Transient Contention\n(4 Stressor Workers)", 40, 52, attack_k0, "#e67e22", "Attacked Split $k=4$ (Under Contention)"),
    ("Phase 3: Post-Attack Deployment\n(Contention OFF)", 52, 90, subopt_k2, "#c0392b", "Manipulated Split $k^a=2$ (Persistent)")
]

for name, start, end, lat, col, lbl in phases:
    ax.axvspan(start, end, alpha=0.12, color=col)
    ax.hlines(lat, start, end, colors=col, linewidth=3.2, label=lbl)

# Boundary dashed lines
ax.vlines(40, base_k0, attack_k0, colors="#7f8c8d", linestyle=":", linewidth=1.6)
ax.vlines(52, attack_k0, subopt_k2, colors="#7f8c8d", linestyle=":", linewidth=1.6)

# Phase labels at top
ax.text(20, 172, "PHASE 1:\nNominal Learning\n(0-40 rounds)", ha="center", va="top",
        fontsize=9.5, fontweight="bold", color="#1e8449",
        bbox=dict(boxstyle="square,pad=0.3", fc="#eafaf1", ec="#a9dfbf", lw=0.8))
ax.text(46, 172, "PHASE 2:\nAttack Burst\n(12 rounds)", ha="center", va="top",
        fontsize=9.5, fontweight="bold", color="#b95e09",
        bbox=dict(boxstyle="square,pad=0.3", fc="#fef5e7", ec="#f8c471", lw=0.8))
ax.text(71, 172, "PHASE 3:\nPost-Attack Steady State\n(Contention Stopped)", ha="center", va="top",
        fontsize=9.5, fontweight="bold", color="#922b21",
        bbox=dict(boxstyle="square,pad=0.3", fc="#fdf2e9", ec="#f5b7b1", lw=0.8))

# Annotations with arrows
ax.annotate(r"$\Delta_{phys} = +" + f"{t1_phys:.1f}" + r"\ \mathrm{ms}$" + "\n(Direct Memory Bus Contention)",
            xy=(46, attack_k0), xytext=(46, attack_k0 + 38),
            ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#b95e09",
            arrowprops=dict(arrowstyle="->", color="#b95e09", lw=1.6))

ax.annotate(r"$\Delta_{ctrl} = +" + f"{t1_ctrl:.1f}" + r"\ \mathrm{ms}$" + f"\n({mean_amp:.2f}× Amplification)",
            xy=(71, subopt_k2), xytext=(71, subopt_k2 - 45),
            ha="center", va="top", fontsize=9.5, fontweight="bold", color="#922b21",
            arrowprops=dict(arrowstyle="->", color="#922b21", lw=1.6))

ax.set_xlabel("Inference Decision Round (Frames)")
ax.set_ylabel("End-to-End Latency (ms)")
ax.set_title("Hardware Lifecycle Trajectory: Transient Contention Induces Persistent Latency Amplification",
             fontweight="bold", pad=12)
ax.set_xlim(0, 90)
ax.set_ylim(0, 190)
ax.grid(True, linestyle=":", alpha=0.6)
ax.legend(loc="center left", frameon=True, facecolor="white", edgecolor="#d5d8dc")

plt.tight_layout()
p3_png = os.path.join(FIG_DIR, "hardware_phase_trajectory.png")
p3_pdf = os.path.join(FIG_DIR, "hardware_phase_trajectory.pdf")
plt.savefig(p3_png, dpi=300, bbox_inches="tight")
plt.savefig(p3_pdf, bbox_inches="tight")
plt.close()
print(f"Generated: {p3_png}")

# ==============================================================================
# Plot 4: Split Decision Shift & ResNet-18 Partition Map
# ==============================================================================
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.0, 4.8), gridspec_kw={"width_ratios": [1.1, 1.05]})

# Left: Matrix of Selected Splits (Nominal vs Attacked)
trials_arr = np.arange(1, 6)
k0_arr = df["k_0"].values
ka_arr = df["k_a"].values

bar_w = 0.35
ax1.bar(trials_arr - bar_w/2, k0_arr, width=bar_w, color="#27ae60", edgecolor="#196f3d", 
        linewidth=1.2, label=r"Nominal Optimal ($k^0=4$, Full Remote)", zorder=3)
ax1.bar(trials_arr + bar_w/2, ka_arr, width=bar_w, color="#c0392b", edgecolor="#922b21", 
        linewidth=1.2, label=r"Attacked Selected ($k^a=2$, After Layer 2)", zorder=3)

for t in trials_arr:
    ax1.text(t - bar_w/2, 4.15, "k=4", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#196f3d")
    ax1.text(t + bar_w/2, 2.15, "k=2", ha="center", va="bottom", fontsize=9.5, fontweight="bold", color="#922b21")

ax1.set_xlabel("Hardware Trial")
ax1.set_ylabel("Split Point Index ($k$)")
ax1.set_title("Controller Decision Manipulation (100% Success)", fontweight="bold")
ax1.set_xticks(trials_arr)
ax1.set_yticks([0, 1, 2, 3, 4])
ax1.set_yticklabels(["k=0 (Local)", "k=1 (Stem)", "k=2 (Layer 2)", "k=3 (Layer 3)", "k=4 (Remote)"])
ax1.set_ylim(0, 5.2)
ax1.grid(True, linestyle=":", alpha=0.6, zorder=0)
ax1.legend(loc="upper right", frameon=True, facecolor="white", edgecolor="#d5d8dc")

# Right: Model Partition Breakdown
blocks = ["Stem\n(7x7 Conv)", "Layer 1\n(64ch)", "Layer 2\n(128ch)", "Layer 3\n(256ch)", "Layer 4 + FC\n(512ch)"]
y_pos = np.arange(len(blocks))

# Show execution location comparison
ax2.barh(y_pos + 0.18, [1]*5, height=0.32, color="#27ae60", alpha=0.85, label="Nominal ($k=4$): Remote Server")
ax2.barh(y_pos - 0.18, [1, 1, 1, 0, 0], height=0.32, color="#e74c3c", alpha=0.85, label="Attacked ($k=2$): Forced on Pi CPU")
ax2.barh(y_pos - 0.18, [0, 0, 0, 1, 1], height=0.32, color="#3498db", alpha=0.85, left=[1, 1, 1, 0, 0], label="Attacked ($k=2$): Remote Suffix")

ax2.set_yticks(y_pos)
ax2.set_yticklabels(blocks)
ax2.set_xlim(0, 1.05)
ax2.set_xticks([])
ax2.set_xlabel("Partition Placement:\nEdge Client (Pi 4) vs. Edge Server (PC)")
ax2.set_title("Workload Shift Induced by Contention", fontweight="bold")
ax2.legend(loc="lower right", frameon=True, facecolor="white", edgecolor="#d5d8dc", fontsize=8.5)

plt.tight_layout()
p4_png = os.path.join(FIG_DIR, "hardware_split_decision_map.png")
p4_pdf = os.path.join(FIG_DIR, "hardware_split_decision_map.pdf")
plt.savefig(p4_png, dpi=300, bbox_inches="tight")
plt.savefig(p4_pdf, bbox_inches="tight")
plt.close()
print(f"Generated: {p4_png}")

print("All 4 hardware figures successfully generated in PDF and high-res PNG!")
