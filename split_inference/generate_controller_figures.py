"""
Script to generate high-resolution, publication-quality pedagogical figures
explaining:
1. muLinUCB Controller (Autodidactic Neurosurgeon - ANS, WWW 2021)
2. SPViT Controller (Adaptive Split & ARIMA, IEEE TMC 2025)

Refined to strictly reflect the exact mathematical, architectural, and
code-level implementation in the repository.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import FancyBboxPatch

os.makedirs("results/figures", exist_ok=True)

# Styling configurations
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
plt.rcParams['mathtext.fontset'] = 'dejavusans'

# ==============================================================================
# FIGURE 1: muLinUCB CONTROLLER (ANS)
# ==============================================================================
def draw_mulinucb_figure():
    fig = plt.figure(figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')

    # Title Banner
    fig.text(0.5, 0.97, "muLinUCB Controller: Sequential Layer Partitioning & Contextual Bandit",
             ha='center', va='top', fontsize=17, fontweight='bold', color='#0F172A')
    fig.text(0.5, 0.94, "Autodidactic Neurosurgeon (ANS) • Online Learning for Deep Inference Offloading",
             ha='center', va='top', fontsize=11.5, color='#475569')

    # Subplot A (Top: What it does) & Subplot B (Bottom: How it works)
    ax_a = fig.add_axes([0.03, 0.54, 0.94, 0.38])
    ax_b = fig.add_axes([0.03, 0.05, 0.94, 0.45])

    for ax in [ax_a, ax_b]:
        ax.set_facecolor('#FFFFFF')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        rect = FancyBboxPatch((0, 0), 100, 100, boxstyle="round,pad=0,rounding_size=2",
                              edgecolor='#CBD5E1', facecolor='#FFFFFF', lw=1.5, zorder=0)
        ax.add_patch(rect)

    # -------------------------------------------------------------------------
    # PANEL A: WHAT IT DOES
    # -------------------------------------------------------------------------
    ax_a.text(3, 94, "(A) What It Does: Sequential Layer-Wise Partitioning of Deep Neural Networks (CNNs)",
              fontsize=12.5, fontweight='bold', color='#1E293B')

    # Edge Device Box
    edge_box = FancyBboxPatch((4, 20), 34, 68, boxstyle="round,pad=1,rounding_size=3",
                              edgecolor='#3B82F6', facecolor='#EFF6FF', lw=2, zorder=1)
    ax_a.add_patch(edge_box)
    ax_a.text(21, 82, "Mobile Edge Device (Raspberry Pi 4 / CPU)", ha='center', fontsize=11, fontweight='bold', color='#1D4ED8')
    ax_a.text(21, 76, "Local Prefix Computation F_k(x): Layers 1 to k", ha='center', fontsize=9.5, color='#2563EB')

    # Input Image
    input_box = FancyBboxPatch((6, 32), 6.5, 36, boxstyle="round,pad=0.5,rounding_size=2",
                               edgecolor='#64748B', facecolor='#F1F5F9', lw=1.5, zorder=2)
    ax_a.add_patch(input_box)
    ax_a.text(9.25, 50, "Input\nRaw\nImage\n(x)", ha='center', va='center', fontsize=9, fontweight='bold', color='#334155')

    ax_a.annotate('', xy=(14.5, 50), xytext=(12.8, 50),
                  arrowprops=dict(arrowstyle="->", lw=2, color='#3B82F6'))

    # Edge DNN Layers
    layer_colors = ['#93C5FD', '#60A5FA', '#3B82F6']
    for idx, col in enumerate(layer_colors):
        lx = 15.5 + idx * 7
        l_box = FancyBboxPatch((lx, 32), 5.5, 36, boxstyle="round,pad=0.5,rounding_size=2",
                               edgecolor='#1E40AF', facecolor=col, lw=1.5, zorder=2)
        ax_a.add_patch(l_box)
        ax_a.text(lx + 2.75, 50, f"Layer\n{idx+1}", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#FFFFFF')
        if idx < 2:
            ax_a.annotate('', xy=(lx + 7, 50), xytext=(lx + 5.7, 50),
                          arrowprops=dict(arrowstyle="->", lw=1.5, color='#1E40AF'))

    ax_a.text(24.5, 24, "Edge Latency: T_edge = frontDelay(k)", ha='center', fontsize=9.5, fontweight='bold', color='#1E40AF')

    # Split Point Indicator
    split_line = FancyBboxPatch((36.5, 18), 4, 72, boxstyle="round,pad=0.5,rounding_size=1",
                                edgecolor='#DC2626', facecolor='#FEE2E2', lw=2, ls='--', zorder=3)
    ax_a.add_patch(split_line)
    ax_a.text(38.5, 54, "SPLIT POINT  k  (Discrete Action)", ha='center', va='center',
              fontsize=8.5, fontweight='bold', color='#B91C1C', rotation=90)

    # Transmission Channel Box
    channel_box = FancyBboxPatch((42, 24), 16, 60, boxstyle="round,pad=1,rounding_size=3",
                                 edgecolor='#F59E0B', facecolor='#FEF3C7', lw=2, zorder=1)
    ax_a.add_patch(channel_box)
    ax_a.text(50, 78, "Wireless Channel", ha='center', fontsize=11, fontweight='bold', color='#B45309')
    ax_a.text(50, 72, "TCP Binary Protocol ('SPLT')", ha='center', fontsize=9, color='#D97706')

    # Tensor transmitted
    tensor_box = FancyBboxPatch((44, 42), 12, 20, boxstyle="round,pad=0.5,rounding_size=2",
                                edgecolor='#D97706', facecolor='#FDE68A', lw=1.5, zorder=2)
    ax_a.add_patch(tensor_box)
    ax_a.text(50, 52, "Intermediate\nTensor Z_k\nSize: mid_data_size", ha='center', va='center',
              fontsize=7.8, fontweight='bold', color='#92400E')
    ax_a.text(50, 32, "T_trans = mid_data / B", ha='center', fontsize=9, fontweight='bold', color='#B45309')

    # Edge Server Box
    server_box = FancyBboxPatch((60, 20), 36, 68, boxstyle="round,pad=1,rounding_size=3",
                                edgecolor='#10B981', facecolor='#ECFDF5', lw=2, zorder=1)
    ax_a.add_patch(server_box)
    ax_a.text(78, 82, "Edge Server (Workstation / GPU)", ha='center', fontsize=11, fontweight='bold', color='#047857')
    ax_a.text(78, 76, "Remote Suffix Computation G_k(Z_k): Layers k+1 to K", ha='center', fontsize=9.5, color='#059669')

    # Server DNN Layers
    s_layer_colors = ['#34D399', '#10B981', '#059669']
    for idx, col in enumerate(s_layer_colors):
        sx = 63 + idx * 7.5
        sl_box = FancyBboxPatch((sx, 32), 6, 36, boxstyle="round,pad=0.5,rounding_size=2",
                                edgecolor='#065F46', facecolor=col, lw=1.5, zorder=2)
        ax_a.add_patch(sl_box)
        ax_a.text(sx + 3, 50, f"Layer\nk+{idx+1}", ha='center', va='center', fontsize=8.5, fontweight='bold', color='#FFFFFF')
        if idx < 2:
            ax_a.annotate('', xy=(sx + 7.5, 50), xytext=(sx + 6.2, 50),
                          arrowprops=dict(arrowstyle="->", lw=1.5, color='#065F46'))

    # Output Box
    out_box = FancyBboxPatch((87.5, 32), 7, 36, boxstyle="round,pad=0.5,rounding_size=2",
                             edgecolor='#64748B', facecolor='#F1F5F9', lw=1.5, zorder=2)
    ax_a.add_patch(out_box)
    ax_a.text(91, 50, "Final\nClass\nOutput\ny_hat", ha='center', va='center', fontsize=9, fontweight='bold', color='#334155')

    ax_a.annotate('', xy=(87.2, 50), xytext=(85.5, 50),
                  arrowprops=dict(arrowstyle="->", lw=2, color='#10B981'))

    ax_a.text(78, 24, "Server Latency: T_server(conv_macs, fc_macs)", ha='center', fontsize=9.5, fontweight='bold', color='#065F46')

    # Formula Callout at Bottom of Panel A
    formula_box = FancyBboxPatch((18, 3), 64, 12, boxstyle="round,pad=0.5,rounding_size=2",
                                 edgecolor='#94A3B8', facecolor='#F8FAFC', lw=1.2, zorder=2)
    ax_a.add_patch(formula_box)
    ax_a.text(50, 9.5, "Total Sequential Latency:  T_total(k) = T_edge(1..k) + T_trans(Z_k) + T_server(k+1..K)",
              ha='center', fontsize=10, fontweight='bold', color='#0F172A')
    ax_a.text(50, 5, "Discrete Split Points: k=0 (Full Local Edge)  |  k=1..3 (Collaborative Split)  |  k=4 (Full Remote Server)",
              ha='center', fontsize=8.8, color='#475569')

    # -------------------------------------------------------------------------
    # PANEL B: HOW IT WORKS
    # -------------------------------------------------------------------------
    ax_b.text(3, 95, "(B) How It Works: muLinUCB Contextual Bandit Decision & Online Learning Loop",
              fontsize=12.5, fontweight='bold', color='#1E293B')

    # Step 1: Context Vector
    b1_box = FancyBboxPatch((3.5, 25), 18.5, 66, boxstyle="round,pad=1,rounding_size=3",
                            edgecolor='#6366F1', facecolor='#EEF2FF', lw=1.8, zorder=1)
    ax_b.add_patch(b1_box)
    ax_b.text(12.75, 86, "1. Context Vector", ha='center', fontsize=10.5, fontweight='bold', color='#4338CA')
    ax_b.text(12.75, 80, "fillThetaContext() in R^7", ha='center', fontsize=8.5, color='#4F46E5')
    features = [
        "x[0]: conv_macs (Remote)",
        "x[1]: conv_layers (Count)",
        "x[2]: fc_macs (Remote)",
        "x[3]: fc_layers (Count)",
        "x[4]: act_macs (Remote)",
        "x[5]: act_layers (Count)",
        "x[6]: mid_data_size (Bytes)"
    ]
    for i, f in enumerate(features):
        ax_b.text(5, 72 - i * 5.8, f, fontsize=7.8, color='#312E81')

    # Arrow 1 -> 2
    ax_b.annotate('', xy=(23.5, 58), xytext=(22.2, 58),
                  arrowprops=dict(arrowstyle="->", lw=2, color='#6366F1'))

    # Step 2: Ridge Regression & UCB Estimator
    b2_box = FancyBboxPatch((24, 25), 32.5, 66, boxstyle="round,pad=1,rounding_size=3",
                            edgecolor='#8B5CF6', facecolor='#F5F3FF', lw=1.8, zorder=1)
    ax_b.add_patch(b2_box)
    ax_b.text(40.25, 86, "2. Ridge Regression & LCB Estimator", ha='center', fontsize=10.5, fontweight='bold', color='#6D28D9')
    ax_b.text(40.25, 79.5, "Online Matrix State (beta = max(1, Cx^2)):", ha='center', fontsize=8.8, fontweight='bold', color='#5B21B6')
    ax_b.text(40.25, 73.5, "A = sum(x x^T) + beta * I  in R^(7x7)", ha='center', fontsize=8.5, color='#4C1D95')
    ax_b.text(40.25, 67.5, "b = sum(x * y)  in R^7", ha='center', fontsize=8.5, color='#4C1D95')
    ax_b.text(40.25, 61.5, "theta_hat = A^(-1) b  (Tracks 1/V_server & 1/BW)", ha='center', fontsize=8.3, fontweight='bold', color='#5B21B6')

    # UCB equation inner box
    ucb_eq_box = FancyBboxPatch((25.5, 28), 29.5, 28, boxstyle="round,pad=0.5,rounding_size=2",
                                edgecolor='#7C3AED', facecolor='#EDE9FE', lw=1.2, zorder=2)
    ax_b.add_patch(ucb_eq_box)
    ax_b.text(40.25, 51, "Optimistic Delay Prediction L_hat(a):", ha='center', fontsize=8.5, fontweight='bold', color='#4C1D95')
    ax_b.text(40.25, 44.5, "L_hat(a) = x_a^T theta_hat - alpha * sqrt((1-L) x_a^T A^(-1) x_a)",
              ha='center', fontsize=7.6, fontweight='bold', color='#3730A3')
    ax_b.text(40.25, 38.5, "+ frontDelay(a)", ha='center', fontsize=8, color='#3730A3')
    ax_b.text(40.25, 32.5, "Mean Delay Exploitation  |  Variance Bonus Exploration",
              ha='center', fontsize=7.2, color='#5B21B6')

    # Arrow 2 -> 3
    ax_b.annotate('', xy=(58, 58), xytext=(56.7, 58),
                  arrowprops=dict(arrowstyle="->", lw=2, color='#8B5CF6'))

    # Step 3: Decision Selection
    b3_box = FancyBboxPatch((58.5, 25), 18.5, 66, boxstyle="round,pad=1,rounding_size=3",
                            edgecolor='#EC4899', facecolor='#FDF2F8', lw=1.8, zorder=1)
    ax_b.add_patch(b3_box)
    ax_b.text(67.75, 86, "3. Decision Selection", ha='center', fontsize=10.5, fontweight='bold', color='#BE185D')
    ax_b.text(67.75, 78, "Optimal Split Point k*:", ha='center', fontsize=8.8, color='#9D174D')
    ax_b.text(67.75, 70, "k* = argmin_a L_hat(a)", ha='center', fontsize=9.5, fontweight='bold', color='#831843')

    # Exploration sub-box
    ax_b.text(67.75, 56, "Forced Sampling:", ha='center', fontsize=8.5, fontweight='bold', color='#9D174D')
    ax_b.text(67.75, 47, "If t mod forceSample == 0:\nargmin over non-local\nsplit arms [0:-1]", ha='center',
              fontsize=7.8, color='#701A75')
    ax_b.text(67.75, 33, "Confidence Weighting:\nKey Frame L_key = 0.8\nNon-Key L_nonkey = 0.2", ha='center',
              fontsize=7.5, color='#9D174D')

    # Arrow 3 -> 4
    ax_b.annotate('', xy=(78.5, 58), xytext=(77.2, 58),
                  arrowprops=dict(arrowstyle="->", lw=2, color='#EC4899'))

    # Step 4: Online Feedback & Recursive Update
    b4_box = FancyBboxPatch((79, 25), 18, 66, boxstyle="round,pad=1,rounding_size=3",
                            edgecolor='#059669', facecolor='#ECFDF5', lw=1.8, zorder=1)
    ax_b.add_patch(b4_box)
    ax_b.text(88, 86, "4. Online Update", ha='center', fontsize=10.5, fontweight='bold', color='#047857')
    ax_b.text(88, 79, "Measure Real Latency y_t", ha='center', fontsize=8.5, color='#065F46')
    ax_b.text(88, 73, "y_t = t_loc + t_comm_server", ha='center', fontsize=8, color='#065F46')

    ax_b.text(88, 62, "Recursive Updates (updateA_b):", ha='center', fontsize=8, fontweight='bold', color='#047857')
    ax_b.text(88, 55, "A <- A + x x^T", ha='center', fontsize=9, fontweight='bold', color='#065F46')
    ax_b.text(88, 48, "b <- b + x * y_t", ha='center', fontsize=9, fontweight='bold', color='#065F46')
    ax_b.text(88, 36, "Doubling Trick:\nIf current_frame > frameNum:\nframeNum *= 2, updates alpha",
              ha='center', fontsize=7.4, color='#065F46')

    # Feedback loop curved arrow below boxes
    loop_arrow = patches.FancyArrowPatch((88, 22), (40.25, 22),
                                         connectionstyle="arc3,rad=-0.10",
                                         arrowstyle="->", lw=2, color='#059669', ls='--')
    ax_b.add_patch(loop_arrow)
    ax_b.text(64, 8, "Closed-Loop Feedback: theta_hat tracks network & server load online without external probes",
              ha='center', fontsize=8.8, fontweight='bold', color='#047857')

    output_png = "results/figures/mulinucb_controller_explained.png"
    output_pdf = "results/figures/mulinucb_controller_explained.pdf"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_png} and {output_pdf}")


# ==============================================================================
# FIGURE 2: SPViT CONTROLLER (AdaptiveSplit & ARIMA)
# ==============================================================================
def draw_spvit_figure():
    fig = plt.figure(figsize=(16, 11), dpi=300)
    fig.patch.set_facecolor('#F8FAFC')

    # Title Banner
    fig.text(0.5, 0.97, "SPViT Controller: Fine-Grained Parallel ViT Splitting & ARIMA Offloading",
             ha='center', va='top', fontsize=17, fontweight='bold', color='#0F172A')
    fig.text(0.5, 0.94, "SPViT (IEEE TMC 2025) • Dynamic Head & Neuron Partitioning for Vision Transformers",
             ha='center', va='top', fontsize=11.5, color='#475569')

    # Subplots
    ax_a = fig.add_axes([0.03, 0.54, 0.94, 0.38])
    ax_b = fig.add_axes([0.03, 0.05, 0.94, 0.45])

    for ax in [ax_a, ax_b]:
        ax.set_facecolor('#FFFFFF')
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.axis('off')
        rect = FancyBboxPatch((0, 0), 100, 100, boxstyle="round,pad=0,rounding_size=2",
                              edgecolor='#CBD5E1', facecolor='#FFFFFF', lw=1.5, zorder=0)
        ax.add_patch(rect)

    # -------------------------------------------------------------------------
    # PANEL A: WHAT IT DOES
    # -------------------------------------------------------------------------
    ax_a.text(3, 94, "(A) What It Does: Fine-Grained Intra-Block Parallel Splitting of Vision Transformers (ViT)",
              fontsize=12.5, fontweight='bold', color='#1E293B')

    # Input Tokens Box
    in_tokens = FancyBboxPatch((3, 30), 8, 42, boxstyle="round,pad=0.5,rounding_size=2",
                               edgecolor='#64748B', facecolor='#F1F5F9', lw=1.5, zorder=2)
    ax_a.add_patch(in_tokens)
    ax_a.text(7, 51, "Patch\nTokens\nX\n(L x D)", ha='center', va='center', fontsize=9.5, fontweight='bold', color='#334155')

    # Arrow to Split MSA
    ax_a.annotate('', xy=(12.5, 51), xytext=(11.2, 51),
                  arrowprops=dict(arrowstyle="->", lw=2, color='#64748B'))

    # Sub-block 1: Head-Width Splitting (MSA)
    msa_container = FancyBboxPatch((13, 16), 38, 70, boxstyle="round,pad=1,rounding_size=3",
                                  edgecolor='#8B5CF6', facecolor='#FAF5FF', lw=1.8, zorder=1)
    ax_a.add_patch(msa_container)
    ax_a.text(32, 81, "1. Head-Width Splitting: Multi-Head Attention (MSA)", ha='center',
              fontsize=10.5, fontweight='bold', color='#6D28D9')
    ax_a.text(32, 75, "Total H Heads split into h_e (Edge) and h_s (Server)", ha='center',
              fontsize=9, color='#7C3AED')

    # Edge MSA heads
    e_msa = FancyBboxPatch((15, 48), 34, 23, boxstyle="round,pad=0.5,rounding_size=2",
                           edgecolor='#3B82F6', facecolor='#EFF6FF', lw=1.5, zorder=2)
    ax_a.add_patch(e_msa)
    ax_a.text(32, 65, "EDGE: Computes h_e Heads in Parallel", ha='center', fontsize=9.5, fontweight='bold', color='#1D4ED8')
    ax_a.text(32, 56, "Q_e, K_e, V_e -> Softmax(Q K^T / sqrt(d)) V -> O_e\nFLOPs_MSA_e = 4 h_e L D^2 + 2 h_e L^2 D",
              ha='center', fontsize=8, color='#1E40AF')

    # Server MSA heads
    s_msa = FancyBboxPatch((15, 21), 34, 23, boxstyle="round,pad=0.5,rounding_size=2",
                           edgecolor='#10B981', facecolor='#ECFDF5', lw=1.5, zorder=2)
    ax_a.add_patch(s_msa)
    ax_a.text(32, 38, "SERVER: Computes h_s = (H - h_e) Heads", ha='center', fontsize=9.5, fontweight='bold', color='#047857')
    ax_a.text(32, 29, "Parallel execution + transmitted via TCP socket\nFLOPs_MSA_s = 4 h_s L D^2 + 2 h_s L^2 D",
              ha='center', fontsize=8, color='#065F46')

    # Arrow to Neuron Splitting
    ax_a.annotate('', xy=(53.5, 51), xytext=(51.2, 51),
                  arrowprops=dict(arrowstyle="->", lw=2, color='#64748B'))

    # Sub-block 2: Neuron-Width Splitting (FFN / MLP)
    ffn_container = FancyBboxPatch((54, 16), 36, 70, boxstyle="round,pad=1,rounding_size=3",
                                  edgecolor='#EC4899', facecolor='#FDF2F8', lw=1.8, zorder=1)
    ax_a.add_patch(ffn_container)
    ax_a.text(72, 81, "2. Neuron-Width Splitting: Feed-Forward (FFN)", ha='center',
              fontsize=10.5, fontweight='bold', color='#BE185D')
    ax_a.text(72, 75, "Hidden dimension M split into m_e (Edge) and m_s (Server)", ha='center',
              fontsize=9, color='#9D174D')

    # Edge FFN
    e_ffn = FancyBboxPatch((56, 48), 32, 23, boxstyle="round,pad=0.5,rounding_size=2",
                           edgecolor='#3B82F6', facecolor='#EFF6FF', lw=1.5, zorder=2)
    ax_a.add_patch(e_ffn)
    ax_a.text(72, 65, "EDGE: m_e Neurons (W1_e, W2_e)", ha='center', fontsize=9.5, fontweight='bold', color='#1D4ED8')
    ax_a.text(72, 56, "Partial Forward: Y_edge = GELU(X W1_e) W2_e\nFLOPs_FC_e = L (m_e + 1) D",
              ha='center', fontsize=8, color='#1E40AF')

    # Server FFN
    s_ffn = FancyBboxPatch((56, 21), 32, 23, boxstyle="round,pad=0.5,rounding_size=2",
                           edgecolor='#10B981', facecolor='#ECFDF5', lw=1.5, zorder=2)
    ax_a.add_patch(s_ffn)
    ax_a.text(72, 38, "SERVER: m_s = (M - m_e) Neurons", ha='center', fontsize=9.5, fontweight='bold', color='#047857')
    ax_a.text(72, 29, "Partial Forward: Y_server = GELU(X W1_s) W2_s\nFLOPs_FC_s = L (m_s + 1) D",
              ha='center', fontsize=8, color='#065F46')

    # Linear Recombination Addition
    ax_a.annotate('', xy=(91.5, 51), xytext=(88.5, 60),
                  arrowprops=dict(arrowstyle="->", lw=1.5, color='#3B82F6'))
    ax_a.annotate('', xy=(91.5, 51), xytext=(88.5, 32),
                  arrowprops=dict(arrowstyle="->", lw=1.5, color='#10B981'))

    out_add = FancyBboxPatch((91.5, 40), 6.5, 22, boxstyle="round,pad=0.5,rounding_size=2",
                             edgecolor='#D97706', facecolor='#FEF3C7', lw=1.5, zorder=2)
    ax_a.add_patch(out_add)
    ax_a.text(94.75, 51, "Exact Sum:\nY = Y_e + Y_s\n(Lossless)", ha='center', va='center',
              fontsize=7.8, fontweight='bold', color='#92400E')

    # Bottom Callout
    ax_a.text(50, 6, "Key Advantage: True intra-layer concurrency balances Edge and Server execution without pipeline bubbles",
              ha='center', fontsize=9.5, fontweight='bold', color='#334155')

    # -------------------------------------------------------------------------
    # PANEL B: HOW IT WORKS
    # -------------------------------------------------------------------------
    ax_b.text(3, 95, "(B) How It Works: Real-Time ARIMA Forecasting & Dynamic Load Balancing (AdaptiveSplit)",
              fontsize=12.5, fontweight='bold', color='#1E293B')

    # Step 1: Profiling
    p1_box = FancyBboxPatch((3.5, 25), 21, 66, boxstyle="round,pad=1,rounding_size=3",
                            edgecolor='#0284C7', facecolor='#F0F9FF', lw=1.8, zorder=1)
    ax_b.add_patch(p1_box)
    ax_b.text(14, 86, "1. Runtime Profiling", ha='center', fontsize=10.5, fontweight='bold', color='#0369A1')
    ax_b.text(14, 80, "Timestamps & History", ha='center', fontsize=9, color='#0284C7')
    prof_items = [
        "t_e : Edge complete time",
        "t_s : Server start time",
        "t_r : Server reply time",
        "",
        "Compute Capability:",
        "FLOPS_d = (FLOPs_a + FLOPs_f)",
        "          / (t_e - t_s)",
        "",
        "Network Bandwidth:",
        "BW_d = Params / (t_r - t_s)"
    ]
    for idx, itm in enumerate(prof_items):
        weight = 'bold' if ':' in itm or 'FLOPS_d' in itm or 'BW_d' in itm else 'normal'
        ax_b.text(5, 73 - idx * 4.4, itm, fontsize=7.6, fontweight=weight, color='#0C4A6E')

    # Arrow 1 -> 2
    ax_b.annotate('', xy=(26.2, 58), xytext=(24.8, 58),
                  arrowprops=dict(arrowstyle="->", lw=2, color='#0284C7'))

    # Step 2: ARIMA Forecasting
    p2_box = FancyBboxPatch((26.5, 25), 21.5, 66, boxstyle="round,pad=1,rounding_size=3",
                            edgecolor='#D97706', facecolor='#FFFBEB', lw=1.8, zorder=1)
    ax_b.add_patch(p2_box)
    ax_b.text(37.25, 86, "2. ARIMA Predictor", ha='center', fontsize=10.5, fontweight='bold', color='#B45309')
    ax_b.text(37.25, 80, "ARIMA(p=5, d=1, q=0)", ha='center', fontsize=9, color='#D97706')

    arima_desc = [
        "Sliding Window Sequences:",
        "seq_flops[d], seq_bw[d]",
        "",
        "First-order Differencing:",
        "Delta y_t = y_t - y_{t-1}",
        "",
        "Autoregressive Forecast:",
        "pre_flops_d = ARIMA.fit()",
        "pre_bw_d = ARIMA.fit()",
        "",
        "Online forecast of next-step\nprocessing speed & bandwidth"
    ]
    for idx, itm in enumerate(arima_desc):
        weight = 'bold' if 'Forecast' in itm or 'Predictor' in itm or 'Sequences:' in itm else 'normal'
        ax_b.text(28, 73 - idx * 4.3, itm, fontsize=7.4, fontweight=weight, color='#78350F')

    # Arrow 2 -> 3
    ax_b.annotate('', xy=(49.5, 58), xytext=(48.2, 58),
                  arrowprops=dict(arrowstyle="->", lw=2, color='#D97706'))

    # Step 3: Latency Modeling DL
    p3_box = FancyBboxPatch((50, 25), 23, 66, boxstyle="round,pad=1,rounding_size=3",
                            edgecolor='#7C3AED', facecolor='#F5F3FF', lw=1.8, zorder=1)
    ax_b.add_patch(p3_box)
    ax_b.text(61.5, 86, "3. Projected Latency DL", ha='center', fontsize=10.5, fontweight='bold', color='#6D28D9')
    ax_b.text(61.5, 80, "Delay Model compute_DL()", ha='center', fontsize=9, color='#7C3AED')

    dl_items = [
        "Device Projected Delay DL_d:",
        "DL_d = (Params / pre_bw_d)",
        "     + (FLOPs_a / pre_flops_d)",
        "     + (FLOPs_f / pre_flops_d)",
        "",
        "Imbalance Metric eta:",
        "eta = |max(DL) - min(DL)|",
        "      - lambda_tolerance",
        "",
        "Status Check:",
        "eta <= 0: Workloads balanced",
        "eta > 0: Rebalance required"
    ]
    for idx, itm in enumerate(dl_items):
        weight = 'bold' if 'eta' in itm or 'Delay DL_d:' in itm or 'Status' in itm else 'normal'
        ax_b.text(51.5, 74 - idx * 3.8, itm, fontsize=7.4, fontweight=weight, color='#4C1D95')

    # Arrow 3 -> 4
    ax_b.annotate('', xy=(74.5, 58), xytext=(73.2, 58),
                  arrowprops=dict(arrowstyle="->", lw=2, color='#7C3AED'))

    # Step 4: Greedy Partition Adjustment
    p4_box = FancyBboxPatch((75, 25), 22, 66, boxstyle="round,pad=1,rounding_size=3",
                            edgecolor='#059669', facecolor='#ECFDF5', lw=1.8, zorder=1)
    ax_b.add_patch(p4_box)
    ax_b.text(86, 86, "4. Greedy Offloading", ha='center', fontsize=10.5, fontweight='bold', color='#047857')
    ax_b.text(86, 80, "adaptive_vit_offloading()", ha='center', fontsize=8.8, color='#059669')

    greedy_steps = [
        "If eta > 0 (imbalance):",
        "",
        "1. Rebalance Heads:",
        "   for h in 1..head_count:",
        "     d* = argmin(DL)",
        "     Assign head h to d*",
        "     DL[d*] += FLOPs_h / FLOPS",
        "",
        "2. Rebalance Neurons:",
        "   for n in 1..neuron_count:",
        "     d* = argmin(DL)",
        "     Assign neuron n to d*",
        "     DL[d*] += FLOPs_n / FLOPS",
        "",
        "Output: Optimal (h_e, m_e)"
    ]
    for idx, itm in enumerate(greedy_steps):
        weight = 'bold' if 'If' in itm or 'Output' in itm or '1.' in itm or '2.' in itm else 'normal'
        ax_b.text(76.5, 75 - idx * 3.3, itm, fontsize=7.3, fontweight=weight, color='#064E3B')

    # Closed-loop arrow from Step 4 back to Step 1 below boxes
    spvit_loop = patches.FancyArrowPatch((86, 22), (14, 22),
                                         connectionstyle="arc3,rad=-0.10",
                                         arrowstyle="->", lw=2, color='#059669', ls='--')
    ax_b.add_patch(spvit_loop)
    ax_b.text(50, 8, "Continuous Feedback Loop: Dynamically adapts (h_e, m_e) per minibatch under varying network & edge load",
              ha='center', fontsize=8.8, fontweight='bold', color='#047857')

    output_png = "results/figures/spvit_controller_explained.png"
    output_pdf = "results/figures/spvit_controller_explained.pdf"
    plt.savefig(output_png, dpi=300, bbox_inches='tight')
    plt.savefig(output_pdf, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_png} and {output_pdf}")


if __name__ == '__main__':
    draw_mulinucb_figure()
    draw_spvit_figure()
    print("Done generating updated controller figures!")
