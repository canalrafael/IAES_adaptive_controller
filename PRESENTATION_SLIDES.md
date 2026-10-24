# 30-Minute Presentation: Attacking Split Inference via Microarchitectural Contention
**Format:** Google Slides Ready (Plain Text, No LaTeX)  
**Date:** 2026-09-28  
**Topic:** Adversarial Manipulation of the Autodidactic Neurosurgeon (`muLinUCB`) Adaptive Split Controller  

---

## Slide 1: Introduction & Goal

* **Slide Title:** Attacking Split Inference: Manipulating the muLinUCB Controller via Memory Contention
* **Slide Bullets:**
  * Context: Collaborative DNN split inference between an edge client (Raspberry Pi 4) and an edge server (PC).
  * Controller studied: Autodidactic Neurosurgeon (ANS) with the muLinUCB algorithm (WWW 2021).
  * Main Question: Can an unprivileged local application generating hardware memory contention fool this adaptive controller?
  * Key Result: Yes. 100% manipulation rate, causing a persistent latency penalty 2.4x larger than the attack itself.
* **What to say:**
  > "Today I'm presenting our results on attacking adaptive split inference. We wanted to see if an unprivileged background process generating memory contention on a Raspberry Pi can trick an online learning controller into picking bad split points. We tested the muLinUCB controller from the Autodidactic Neurosurgeon paper, and we found that a short burst of contention permanently locks the controller into a suboptimal split point, giving a 2.4x amplification in latency degradation."

---

## Slide 2: Split Inference Basics

* **Slide Title:** Split Inference: Balancing Local vs. Remote Workload
* **Slide Bullets:**
  * End-to-end latency: `d_p = d_local + d_transmission + d_server + noise`
  * Split point selection:
    * `p = 0` (or `k = 4` in our ResNet-18): Send raw image to server (Full Offload).
    * `p = P` (or `k = 0` in our ResNet-18): Run entire model on the edge device (Full Local).
    * `0 < p < P`: Execute prefix locally, send intermediate activation, execute suffix on server.
  * Trade-off: Earlier split = less local CPU/GPU work, but larger data sent over network.
* **What to say:**
  > "In split inference, we split a neural network at layer p. The total latency is the local prefix time, plus transmission time over the network, plus server suffix time. If the network is fast, offloading everything is usually best. If the network is slow or the edge device is fast, running locally or splitting at a layer with small activation sizes is better. The challenge is knowing where to split in real-time."

---

## Slide 3: Why We Chose the ANS Controller (vs. Others)

* **Slide Title:** Why Autodidactic Neurosurgeon (ANS)?
* **Slide Bullets:**
  * **Fit to Our Architecture:**
    * Designed specifically for asymmetric edge-server setups (low-power edge client + remote workstation server).
    * Lightweight: All controller computation runs directly on the edge device with minimal CPU and memory overhead (order d^2 per frame, where d = 7).
    * Requires zero cooperation from the server beyond standard result replies (learns purely from round-trip delay).
  * **Code & Algorithm Availability:**
    * Unlike closed-source or hardware-specific proprietary frameworks, ANS has a complete, mathematically reproducible open formulation published at WWW 2021.
    * Easy to implement faithfully in user space without custom kernel patches or specialized profiling daemons.
  * **Comparison to Alternatives:**
    * *Neurosurgeon / JointDNN:* Require heavy offline profiling and pre-calculated lookup tables that break under real dynamic conditions.
    * *Layer-wise models (DADS, Edge AI):* Sum execution layer-by-layer using FLOPs, completely missing PyTorch/cuDNN runtime optimizations.
* **What to say:**
  > "We chose the Autodidactic Neurosurgeon controller for two practical reasons: architectural fit and availability. First, it fits our setup perfectly: we have a resource-constrained edge device (Raspberry Pi) communicating with a PC server. The controller runs entirely on the Pi with minimal compute overhead—just a 7x7 matrix inversion—and it needs no special telemetry from the server other than the return of the inference result. Second, availability: unlike many split-inference proposals that rely on closed-source frameworks or proprietary hardware hooks, ANS published a clean, mathematically complete algorithm at WWW 2021 that can be implemented cleanly and verified line-by-line. Other surveyed options either require extensive offline profiling or rely on layer-wise FLOP models that produce large prediction errors."

---

## Slide 4: Proving It Is Online Learning (What are Matrix A and Vector b?)

* **Slide Title:** Proof of Online Learning: What are Matrix A and Vector b?
* **Slide Bullets:**
  * Starts with zero knowledge: No offline dataset, no pre-trained delay model.
  * **What is Matrix A (7x7)?**
    * It is the feature covariance / design matrix: `A = sum(x_p * transpose(x_p)) + beta * Identity`.
    * Tracks how much the controller has explored each dimension of the 7D context space.
    * Its inverse, `inverse(A)`, represents the uncertainty of our estimate (larger uncertainty = higher exploration incentive).
  * **What is Vector b (7x1)?**
    * It is the accumulated feedback vector: `b = sum(x_p * observed_delay)`.
    * Accumulates the actual round-trip offloading delays observed for the tested context features.
  * **Online Ridge Regression on every frame `t`:**
    * Solves the linear weights in closed form: `theta = inverse(A) * b`.
    * When a split is tested, updates incrementally in real time:
      `A = A + x_p * transpose(x_p)`
      `b = b + x_p * observed_delay`
    * No retraining from scratch; state updates in order d^2 operations.
* **What to say:**
  > "To prove that this is true online learning, let's look at what matrix A and vector b actually are. The algorithm is running incremental ridge regression in real time. Matrix A is the feature covariance matrix. It starts as an identity matrix and accumulates the outer product of every context vector tested so far: x times transpose of x. It literally records where in the 7-dimensional feature space the system has gathered data. The inverse of A tells the controller how uncertain it is about each direction in the feature space. Vector b is the accumulated feedback vector. Every time an inference finishes and we measure the actual latency, we multiply that delay by the context vector x and add it to b. At each frame, the environmental parameter vector theta is simply computed as inverse(A) times b. There is no historical dataset stored in memory and no offline training loop. Every single frame directly updates A and b in real time."

---

## Slide 5: The 7D Context Vector

* **Slide Title:** Context Representation (7-Dimensional Feature Vector)
* **Slide Bullets:**
  * For each split point `p`, the context vector `x_p` has 7 features:
    1. `m_c`: Number of multiply-accumulate (MAC) units in remaining Conv layers.
    2. `m_f`: MAC units in remaining Fully Connected layers.
    3. `m_a`: MAC units in remaining Activation layers.
    4. `n_c`: Count of remaining Conv layers.
    5. `n_f`: Count of remaining Fully Connected layers.
    6. `n_a`: Count of remaining Activation layers.
    7. `psi_p`: Byte size of intermediate tensor sent over the wire.
  * Captures both server computation complexity and transmission data volume.
* **What to say:**
  > "Instead of learning each split point as an independent arm, muLinUCB uses a 7-dimensional context vector describing the suffix. It splits MAC counts and layer counts by layer type—convolutional, fully connected, and activation—plus the intermediate tensor size in bytes. This allows knowledge learned from testing one split point to transfer to the others."

---

## Slide 6: Special Features of muLinUCB

* **Slide Title:** Forced Sampling & Avoiding the Local Processing Trap
* **Slide Bullets:**
  * The Problem with standard LinUCB:
    * If the controller ever chooses pure local execution, offloading delay is zero and context is zero.
    * Matrix A and vector b receive zero updates.
    * Standard LinUCB gets trapped in pure local execution forever.
  * The Solution in muLinUCB:
    * **Forced Sampling:** Every `T^mu` frames (with `mu = 0.25`), pure local execution is disabled, forcing the system to test an offload split.
    * **Key-frame priority:** Critical frames receive weight `L = 0.8` (less exploration, play safe). Non-key frames receive `L = 0.2` (more exploration).
    * **Doubling trick:** Frame horizon `T` doubles dynamically (200, 400, 800...) to handle unknown streaming durations.
* **What to say:**
  > "The paper solved a critical flaw in classic LinUCB: if the controller ever decides to run 100% locally, no network packet is sent, so no delay feedback is generated, and LinUCB gets permanently stuck in local mode. muLinUCB fixes this with forced sampling: every T-to-the-mu frames, it forces the system to sample an offloading point. It also weights key frames to prioritize performance over exploration."

---

## Slide 7: Integrity: We Changed Zero Controller Code

* **Slide Title:** Controller Implementation Integrity
* **Slide Bullets:**
  * File in repository: `split_inference/controller/muLinUCB.py`
  * Exactly matches Algorithm 1 and Theorem 1 from WWW 2021:
    * Context extraction: `fillThetaContext()`
    * Matrix updates: `updateA_b()`
    * Decision rule: `getEstimationAction()`
    * Exploration parameter alpha and forced sampling frequency: exactly as defined in the paper.
  * **Zero lines of controller logic were changed.**
  * Only modifications in the repo were infrastructure wrappers (network socket communication and ARM64 compatibility for the Raspberry Pi).
* **What to say:**
  > "We did not modify the controller algorithm in any way. The file `muLinUCB.py` in our repo is a direct implementation of the paper's Algorithm 1. The equations for alpha, the matrix updates, the forced sampling rate of 0.25, and the decision rules are 100% original. The only code we wrote was the networking protocol to transmit tensors between the Pi and the PC."

---

## Slide 8: The Attack: Microarchitectural Contention

* **Slide Title:** Threat Model: Unprivileged Memory Contention
* **Slide Bullets:**
  * Attacker model:
    * Unprivileged user-space process running on the edge device (Raspberry Pi 4).
    * No root/sudo permissions.
    * Cannot inspect packets, cannot change model weights, cannot touch the controller process.
  * Attack vector:
    * Competes for shared hardware resources: the 1 MB shared L2 cache and the LPDDR4 memory bus.
  * How contention was injected (`stressor.py`):
    * Spawns 4 worker processes (1 per physical Cortex-A72 core).
    * Each worker continuously writes, multiplies, and copies large 32 MB arrays (`np.copyto`, `arr1 += arr2`).
    * Total working set (128 MB) far exceeds the 1 MB L2 cache, thrashing the cache and saturating the memory bus.
* **What to say:**
  > "Now, the attack. We assume an unprivileged process running on the Raspberry Pi alongside the split inference app. It has no root permissions and cannot touch the network. We created a memory stressor that spawns 4 worker processes, matching the Pi's 4 CPU cores. Each worker constantly accesses 32 megabytes of float arrays in a tight loop. Because 32 MB is much bigger than the Pi's 1 MB shared L2 cache, this constantly evicts cache lines and saturates the RAM memory bus."

---

## Slide 9: Why the Split Point Changes (The Attribution Error)

* **Slide Title:** Why the Decision Changes: The Controller's Attribution Error
* **Slide Bullets:**
  * **The Controller's Blind Spot:**
    * muLinUCB only observes total round-trip delay. It cannot tell where a delay occurred.
    * It mathematically assumes that any delay in offloading is caused by the **server** or the **network**.
  * **What Memory Contention Actually Delays:**
    * In PyTorch, sending activations requires tensor serialization, buffer allocation, and socket copying in RAM.
    * Memory bus saturation stalls these local memory operations, adding +47 ms to the measured round-trip time.
  * **The False Assumption:**
    * The controller sees full remote offload (`k = 4`, 1.81 Billion server MACs) jump from 18 ms to 65 ms.
    * The linear regression attributes this delay to the server, believing the server is severely overloaded (`theta` spikes).
  * **The Shift to Split `k = 2`:**
    * To "relieve" the perceived server overload, the controller chooses a split that offloads fewer MACs.
    * Split `k = 2` cuts the server workload in half (from 1.81G MACs to 819M MACs).
    * **The Trap:** The server was never overloaded! Forcing those layers onto the Raspberry Pi CPU creates a permanent +115 ms latency penalty.
* **What to say:**
  > "You might wonder: if the network is fast and the contention is on the local RAM and cache, why would the controller change its split decision? The reason is what we call an attribution error. The controller only measures total round-trip latency. When our stressor saturates the Pi's memory bus, local tensor serialization and socket memory copies stall, adding 47 ms to the round-trip measurement. But the controller doesn't know that. Because split point 4 has 1.8 billion MACs on the server, the algorithm falsely assumes that the server itself is overloaded. To 'relieve' the server, the controller shifts work onto the Pi, picking split point 2 where the server computes less than half the MACs. In trying to avoid an imaginary server bottleneck, the controller forces the Pi CPU to compute three neural network blocks, creating a massive, permanent 115 ms penalty."

---

## Slide 10: Why the Controller Stays Locked

* **Slide Title:** Why Does the Bad Decision Persist After the Attack?
* **Slide Bullets:**
  * Contention stops after 12 rounds (0 worker processes active).
  * Why doesn't the controller recover?
    1. **Variance reduction in A:** Once the controller picks `k = 2`, every subsequent frame adds more samples of `x_2 * transpose(x_2)` into matrix A.
    2. **Exploration bonus collapses:** As A grows, `alpha * sqrt(x * inv(A) * x)` shrinks to zero. The controller becomes mathematically 'confident' that `k = 2` is optimal.
    3. **Forced sampling is too sparse:** With `mu = 0.25`, forced sampling only tests other splits once every 4 to 8 frames (and even less as horizon doubles). A single unattacked sample cannot overturn the dozens of biased samples already accumulated in A and b.
  * Result: The controller is permanently stuck in split `k = 2`.
* **What to say:**
  > "Here is the key finding: after 12 rounds, we kill the attack completely. Contention drops to zero. But the controller never goes back to split 4. Why? Because every time it runs at split 2, it adds more data to matrix A. The confidence interval shrinks, meaning the bandit stops exploring and acts greedily. Forced sampling happens too rarely to overturn the biased data already baked into A and b. The attack has stopped, but the bad decision is permanently locked in."

---

## Slide 11: Hardware Setup & Communication Protocol

* **Slide Title:** Physical Hardware Setup & Network Protocol
* **Slide Bullets:**
  * Physical hardware:
    * **Edge Client:** Raspberry Pi 4 (Cortex-A72 quad-core, 4 GB RAM, Raspberry Pi OS 64-bit), IP `192.168.1.222`.
    * **Edge Server:** PC (Windows 11, PyTorch CPU/CUDA), IP `192.168.1.19`, Port `5005`.
    * **Connection:** Physical Gigabit Ethernet cable through LAN router (Zero artificial simulation).
  * Binary socket streaming protocol (`protocol.py`):
    * **Client -> Server (Activations):**
      `[Magic 'SPLT' (4B)][Payload Length (4B)][Split Index k (1B)][PyTorch Tensor Bytes]`
    * **Server -> Client (Predictions):**
      `[Magic 'RESP' (4B)][Payload Length (4B)][PyTorch Logits Bytes]`
    * Uses blocking exact-send and exact-receive loops over TCP.
* **What to say:**
  > "To validate this on real hardware, we connected a Raspberry Pi 4 to our PC workstation over a physical Gigabit Ethernet cable. We wrote a lightweight binary protocol over raw TCP sockets. The Pi sends a 9-byte header with the magic word 'SPLT', payload size, and the split point index k, followed by the serialized PyTorch activation tensor. The PC executes the remaining layers, packages the output logits with an 8-byte 'RESP' header, and sends it back. No artificial network latency emulation was used—this is 100% real physical timing."

---

## Slide 12: Hardware Results: Go/No-Go Trial Table

* **Slide Title:** Experimental Results (5 Independent Hardware Trials)
* **Slide Bullets:**
  * Trial-by-Trial Data:
    * **Trial 1:** k0 = 4 -> ka = 2 | Delta_phys = +47.07 ms | Delta_ctrl = +113.52 ms | **Amp: 2.41x**
    * **Trial 2:** k0 = 4 -> ka = 2 | Delta_phys = +49.25 ms | Delta_ctrl = +115.99 ms | **Amp: 2.36x**
    * **Trial 3:** k0 = 4 -> ka = 2 | Delta_phys = +41.75 ms | Delta_ctrl = +114.52 ms | **Amp: 2.74x**
    * **Trial 4:** k0 = 4 -> ka = 2 | Delta_phys = +54.48 ms | Delta_ctrl = +115.13 ms | **Amp: 2.11x**
    * **Trial 5:** k0 = 4 -> ka = 2 | Delta_phys = +46.17 ms | Delta_ctrl = +115.58 ms | **Amp: 2.50x**
  * Summary:
    * **Decision Manipulation Rate: 100.0%** (5/5 trials)
    * Mean Direct Physical Slowdown (`Delta_phys`): **+47.74 ms**
    * Mean Persistent Control Damage (`Delta_ctrl`): **+114.95 ms**
    * Mean Amplification Factor: **2.43x**
* **What to say:**
  > "Here are the numbers from our 5 trials on real hardware. In all 5 trials, the manipulation rate was 100%—every single trial shifted from split 4 to split 2. The direct physical slowdown during the attack averaged 47.74 ms. But the persistent damage after the attack stopped was 114.95 ms. The standard deviation of the post-attack overhead was under 1 ms across all trials. In other words, the control penalty is 2.43 times larger than the physical attack itself, and it stays there permanently."

---

## Slide 13: Hardware Lifecycle Plot

* **Slide Title:** Attack Lifecycle: Nominal vs. Attack vs. Post-Attack
* **Slide Bullets:**
  * *(Insert Figure: `results/figures/hardware_phase_trajectory.png`)*
  * **Phase 1 (Rounds 0-40):** Clean learning. Controller picks `k = 4` (full offload). Latency = ~18 ms.
  * **Phase 2 (Rounds 40-52):** 4 contention workers active. `Delta_phys = +47.1 ms`. Latency = ~65 ms. Controller belief gets corrupted.
  * **Phase 3 (Rounds 52-90):** Contention stopped (0 workers). Latency jumps to ~131 ms (`Delta_ctrl = +113.5 ms`).
  * Asymmetric Attack: A short 12-round attack causes permanent latency degradation.
* **What to say:**
  > "This figure shows the timeline of one trial. In Phase 1, the green line shows normal operation at 18 ms. In Phase 2, orange shows the 12 rounds where contention was active—latency increased by 47 ms. Then at round 52, contention ends. Instead of returning to green, latency jumps to the red line at 131 ms because the controller is now stuck running the first three blocks on the Pi CPU. This demonstrates an asymmetric denial of service: 12 frames of contention creates infinite frames of slowdown."

---

## Slide 14: Per-Trial Comparison & Summary Plots

* **Slide Title:** Physical Impact vs. Control Amplification
* **Slide Bullets:**
  * *(Insert Figures: `results/figures/hardware_delta_comparison.png` and `results/figures/hardware_amplification_summary.png`)*
  * Left: Direct comparison showing `Delta_ctrl` consistently dwarfs `Delta_phys`.
  * Right: Amplification ratio is consistently between 2.11x and 2.74x.
  * The attack is reproducible and statistically stable on physical ARM hardware.
* **What to say:**
  > "These two plots show the comparison across trials. In every single trial, the blue bar (persistent control damage) is more than double the orange bar (the physical contention). The amplification factor is solidly between 2.1x and 2.7x. This confirms that the vulnerability is not a simulation artifact—it happens reliably on physical hardware."

---

## Slide 15: Architectural Workload Shift

* **Slide Title:** Workload Shift Induced by the Contention
* **Slide Bullets:**
  * *(Insert Figure: `results/figures/hardware_split_decision_map.png`)*
  * **Nominal (`k = 4`):**
    * Stem + Layer 1 + Layer 2 + Layer 3 + Layer 4 + FC computed on Server GPU.
    * Edge Pi CPU usage: negligible.
  * **Manipulated (`k = 2`):**
    * Stem + Layer 1 + Layer 2 forced onto Raspberry Pi CPU.
    * Only Layer 3 + Layer 4 + FC computed on Server GPU.
  * The attacker successfully transforms an edge-assisted GPU pipeline into a CPU-bound bottleneck.
* **What to say:**
  > "This diagram illustrates the architectural impact. Under nominal conditions, the entire ResNet-18 model runs on the server GPU. But under attack, the controller decides to run the 7x7 conv stem, Layer 1, and Layer 2 on the Raspberry Pi CPU. By exploiting the online learning feedback loop, the attacker successfully turns a high-performance GPU offload pipeline into a sluggish, CPU-bound pipeline without ever needing administrative privileges."

---

## Slide 16: Summary & Next Steps

* **Slide Title:** Key Takeaways & Current Next Steps
* **Slide Bullets:**
  * **Key Takeaway:** Online learning split controllers (ANS / muLinUCB) have a fundamental security blind spot: they cannot distinguish between wireless network degradation and local microarchitectural contention.
  * **Consequence:** An unprivileged co-located process can cause 100% split manipulation and a persistent 2.4x latency penalty.
  * **Current & Next Steps:**
    1. Running full Stage 2 formal sweep across contention worker counts (1 to 8 workers) and models (ResNet-18 and MobileNetV2).
    2. Deploying inside the Bao Hypervisor on ARM64 to evaluate virtual machine isolation as a defense.
* **What to say:**
  > "To wrap up: Online learning controllers like muLinUCB are great for adapting to natural network shifts, but they have a severe blind spot: they assume all latency variations come from the network or server. Because they cannot detect local hardware cache and memory contention, an unprivileged attacker can easily manipulate them, resulting in a persistent 2.4x latency penalty. Right now, we are running our Stage 2 formal sweep across varying contention intensities and multiple neural networks, and our next step is to evaluate this setup inside the Bao hypervisor to see how hardware cache partitioning can mitigate this attack. Thank you, and I'm ready for questions."
