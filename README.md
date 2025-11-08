> **Sequence Models for Decision Making: Language Models to Trajectory Optimization**
## 🧭 Overview

This repository explores **Transformer** and **State Space Model (SSM)** architectures for **learning-based trajectory optimization**, specifically for autonomous spacecraft rendezvous and free-flyer operations.

The work investigates how **sequence models** — particularly **Transformers**, **S4D**, and **S6 (Mamba)** — can provide *warm-starts* to traditional optimization solvers, enabling **faster convergence**, **generalization**, and **safety guarantees** under onboard compute constraints.

---

## 📚 Motivation

Upcoming **in-orbit servicing, assembly, and logistics missions** require autonomous systems to compute safe, optimal trajectories onboard.

Traditional methods:
- ✅ *Closed-form algorithms*: lightweight, interpretable, but not scalable.  
- ✅ *Numerical optimization*: general and constraint-satisfying, but computationally heavy.

**Goal:** Develop a learning-based trajectory optimizer that combines:
- the **efficiency and generalization** of neural sequence models, and  
- the **safety and optimality** of numerical solvers.

---

## 🧩 Approach

### 1. Learning-Based Warm-Starting
We train high-capacity models to predict trajectory sequences — initial guesses for a numerical optimizer — minimizing computational overhead during onboard execution.

### 2. Models Studied
| Model | Type | Highlights |
|-------|------|-------------|
| **Transformer (ART)** | Contextual Sequence Model | Best overall performance and generalization |
| **S4D** | Structured State-Space (Diagonal) | Lightweight and stable initialization |
| **S6 (Mamba)** | Selective State-Space | Time-varying, hardware-aware, efficient |

### 3. Problem Setup
We evaluate all models on a **Freeflyer** simulation:
- 6-DOF **roto-translational dynamics**
- **Non-convex obstacle fields**
- **Fuel-minimization objective**
- Randomized initial and final conditions

---

## 🧠 Technical Details

- **Data Representation:**  
  Sequential trajectory tokens include state, action, rewards-to-go, and constraints-to-go.  
  Both *minimal* and *extended* representations were evaluated.

- **Optimization Formulation:**  
  Models learn conditional distributions  
  \( p_\theta(x_{1:N}, u_{1:N} \mid x_0) \)  
  to predict warm-start trajectories for a convex-relaxed solver.

- **Evaluation Metrics:**  
  - State RMSE  
  - Control RMSE  
  - Number of Sequential Convex Programming (SCP) iterations  
  - Generalization across unseen initial-final conditions

---

## 📊 Results

| Model | State RMSE ↓ | Control RMSE ↓ | SCP Iters ↓ | Generalization ↑ |
|:------|:-------------:|:---------------:|:-------------:|:----------------:|
| **Transformer (ART)** | 🥇 Best | 🥇 Best | 🥇 Fewest | 🥇 Highest |
| **S6 (Mamba)** | 2nd | 2nd | Moderate | Good |
| **S4D** | 3rd | 3rd | Highest | Lowest |

**Inference Speed:**  
SSM-based models (S4D, S6) exhibit inference times comparable to Transformers due to short sequence context (~100).  
Future improvements target GPU acceleration via CUDA Graphs.

---

## 🎥 Visualization

Trajectories are visualized in 3D free-flyer scenarios:
- Predicted vs. ground-truth paths  
- Obstacle avoidance regions  
- State evolution over time  

*(Have to add figures or link to visualization script here.)*

---

## 🧪 Key Insights

- **ART Transformer** consistently yields the best warm-start quality and generalization.
- **S6 (Mamba)** captures richer context than S4D, supporting adaptive sequence modeling.
- **Learning-based warm-starting** significantly reduces computational load during trajectory refinement.

---

## 🧾 References

1. Guffanti et al., *“Transformers for Trajectory Optimization with Application to Spacecraft Rendezvous,”* IEEE Aerospace Conference, 2024.  
2. Gu et al., *“Efficiently Modeling Long Sequences with Structured State Spaces (S4),”* ICLR, 2022.  
3. Gu et al., *“On the Parameterization and Initialization of Diagonal State Space Models (S4D),”* NeurIPS, 2022.  
4. Gu & Dao, *“Mamba: Linear-Time Sequence Modeling with Selective State Spaces,”* arXiv:2312.00752, 2023.

This repository uses https://github.com/Stanford-CAESAR/art-aeroconf24 , https://github.com/DavideCelestini/transformermpc-ral24, https://github.com/state-spaces/mamba repository codes to develop the model.
Thanks!
---

## 🛠️ Repository Structure 
<Yet to fill out>

