# Randomness-in-Algorithm-Trading---BSML

## Abstract

Deterministic algorithmic execution strategies, while statistically efficient, are vulnerable to exploitation by adversaries who can infer and anticipate order flows.  
This project develops a framework for **stochastic execution policies** to defend against such adversarial learning dynamics.

We evaluate three randomization mechanisms:
- **Uniform randomization**
- **Ornstein–Uhlenbeck (OU)** mean-reverting noise
- **Pink (1/f)** spectral noise

using both:
- static machine-learning binary classifiers, and  
- regression-based adversaries predicting execution prices.

According to the paper’s main findings :contentReference[oaicite:1]{index=1}:
- OU policy reduces adversarial predictability by **27%** (AUC 0.78 → 0.57)  
- OU improves implementation shortfall by **14.8 bps**, driven by **10.6 bps decrease in adverse selection**  
- Portfolio Sharpe improves **19%** (0.94 → 1.12)  
- OU reaches adversarial safety in **5 iterations** (vs 15–20 for alternatives)  
- Estimated benefit: **$37M annual** at $100M daily execution volume  
- All policies satisfy strict exposure and turnover constraints during backtests

This establishes that adversarial robustness requires **dynamic, game-theoretic adaptation**, not static defense.

---
| Name                     | Role                                       | Profiles                    |
| ------------------------ | ------------------------------------------ | --------------------------- |
| **Vincenzo Della Ratta** | Infrastructure, Pipeline, Backtesting (P3) | [GitHub](#) · [LinkedIn](#) |
| **Preslav Georgiev**     |                                            | [GitHub](#) · [LinkedIn](#) |
| **Matteo Roda**          |                                            | [GitHub](#) · [LinkedIn](#) |
| **Rayi Makori**          |                                            | [GitHub](#) · [LinkedIn](#) |
| **Hunor Csenteri**       |                                            | [GitHub](#) · [LinkedIn](#) |
| **Neel Roy**             |                                            | [GitHub](#) · [LinkedIn](#) |
| **David Livshits**       |                                            | [GitHub](#) · [LinkedIn](#) |

**Full Paper (PDF)**: attached in repo (docs/BSML_final.pdf)

**BSML Website**: [https://bsmachinelearning.com](https://bsmachinelearning.com/)
