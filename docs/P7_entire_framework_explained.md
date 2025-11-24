# P7 Adaptive Adversary Framework - Technical Documentation

**Owner:** P7 (Matteo)  
**Project:** Randomness in Algorithm Trading - BSML  
**Week:** 3  
**Date:** November 2025

---

## Executive Summary

The adaptive adversary framework is a machine learning-based system designed to evaluate the predictability of algorithmic trading policies. By training classifiers to predict future trading behavior, we simulate an adversarial actor attempting to front-run or anticipate trades. This framework enables quantitative comparison of different randomization strategies' effectiveness at evading pattern detection.

---

## 1. Adversary Concept

### 1.1 What is an Adversary?

In algorithmic trading, an **adversary** is any market participant who:
- Observes historical trading patterns
- Uses pattern recognition to predict future trades
- Exploits predictions for profit (e.g., front-running, adverse selection)

### 1.2 Threat Model

Our adversary has access to:
- **Historical trade data:** Past execution timestamps, directions, quantities
- **Market context:** Price movements, volatility, calendar features
- **Machine learning capabilities:** Can train sophisticated pattern recognition models

Our adversary **cannot** access:
- Future information (no time travel)
- Internal decision logic or signals
- Non-public order book information

---

## 2. Adversary Framework Architecture

### 2.1 Overall Design
```
┌─────────────────────────────────────────────────────────────┐
│                    ADVERSARY FRAMEWORK                       │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  1. BASELINE POLICY        →  Generate trades (predictable) │
│  2. RANDOMIZED POLICIES    →  Generate trades (randomized)  │
│                                                              │
│  3. FEATURE EXTRACTION     →  Create prediction datasets    │
│                                                              │
│  4. ADVERSARY TRAINING     →  Train on baseline patterns    │
│                                                              │
│  5. PATTERN EVALUATION     →  Test on all policies          │
│                                                              │
│  6. COMPARISON             →  Identify best randomization   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### 2.2 Component Breakdown

#### **Phase 1: Trade Generation**
- Generate baseline trades (deterministic daily execution)
- Generate randomized trades using three strategies:
  - **Uniform:** Independent random noise on price and time
  - **Pink Noise:** 1/f correlated price noise (persistent drifts)
  - **Ornstein-Uhlenbeck:** Mean-reverting price noise

#### **Phase 2: Feature Engineering**
For each trading day, extract temporal and market features:
- **Temporal patterns:** Days since last trade, inter-trade intervals, consecutive trading days
- **Trading patterns:** Buy/sell ratios, direction changes, quantity statistics
- **Market context:** Recent returns, volatility, price levels
- **Calendar features:** Day of week, day of month

#### **Phase 3: Prediction Task**
**Binary classification:** Given recent trading history, predict if there will be a trade tomorrow.
- **Label = 1:** Trade occurs tomorrow
- **Label = 0:** No trade tomorrow

#### **Phase 4: Adversary Model**
**Gradient Boosting Classifier** with:
- **Training data:** Baseline policy (learns predictable patterns)
- **Algorithm:** Gradient Boosting (captures non-linear patterns)
- **Evaluation metric:** Area Under ROC Curve (AUC)
  - AUC = 1.0: Perfect prediction (completely predictable)
  - AUC = 0.5: Random guessing (completely unpredictable)

#### **Phase 5: Cross-Policy Evaluation**
Train **once** on baseline, test on **all policies**:
```
Baseline data → Train Adversary → Trained Model
                                        ↓
                        Test on: Baseline (high AUC expected)
                        Test on: Uniform (lower AUC desired)
                        Test on: Pink Noise (lower AUC desired)
                        Test on: OU (lower AUC desired)
```

**Key insight:** If a randomized policy achieves low AUC when tested with a predictor trained on baseline patterns, it means the randomization successfully broke the predictable structure.

---

## 3. What the Adversary Does

### 3.1 Training Phase

**Objective:** Learn baseline trading patterns

**Process:**
1. Observe 60% of baseline trade history (training set)
2. Extract features from each observation
3. Train gradient boosting model to predict next-day trades
4. Validate on 20% of baseline data (validation set)

**Outcome:** Model learns that baseline:
- Trades every consecutive day (feature: `consecutive_trade_days`)
- Trades on specific days of week (feature: `day_of_week`)
- Has regular inter-trade intervals (feature: `avg_interval`)

### 3.2 Testing Phase

**Objective:** Attempt to predict each policy's trades

**Process:**
1. Apply trained model to each policy's validation set
2. Generate predictions: "Will this policy trade tomorrow?"
3. Compare predictions to actual trades
4. Calculate AUC for each policy

**Outcome:** Quantifies how well adversary can anticipate each policy's behavior

### 3.3 Interpretation

| AUC Range | Interpretation | Adversary Capability |
|-----------|----------------|---------------------|
| 0.90-1.00 | Highly predictable | Can accurately anticipate trades → high front-running risk |
| 0.70-0.89 | Moderately predictable | Some pattern detection possible → moderate risk |
| 0.55-0.69 | Slightly predictable | Weak patterns remain → low risk |
| 0.50-0.54 | Unpredictable | No better than random guessing → minimal risk |

---

## 4. Adaptive Loop Mechanism

### 4.1 Motivation

Initial randomization parameters may be:
- **Too weak:** Patterns still detectable (high AUC)
- **Too strong:** Excessive noise, potential strategy degradation

**Solution:** Adaptive loop automatically tunes randomization strength.

### 4.2 Algorithm
```python
Initialize: price_noise = 0.03, time_noise = 30 minutes
Target: AUC < 0.55 (unpredictable)

For each iteration:
    1. Generate trades with current noise parameters
    2. Train adversary on baseline
    3. Test adversary on all policies
    4. Get best (lowest) AUC among randomized policies
    
    5. Decision:
       If best_AUC > 0.65:  # Still too predictable
           → INCREASE noise by 25%
       Elif best_AUC < 0.55:  # Achieved target
           → SUCCESS, converge
       Else:  # In target range
           → HOLD parameters
    
    6. Repeat until convergence
```

### 4.3 Convergence Criteria

Stop when:
- Best AUC < 0.55 for 3 consecutive iterations (success), OR
- Maximum 10 iterations reached (report final state)

---

## 5. Key Metrics

### 5.1 Primary Metric: AUC (Area Under ROC Curve)

**Definition:** Probability that adversary ranks a random positive instance (trade day) higher than a random negative instance (no-trade day)

**Why AUC?**
- Threshold-independent: Robust across different decision boundaries
- Handles class imbalance: Valid even when trades are rare
- Probabilistic interpretation: Directly measures ranking quality

### 5.2 Secondary Metrics

- **AUC Reduction:** `(Baseline_AUC - Policy_AUC) / Baseline_AUC`
  - Percentage reduction in predictability
  - Higher = better randomization
  
- **Accuracy:** Percentage of correct predictions
  - Less informative due to class imbalance
  
- **F1 Score:** Harmonic mean of precision and recall
  - Focuses on positive class (trade days)

---

## 6. Implementation Details

### 6.1 Technology Stack

- **Language:** Python 3.10+
- **ML Framework:** scikit-learn (GradientBoostingClassifier)
- **Data Processing:** pandas, numpy
- **Evaluation:** sklearn.metrics (roc_auc_score, accuracy_score, f1_score)

### 6.2 Model Configuration
```python
GradientBoostingClassifier(
    n_estimators=100,      # 100 boosting stages
    max_depth=5,           # Trees limited to depth 5
    learning_rate=0.1,     # Conservative learning rate
    random_state=42        # Reproducible results
)
```

**Rationale:**
- Gradient boosting captures non-linear patterns and feature interactions
- Conservative hyperparameters prevent overfitting
- 5-fold cross-validation during training ensures robustness

### 6.3 Data Splits

| Split | Percentage | Purpose |
|-------|-----------|---------|
| Train | 60% | Model learning |
| Validation | 20% | Hyperparameter tuning, AUC measurement |
| Test | 20% | Final evaluation (held out) |

**Temporal ordering preserved:** Earlier data for training, later data for validation/test (prevents look-ahead bias)

---

## 7. Limitations and Assumptions

### 7.1 Assumptions

1. **Stationarity:** Market conditions remain relatively stable
2. **Independence:** Adversary observes but doesn't influence trades
3. **Information:** Adversary has historical data but no privileged information
4. **Rationality:** Adversary uses optimal ML techniques available

### 7.2 Limitations

1. **Single adversary model:** Real adversaries may use diverse approaches
2. **No strategic interaction:** Adversary doesn't adapt in response to randomization
3. **Synthetic evaluation:** Uses historical prices, not live market impact
4. **Feature selection:** Limited to engineered features (adversary might discover others)

### 7.3 Future Extensions

- **Ensemble adversaries:** Multiple prediction models
- **Sequence models:** LSTM/Transformer for temporal dependencies
- **Market impact:** Incorporate execution cost models
- **Adaptive adversaries:** Adversary that updates as it observes randomized trades

---

## 8. Validation and Robustness

### 8.1 Cross-Validation

All models use **5-fold cross-validation** during training to:
- Reduce overfitting risk
- Estimate generalization performance
- Validate feature importance consistency

### 8.2 Statistical Significance

Results are considered significant when:
- AUC difference > 0.05 (5 percentage points)
- Consistent across multiple random seeds
- Replicated across different time periods

### 8.3 Sensitivity Analysis

Framework tested under:
- Different lookback windows (3, 5, 7 days)
- Various policy parameters
- Multiple dataset sizes

---

## 9. Ethical Considerations

### 9.1 Defensive Purpose

This framework is designed for **defensive evaluation**, not offensive exploitation:
- Helps traders protect against front-running
- Evaluates randomization effectiveness
- Improves market fairness

### 9.2 Responsible Disclosure

Results shared only with:
- Academic research community
- Trading firms for self-defense
- Not published in forums accessible to malicious actors

---

## 10. Conclusion

The adaptive adversary framework provides a rigorous, quantitative methodology for evaluating trading policy vulnerability to pattern-based attacks. By simulating an intelligent adversary with machine learning capabilities, we can:

1. **Measure predictability** across different randomization strategies
2. **Identify optimal parameters** through adaptive tuning
3. **Compare strategies** objectively using AUC metrics
4. **Validate effectiveness** of anti-front-running measures

This framework bridges theoretical randomization proposals with empirical validation, enabling evidence-based decisions in algorithmic trading system design.

---

## References

- Gradient Boosting: Friedman (2001), "Greedy Function Approximation: A Gradient Boosting Machine"
- ROC Analysis: Fawcett (2006), "An Introduction to ROC Analysis"
- Adversarial ML: Goodfellow et al. (2014), "Explaining and Harnessing Adversarial Examples"
- Algorithmic Trading: Cartea et al. (2015), "Algorithmic and High-Frequency Trading"

---

**Document Version:** 1.0  
**Last Updated:** November 23, 2025  
**Status:** Final
