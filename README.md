# Team 12 — Noise: Impact of Noisy Training Data on Machine Learning Generalization

### A Case Study on the HTRU2 Pulsar Dataset

**Course:** 24SJPCCST503 — Machine Learning | Course Project (Investigation Challenge)  
**Team:** Team 12, Division B, Batch 2024–2028  
**Investigation Topic:** Noise  
**Institution:** St. Joseph's College of Engineering and Technology, Palai (Autonomous)

---

## Team Members

| Sl. No. | Name | Role / Contribution |
|---|---|---|
| 1 | Keerthana G Pillai | Dataset investigation and selection; feature-noise experiment |
| 2 | Lakshmy Byju | Label-noise experiment; transition-zone analysis |
| 3 | Sreehari Sreekumar Nair | Experimental design; notebook development; statistical analysis |
| 4 | Ben Mathew Ninan | Report preparation; visualization; results interpretation |

> Contributions should be confirmed by all team members before final submission.

---

## Research Question

**How does noisy data affect Machine Learning generalization?**

Specifically, this investigation examines whether corruption of input **features** and corruption of training **labels** affect a classifier's ability to generalize to unseen, clean data in the same way, and whether different learning algorithms respond differently to these two types of noise.

---

## Hypotheses

- **Feature noise** will cause a gradual and comparatively mild decline in generalization performance because perturbing the input features does not directly change the training targets.
- **Label noise** will cause a more severe decline because it directly corrupts the supervisory signal used for learning.
- Because HTRU2 is highly imbalanced (~9.16% positive class), class-balance-preserving label noise is expected to disproportionately affect the minority pulsar class as the noise level increases.
- Logistic Regression, Random Forest, and SVM will not necessarily degrade at the same rate under the different noise conditions.

---

## Dataset

**HTRU2 — High Time Resolution Universe Survey 2**  
**Source:** [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/372/htru2)

| Property | Value |
|---|---|
| Samples | 17,898 |
| Features | 8 continuous numerical features |
| Target | Binary — pulsar (1) / non-pulsar (0) |
| Class balance | 90.84% negative, 9.16% positive |
| Missing values | None |

HTRU2 is suitable for this investigation because it is a real-world binary classification problem involving astronomical signal measurements, providing a meaningful setting in which to study the effect of noisy training data on learning and generalization.

---

## Experimental Methodology

### Models

- Logistic Regression
- Random Forest (200 trees)
- Support Vector Machine with RBF kernel

Logistic Regression and SVM use standardized features through a preprocessing pipeline. SVM does **not** use `probability=True`; ROC-AUC and PR-AUC use ranking scores (`decision_function` for SVM). Each fit uses a fresh `sklearn.base.clone` of the template estimator.

### Train/Test Setup

- 80/20 stratified train-test split
- `random_state = 42` for the main Experiment 2 tables
- The **test set remains clean throughout all experiments**
- Noise is introduced only into the training data
- 10 independent noise realizations are used for each non-zero noise condition (the 0% condition is fit once)

Feature-noise percentages are Gaussian scale relative to each feature’s standard deviation. Label-noise percentages are the fraction of training labels flipped. **These are different units** and are never treated as interchangeable “doses.”

### Feature Noise

Zero-mean Gaussian perturbations are added to the training features, scaled by each feature's standard deviation:

`X_noisy = X + α · σ(X) · ε`, where `ε ~ N(0,1)`

Main feature-noise levels:

**0%, 2%, 5%, 10%, 15%, 20%, 30%**

### Label Noise

Symmetric, class-balance-preserving label flips are applied to the training labels, with equal numbers of 0→1 and 1→0 flips.

Main label-noise levels:

**0%, 2%, 4%, 6%, 8%, 10%**

A finer transition analysis uses:

**6%, 8%, 8.5%, 9%, 9.5%, 10%**

### Evaluation Metrics

The models are evaluated on the untouched test set using:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC (average precision; added in the shared library / Experiment 3)

F1-score, ROC-AUC, and PR-AUC are particularly important because the HTRU2 dataset is strongly class-imbalanced. F1 is a thresholded metric; ROC-AUC / PR-AUC are ranking metrics. They need not move together.

---

## Results

### Clean Baseline

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9796 | 0.9443 | 0.8262 | 0.8813 | 0.9710 |
| Random Forest | 0.9810 | 0.9221 | 0.8659 | 0.8931 | 0.9664 |
| SVM | 0.9807 | 0.9481 | 0.8354 | 0.8882 | 0.9542 |

### Feature Noise — F1

| Noise % | LR F1 | RF F1 | SVM F1 |
|---:|---:|---:|---:|
| 0 | 0.8813 | 0.8931 | 0.8882 |
| 2 | 0.8804 | 0.8938 | 0.8885 |
| 5 | 0.8791 | 0.8936 | 0.8854 |
| 10 | 0.8790 | 0.8892 | 0.8839 |
| 15 | 0.8770 | 0.8866 | 0.8856 |
| 20 | 0.8756 | 0.8866 | 0.8846 |
| 30 | 0.8695 | 0.8788 | 0.8799 |

Feature noise produced relatively small changes in F1 across the tested range.

### Label Noise — F1

| Noise % | LR F1 | RF F1 | SVM F1 |
|---:|---:|---:|---:|
| 0 | 0.8813 | 0.8931 | 0.8882 |
| 2 | 0.8460 | 0.8830 | 0.8795 |
| 4 | 0.8183 | 0.8626 | 0.8686 |
| 6 | 0.7669 | 0.7997 | 0.8370 |
| 8 | 0.6275 | 0.6343 | 0.7332 |
| 8.5 | 0.5812 | 0.5881 | 0.6307 |
| 9 | 0.5274 | 0.5504 | 0.4628 |
| 9.5 | 0.4571 | 0.4706 | 0.2493 |
| 10 | 0.3387 | 0.4123 | 0.0693 |

Label noise produced substantially larger F1 degradation than feature noise, particularly in the 8–10% transition region.

### ROC-AUC Under Label Noise

An important part of the analysis is that F1 and ROC-AUC do not deteriorate identically. For example, at 10% label noise:

| Model | F1 | ROC-AUC |
|---|---:|---:|
| Logistic Regression | 0.3387 | 0.9708 |
| Random Forest | 0.4123 | 0.9481 |
| SVM | 0.0693 | 0.7547 |

This shows why relying on accuracy or a single metric would not fully describe the effect of noisy labels.

The SVM ROC-AUC numbers above are from the original Experiment 2 run. The current notebook and library score SVM with `decision_function` (no Platt scaling), so a full re-run can change ranking AUC slightly; F1 from `predict()` is unchanged.

### Minority-Class Label Survival

The transition analysis also tracks the survival of originally positive (pulsar) labels. Mean original-pulsar survival decreases from approximately **67.3% at 6% label noise** to **45.4% at 10% label noise**.

This provides a direct diagnostic for why positive-class detection and F1 deteriorate as label noise approaches the upper end of the tested range.

### Variance Analysis

The repeated experiments also allow variation between noise realizations to be examined. A Levene's test comparing F1 variance at 6% and 9% label noise gave:

| Model | SD at 6% | SD at 9% | Levene p-value |
|---|---:|---:|---:|
| Logistic Regression | 0.0051 | 0.0129 | 0.054 |
| Random Forest | 0.0154 | 0.0208 | 0.385 |
| SVM | 0.0121 | 0.1043 | 0.004 |

The SVM result indicates a statistically significant increase in F1 variance between these two tested noise levels. The specific mechanism behind this behavior was not directly isolated in the experiment and is therefore treated as an interpretation rather than a causal conclusion.

---

## Key Findings

1. **Feature noise and label noise have very different effects.** Feature noise caused relatively small F1 changes even at the highest tested feature-noise level, whereas label noise caused much larger degradation.
2. **Label noise strongly affects minority-class detection.** As label noise increases, the survival of originally positive labels decreases, accompanied by a strong reduction in positive-class recall and F1.
3. **The effect becomes particularly pronounced in the 8–10% transition region.** The finer-grained experiment shows progressive deterioration rather than a single exact failure threshold.
4. **Different metrics reveal different aspects of degradation.** F1 can deteriorate substantially while ROC-AUC remains relatively high for some models, so both threshold-based and ranking-based metrics are informative.
5. **Different models respond differently to noisy training data.** The magnitude and variability of degradation depend on both the noise type and the classifier.

---

## Follow-up experiments (Experiment 3 Extensions)

Experiment 3 systematically investigates the underlying mechanisms, limitations, and practical implications identified in Experiment 2. All implementations reside in the modular `src/htru2_noise/` package and interactive notebook `notebooks/htru2_experiment3_extensions.ipynb`:

### 1. The Decision Threshold Paradox & Operating Point Recovery
- **Empirical Finding:** Under 10% symmetric label noise, Logistic Regression's default $F_1$ collapses from **0.8813 down to 0.3387**, yet its ranking metrics remain virtually intact ($ROC\text{-}AUC = 0.9708$, $PR\text{-}AUC = 0.9316$).
- **Underlying Mechanism:** Symmetric label noise flips equal numbers of 0s and 1s, polluting the minority pulsar class with false positives. This distorts the posterior probability calibration and shifts the decision threshold away from the default operating point ($0.0$ margin / $0.5$ probability).
- **Practical Recovery:** By adjusting the decision threshold on a small clean validation set (shifting LR threshold from $0.0$ to $-1.84$), test $F_1$ is **restored to 0.8926** (matching the clean baseline!). Discriminant feature representation is preserved; the apparent collapse is primarily an operating-point calibration failure.

### 2. Realization-to-Realization Variance & Statistical Testing
- **Empirical Finding:** Levene's test for equality of variances across noise draws confirms that SVM exhibits a statistically significant variance explosion ($p = 0.004$) between 6% and 9% label noise ($SD = 0.0121 \to 0.1043$).
- **Model Contrast:** Random Forest ($p = 0.385$) and Logistic Regression ($p = 0.054$) exhibit no statistically significant variance surge, demonstrating superior realization-level stability.
- **Underlying Mechanism:** The SVM RBF decision boundary is strictly anchored to support vectors near the margin; flipping minority instances near the boundary induces large topological distortions. In contrast, Random Forest's bagging averages out local perturbations.

### 3. Partition Invariance Across Independent Splits
- **Empirical Finding:** Evaluating label noise across 5 independent stratified 80/20 train/test splits confirms that the degradation trajectories, transition zone behavior, and model rankings are invariant to the specific random partition (confirmed by tight 95% confidence intervals).

### 4. Cost-Sensitive Learning (`class_weight='balanced'`) Under Noise
- **Empirical Finding:** While `class_weight='balanced'` boosts recall on clean imbalanced data, applying it under label noise is **counter-productive**.
- **Underlying Mechanism:** At 10% label noise, balanced weighting heavily penalizes false positives (non-pulsars flipped into the minority class), driving precision down to $<25\%$ and severely depressing $F_1$.
- **Recommendation:** Do not combine naive inverse-frequency class reweighting with noisy labels on imbalanced data.

### 5. Directional Asymmetry: Pollution vs Erasure
- **Empirical Finding:** Majority-to-minority noise (pulsar pollution) degrades $F_1$ far more aggressively than minority-to-majority noise (pulsar erasure).
- **Underlying Mechanism:** Because pulsars constitute only ~9.16% of samples, flipping even 4% of majority non-pulsars into pulsars almost doubles the positive class with spurious signals, ruining precision. In contrast, erasing minority samples starves recall but leaves remaining positives pristine.
- **Recommendation:** In annotation campaigns, it is safer to be conservative (allowing false negatives) than to admit false positive contaminants into a rare class.

### 6. Probability Calibration & Reliability
- **Empirical Finding:** Label noise dramatically degrades posterior probability calibration (increasing Brier score loss) and causes probability underconfidence on positive instances.

---

## Quickstart & Reproducibility

### 1. Installation
Clone the repository and install the standard dependencies:
```bash
git clone https://github.com/Keerthana-G-Pillai/HTRU2-Noise-Generalization.git
cd HTRU2-Noise-Generalization

python -m pip install -r requirements.txt
```

### 2. Interactive Notebooks (Recommended)
Open either notebook in VS Code, JupyterLab, or Google Colab and click **Run All**:
- `notebooks/htru2_experiment2_FINAL.ipynb`: Primary course project results, clean baseline, feature noise sweeps, and label noise transition curves.
- `notebooks/htru2_experiment3_extensions.ipynb`: Diagnostic follow-up extensions (threshold shifts, Levene variance test, partition invariance, cost-sensitive weighting, and probability calibration).

### 3. (Optional) Command-Line & Tests
You can also run experiments directly from the terminal or verify the test suite:
```bash
# Fast smoke run (~20 seconds)
python scripts/run_experiments.py --quick

# Standard run with all models (~2-3 minutes)
python scripts/run_experiments.py --standard

# Run unit tests
python -m pytest
```

## Limitations

- The findings are based on the HTRU2 dataset and its approximately 9.16% positive-class imbalance; results may differ for datasets with different class distributions.
- Symmetric, class-balance-preserving label noise is an idealized noise model. Asymmetric noise is included as a robustness check, not as a complete model of real annotation error.
- The main reported tables still use one primary 80/20 split (`random_state=42`). Repeated splits are an extension, not a replacement of those tables.
- Default/lightly tuned model hyperparameters were used; extensive hyperparameter tuning could affect the observed robustness patterns.
- The SVM variance behavior is observed statistically, but its specific underlying mechanism was not directly isolated by the experiment.

---

## Repository Structure

```text
HTRU2-Noise-Generalization/
├── README.md
├── pyproject.toml
├── requirements.txt
├── notebooks/
│   ├── htru2_experiment2_FINAL.ipynb
│   └── htru2_experiment3_extensions.ipynb
├── src/htru2_noise/
├── scripts/run_experiments.py
├── tests/
├── figures/
├── results/
└── data/README.md
```
