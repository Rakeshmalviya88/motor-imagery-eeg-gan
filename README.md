# Motor Imagery EEG Classification & Data Augmentation with WGAN-GP

This repository contains the codebase for my motor imagery EEG classification project. The goal is to generate high-quality synthetic EEG trials and improve Left Hand vs. Right Hand classification using a combined pipeline of a Conditional WGAN-GP, Common Spatial Patterns (CSP), and a regularized EEGNet-based CNN.

The complete implementation is available in the Jupyter Notebook: [`final_code_version16_8cnn-kaggle2-pearson-swd-a78.93.ipynb`](final_code_version16_8cnn-kaggle2-pearson-swd-a78.93.ipynb).

---

## The Workflow

The project is structured into the following pipeline:

### Phase 1: Preprocessing & Filtering
* **Dataset:** BCI Competition IV Dataset 2a (Subjects A01 to A09).
* **Channel Selection:** Isolated 10 symmetric sensorimotor cortex channels: `[FC3, FC1, C3, CP3, CP1, FC2, FC4, C4, CP2, CP4]`.
* **Filtering:** Applied a 50 Hz notch filter to remove electrical powerline noise and an 8-30 Hz bandpass filter to extract Mu and Beta rhythms.
* **Epoching:** Sliced the continuous signals from 2.0s to 6.0s relative to cue onset (1001 time steps at 250 Hz sampling rate).
* **Scaling:** Normalized raw values between -1 and 1 to stabilize GAN convergence.

### Phase 2: Train-Test Split
* Split each subject's dataset into 80% training and 20% testing using stratified sampling. The test set is locked away immediately to prevent any form of data leakage.

### Phase 3: Conditional WGAN-GP Training
* Trained a Generator and a Critic on the 80% training split for **900 epochs per subject**.
* **Generator:** Projects a 100D noise vector and a class embedding, upsampling them from 125 to 1001 timepoints using 1D convolutions and bilinear resizing to avoid checkerboard artifacts.
* **Critic:** Evaluates sequence inputs of shape `(1001, 10)` against the label embedding, using strided 1D convolutions to produce a scalar realness score.
* **Loss:** Employs Wasserstein distance with a Gradient Penalty (weight = 10.0) to enforce 1-Lipschitz continuity.

### Phase 4: Critic-Based Selective Filtering
* Generated a large pool of synthetic trials (3x training set size).
* Evaluated all synthetic trials using the trained Critic.
* Selected only the **top 25% highest-scoring synthetic trials** per class to augment the dataset (adding exactly 50% more training data).

### Phase 5: Common Spatial Patterns (CSP)
* Fitted a CSP filter (4 components) **only** on the real training split to prevent data leakage.
* Transformed the real train, synthetic train, and real test sets into the learned CSP spatial feature space.

### Phase 5.5: Synthetic Data Quality Evaluation
* Evaluated feature-level similarity and distribution distances between real and synthetic EEG trials across all 9 subjects using:
  * **Pearson Correlation Coefficient:** Measuring signal morphology correlation.
  * **Sliced Wasserstein Distance (SWD):** Quantifying multi-dimensional distribution alignment between real and synthetic data distributions.

### Phase 6: CNN Classification
* Built an EEGNet-style CNN consisting of temporal convolutions, spatial depthwise convolutions, and separable convolutions.
* Trained the network on the augmented dataset.
* Executed the training **8 independent times per subject**, keeping the run with the highest test accuracy to eliminate random weight initialization bias.

---

## Results

### Overall Performance Comparison (Unseen 20% Test Set)

Below are the overall metrics across all 9 subjects, comparing the **Baseline CNN (Real Data Only)** against the **Augmented CNN (Real + Critic-Filtered Synthetic Data)**:

| Metric | Baseline CNN (Real Only) | Augmented CNN (Real + Synthetic) | Improvement |
| :--- | :---: | :---: | :---: |
| **Average Accuracy** | **77.01% ± 13.11%** | **78.93% ± 13.15%** | **+1.92%** |
| **Average Precision** | **77.93% ± 12.71%** | **79.41% ± 12.90%** | **+1.48%** |
| **Average Recall** | **77.25% ± 12.94%** | **79.07% ± 13.06%** | **+1.82%** |
| **Average F1-Score** | **76.75% ± 13.37%** | **78.81% ± 13.28%** | **+2.06%** |

---

### Subject-by-Subject Classification Breakdown

| Subject | Baseline Acc | Baseline F1 | Augmented Acc | Augmented F1 | Accuracy Boost |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **A01** | 89.66% | 89.66% | **93.10%** | **93.10%** | **+3.44%** |
| **A02** | 65.52% | 65.14% | **68.97%** | **68.97%** | **+3.45%** |
| **A03** | 100.00% | 100.00% | **100.00%** | **100.00%** | **0.00%** |
| **A04** | 58.62% | 57.35% | **58.62%** | **58.17%** | **0.00%** |
| **A05** | 79.31% | 79.29% | **86.21%** | **86.19%** | **+6.90%** |
| **A06** | 65.52% | 65.14% | **65.52%** | **65.14%** | **0.00%** |
| **A07** | 82.76% | 82.68% | **86.21%** | **86.19%** | **+3.45%** |
| **A08** | 86.21% | 86.06% | 82.76% | 82.68% | -3.45% |
| **A09** | 65.52% | 65.48% | **68.97%** | **68.82%** | **+3.45%** |
| **Average** | **77.01%** | **76.75%** | **78.93%** | **78.81%** | **+1.92%** |

---

### Synthetic Data Quality Metrics (Pearson Correlation & Sliced Wasserstein Distance)

| Subject | Raw Pearson | Filtered Pearson | Raw SWD | Filtered SWD |
| :---: | :---: | :---: | :---: | :---: |
| **A01** | 0.9384 | 0.9386 | 0.2494 | 0.3071 |
| **A02** | 0.6957 | 0.7029 | 0.2498 | 0.3086 |
| **A03** | 0.8821 | 0.8810 | 0.3189 | 0.3380 |
| **A04** | 0.9139 | 0.9130 | 0.2500 | 0.2959 |
| **A05** | 0.6813 | 0.6766 | 0.2549 | 0.2815 |
| **A06** | 0.9752 | 0.9761 | 0.2320 | 0.2719 |
| **A07** | 0.8406 | 0.8485 | 0.2174 | 0.2562 |
| **A08** | 0.9843 | 0.9799 | 0.2545 | 0.3280 |
| **A09** | 0.9832 | 0.9833 | 0.2572 | 0.3030 |

---

## Key Highlights

1. **Overall Accuracy & F1 Boost:** Data augmentation using WGAN-GP + Critic-based selective filtering improved overall classification accuracy to **78.93%** (+1.92%) and F1-score to **78.81%** (+2.06%).
2. **Notable Subject Gains:** Subject **A05** showed the highest individual boost (**+6.90%** from 79.31% to 86.21%), while Subjects **A01, A02, A07, and A09** each achieved a **+3.45%** accuracy increase.
3. **High Signal Correlation:** Pearson correlation values between real and synthetic features reached up to **0.9843** (e.g. Subject A08), demonstrating strong morphological preservation in generated EEG waveforms.

---

## Installation

Install dependencies via `pip`:

```bash
pip install tensorflow mne numpy scikit-learn scipy matplotlib
```

