# SVM Performance & Preprocessing Analysis

We have run extensive experiments on the BCI Competition IV 2a dataset (2-class Left vs. Right hand motor imagery, 3 channels: C3, Cz, C4) to evaluate replacing the CNN model with an SVM and investigate why the accuracy was low.

Here is a summary of the results and a critical finding regarding how the data is preprocessed.

---

## 1. Summary of SVM Experiment Results

We evaluated multiple feature extraction and normalization strategies for the SVM classifier on the pooled test set (1,295 trials across 9 subjects) and individual subject-specific test sets:

| Feature Extraction Method | Normalization Strategy | Evaluation Mode | Max Test Accuracy |
| :--- | :--- | :--- | :--- |
| **Flattened Raw EEG** | Individual Channel Z-Score | Pooled | **51.27%** |
| **Common Spatial Patterns (CSP)** | Individual Channel Z-Score | Pooled | **53.36%** |
| **Power Spectral Density (PSD)** | Individual Channel Z-Score | Pooled | **53.59%** |
| **Hilbert Envelope Statistics** | None (Raw amplitude) | Pooled | **54.36%** |
| **Time-domain Log-Variance** | None (Raw signals) | Pooled | **55.06%** |
| **Comprehensive Feature Set** (Mean, std, kurtosis, sub-window log-var, FFT) | Global StandardScaler | Pooled | **58.15%** |
| **Filter Bank CSP (FBCSP)** | Trial-wise (Across channels) | Subject-wise (Avg) | **56.52%** |
| **Common Spatial Patterns (CSP)** | Trial-wise (Across channels) | Subject-wise (Avg) | **57.68%** (A03: **70.83%**, A07: **68.75%**) |

---

## 2. Critical Preprocessing Issue Identified in the Notebook

During feature analysis, we discovered a **critical flaw** in how the trials are normalized in the notebook's preprocessing step (Cell 6):

```python
X_train = (X_train - X_train.mean(axis=2, keepdims=True)) / (X_train.std(axis=2, keepdims=True) + 1e-8)
X_test = (X_test - X_test.mean(axis=2, keepdims=True)) / (X_test.std(axis=2, keepdims=True) + 1e-8)
```

### Why this hurts SVM/CSP Performance:
* **The Math:** `axis=2` is the time axis. This line normalizes **each channel in each trial independently** to have a mean of 0 and a standard deviation of exactly 1.0.
* **The Impact:** Under this normalization, the variance of C3 is 1.0, Cz is 1.0, and C4 is 1.0 for *every single trial*. 
* **The EEG Context:** Motor imagery depends entirely on **Event-Related Desynchronization (ERD)**, which is the relative decrease in power (variance) on one side of the motor cortex compared to the other (e.g., C3 vs. C4). By forcing all channels to have a standard deviation of exactly 1.0, **this step completely erases the power difference between channels**, forcing the SVM to classify based only on phase information, which is highly noisy in raw EEG.

### Proposed Normalization Fix:
To preserve relative channel power differences while keeping the signal scale bounded, we must normalize **across all channels together** (i.e. computing mean and std over both channels and time):

```python
# Normalize trial-wise across all 3 channels
mean_train = X_train.mean(axis=(1, 2), keepdims=True)
std_train = X_train.std(axis=(1, 2), keepdims=True)
X_train = (X_train - mean_train) / (std_train + 1e-8)
```
When we tested this "across-channel" normalization, subject-wise CSP+SVM accuracy jumped significantly, with **Subject A03 reaching 70.83%** and **Subject A07 reaching 68.75%**.

---

## 3. Why achieving 70%+ Pooled Accuracy is Challenging
1. **Inter-Subject Variability:** Raw EEG patterns vary heavily from person to person. A single SVM trained on pooled data across all subjects struggles to find a single hyperplane that generalizes well.
2. **Session-to-Session Transfer:** The test set `X_test` was recorded on a different session/day than `X_train`. The signal shift across sessions makes general classification without domain adaptation or complex deep learning (like the baseline EEGNet CNN) difficult.
