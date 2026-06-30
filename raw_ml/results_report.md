# Raw ML Screen Detection — Results Report

**Training set**: 100 images (50 Real + 50 Screen)

**Held-out test set**: 25 images (remaining)

**Total features extracted**: 110

**Evaluation**: Stratified 5-Fold Cross-Validation + Held-out Test


## 🏆 Model Leaderboard

| Rank | Model | CV Accuracy | CV F1 | Test Accuracy | Test F1 | Test Precision | Test Recall | Time |
|------|-------|-------------|-------|---------------|---------|----------------|-------------|------|
| 1 | **ExtraTrees** 👑 | 0.8900 ± 0.0200 | 0.8862 ± 0.0225 | 0.9600 | 0.9333 | 1.0000 | 0.8750 | 16.5s |
| 2 | **VotingEnsemble** | 0.8500 ± 0.0548 | 0.8450 ± 0.0580 | 0.9600 | 0.9333 | 1.0000 | 0.8750 | 0.2s |
| 3 | **StackingEnsemble** | 0.8600 ± 0.0374 | 0.8532 ± 0.0431 | 0.9600 | 0.9333 | 1.0000 | 0.8750 | 0.9s |
| 4 | **RandomForest** | 0.8800 ± 0.0510 | 0.8752 ± 0.0557 | 0.9200 | 0.8571 | 1.0000 | 0.7500 | 23.0s |
| 5 | **SVM_Linear** | 0.8700 ± 0.0748 | 0.8662 ± 0.0782 | 0.9200 | 0.8571 | 1.0000 | 0.7500 | 0.0s |
| 6 | **SVM_RBF** | 0.8800 ± 0.0510 | 0.8805 ± 0.0498 | 0.8800 | 0.8421 | 0.7273 | 1.0000 | 0.2s |
| 7 | **BaggingSVM** | 0.8800 ± 0.0400 | 0.8800 ± 0.0400 | 0.8800 | 0.8421 | 0.7273 | 1.0000 | 1.8s |
| 8 | **LogisticRegression** | 0.8600 ± 0.0374 | 0.8606 ± 0.0348 | 0.8400 | 0.7143 | 0.8333 | 0.6250 | 1.0s |
| 9 | **HistGradientBoosting** | 0.8800 ± 0.0400 | 0.8765 ± 0.0416 | 0.7600 | 0.6667 | 0.6000 | 0.7500 | 78.6s |
| 10 | **KNN** | 0.9000 ± 0.0707 | 0.9019 ± 0.0701 | 0.8400 | 0.6667 | 1.0000 | 0.5000 | 0.1s |
| 11 | **AdaBoost** | 0.9000 ± 0.0316 | 0.8967 ± 0.0351 | 0.7600 | 0.6250 | 0.6250 | 0.6250 | 4.8s |
| 12 | **GradientBoosting** | 0.8700 ± 0.0400 | 0.8698 ± 0.0335 | 0.7200 | 0.5882 | 0.5556 | 0.6250 | 38.6s |

## ✅ Best Model: **ExtraTrees**

- **CV Accuracy**: 0.8900 ± 0.0200
- **CV F1 Score**: 0.8862 ± 0.0225
- **Test Accuracy**: 0.9600
- **Test F1 Score**: 0.9333
- **Test Precision**: 1.0000
- **Test Recall**: 0.8750
- **Best Params**: `{'max_depth': None, 'max_features': 'sqrt', 'min_samples_split': 2, 'n_estimators': 300}`

### Confusion Matrix (Test Set)

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 17 | 0 |
| **Actual Screen** | 1 | 7 |

## 📊 All Confusion Matrices (Test Set)

### ExtraTrees

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 17 | 0 |
| **Actual Screen** | 1 | 7 |

### VotingEnsemble

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 17 | 0 |
| **Actual Screen** | 1 | 7 |

### StackingEnsemble

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 17 | 0 |
| **Actual Screen** | 1 | 7 |

### RandomForest

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 17 | 0 |
| **Actual Screen** | 2 | 6 |

### SVM_Linear

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 17 | 0 |
| **Actual Screen** | 2 | 6 |

### SVM_RBF

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 14 | 3 |
| **Actual Screen** | 0 | 8 |

### BaggingSVM

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 14 | 3 |
| **Actual Screen** | 0 | 8 |

### LogisticRegression

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 16 | 1 |
| **Actual Screen** | 3 | 5 |

### HistGradientBoosting

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 13 | 4 |
| **Actual Screen** | 2 | 6 |

### KNN

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 17 | 0 |
| **Actual Screen** | 4 | 4 |

### AdaBoost

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 14 | 3 |
| **Actual Screen** | 3 | 5 |

### GradientBoosting

| | Predicted Real | Predicted Screen |
|---|---|---|
| **Actual Real** | 13 | 4 |
| **Actual Screen** | 3 | 5 |

## 🔬 Top 30 Most Important Features (Mutual Information)

| Rank | Feature | MI Score |
|------|---------|----------|
| 1 | `fft_band_0_energy` | 0.2048 |
| 2 | `fft_band_2_energy` | 0.2036 |
| 3 | `fft_band_3_energy` | 0.2008 |
| 4 | `color_sat_mean` | 0.1790 |
| 5 | `fft_band_1_energy` | 0.1788 |
| 6 | `fft_std` | 0.1746 |
| 7 | `canny_density_50_100` | 0.1538 |
| 8 | `fft_band_4_energy` | 0.1363 |
| 9 | `color_hue_mean` | 0.1345 |
| 10 | `glare_dark_ratio` | 0.1314 |
| 11 | `glcm_entropy` | 0.1273 |
| 12 | `lap_var_k5` | 0.1239 |
| 13 | `color_b_entropy` | 0.1205 |
| 14 | `color_cb_std` | 0.1189 |
| 15 | `color_val_std` | 0.1127 |
| 16 | `brenner_mean` | 0.1125 |
| 17 | `noise_local_var_mean` | 0.1100 |
| 18 | `hough_avg_length` | 0.1093 |
| 19 | `glcm_contrast` | 0.1084 |
| 20 | `lap_var_k7` | 0.1078 |
| 21 | `fft_mean` | 0.1066 |
| 22 | `hough_line_count` | 0.1002 |
| 23 | `sobel_mag_kurt` | 0.0990 |
| 24 | `noise_sigma` | 0.0990 |
| 25 | `lbp_bin_7` | 0.0974 |
| 26 | `fft_peak_count` | 0.0973 |
| 27 | `glare_dynamic_range` | 0.0952 |
| 28 | `lbp_bin_29` | 0.0897 |
| 29 | `fft_total_energy` | 0.0874 |
| 30 | `lbp_bin_0` | 0.0860 |

## 📋 Per-Model Best Hyperparameters

### ExtraTrees

```json
{
  "max_depth": null,
  "max_features": "sqrt",
  "min_samples_split": 2,
  "n_estimators": 300
}
```

### VotingEnsemble

```json
{
  "members": [
    "ExtraTrees",
    "RandomForest",
    "SVM_Linear"
  ]
}
```

### StackingEnsemble

```json
{
  "members": [
    "ExtraTrees",
    "RandomForest",
    "SVM_Linear"
  ],
  "meta": "LogisticRegression"
}
```

### RandomForest

```json
{
  "max_depth": null,
  "max_features": "sqrt",
  "min_samples_leaf": 1,
  "min_samples_split": 2,
  "n_estimators": 100
}
```

### SVM_Linear

```json
{
  "C": 0.01
}
```

### SVM_RBF

```json
{
  "C": 10.0,
  "gamma": 0.01
}
```

### BaggingSVM

```json
{
  "max_features": 0.8,
  "max_samples": 1.0,
  "n_estimators": 10
}
```

### LogisticRegression

```json
{
  "C": 10.0,
  "penalty": "l1",
  "solver": "liblinear"
}
```

### HistGradientBoosting

```json
{
  "l2_regularization": 0.0,
  "learning_rate": 0.05,
  "max_depth": null,
  "max_iter": 300,
  "min_samples_leaf": 10
}
```

### KNN

```json
{
  "metric": "manhattan",
  "n_neighbors": 3,
  "weights": "uniform"
}
```

### AdaBoost

```json
{
  "learning_rate": 0.1,
  "n_estimators": 300
}
```

### GradientBoosting

```json
{
  "learning_rate": 0.01,
  "max_depth": 5,
  "min_samples_split": 2,
  "n_estimators": 100,
  "subsample": 0.8
}
```
