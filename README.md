# EMG Physical Action Analysis & Classification

This project analyzes 8-channel surface electromyography (sEMG) signals to classify physical actions performed by human subjects. The analysis progresses from raw biosignal data exploration and feature extraction to binary classification (Normal vs. Aggressive behaviors) and multiclass performance degradation studies across varying task complexities.

---

## 1. Project Overview & Workflow

The analysis pipeline is organized across 5 sequential Jupyter notebooks:

```
01_data_exploration.ipynb   -->  02_feature_extraction.ipynb  -->  03_label_generation.ipynb
                                                                          |
                                                                          v
05_multiclass_analysis.ipynb <-- 04_binary_classification.ipynb <---------+
```

1. **`01_data_exploration.ipynb`**: Explores the raw 8-channel sEMG dataset, validates channel integrity, verifies sample lengths across recordings, and contrasts normal vs. aggressive action waveforms.
2. **`02_feature_extraction.ipynb`**: Segments continuous action recordings into 15 individual trials, partitions each trial into 4 temporal windows ($N_{seg} = 4$), and extracts statistical features (Mean and Standard Deviation) across all 8 channels (yielding a 64-dimensional feature vector per trial).
3. **`03_label_generation.ipynb`**: Constructs target arrays for multiclass classification (classes 1–20) and binary classification (0: Normal, 1: Aggressive), verifies balanced class distributions, and validates dataset integrity.
4. **`04_binary_classification.ipynb`**: Evaluates $k$-Nearest Neighbors ($k$-NN) inside a data-leakage-free `StandardScaler` pipeline using 5-Fold Stratified Cross-Validation on two binary scenarios (broad Normal vs. Aggressive, and pairwise Bowing vs. Elbowing).
5. **`05_multiclass_analysis.ipynb`**: Conducts a Monte Carlo degradation study across subset sizes from $M = 2$ to $M = 19$ classes (10 random combinations per $M$) and benchmarks full 20-class classification.

---

## 2. Dataset Description

The dataset captures surface EMG recordings across:
- **4 Subjects**: `sub1`, `sub2`, `sub3`, `sub4`
- **8 EMG Sensor Channels**: Placed on skeletal muscle sites across the upper and lower limbs (Right Biceps, Right Triceps, Left Biceps, Left Triceps, Right Thigh, Right Hamstring, Left Thigh, Left Hamstring).
- **20 Physical Action Classes**:
  - **10 Normal Actions**: Bowing, Clapping, Handshaking, Hugging, Jumping, Running, Seating, Standing, Walking, Waving.
  - **10 Aggressive Actions**: Elbowing, Frontkicking, Hammering, Headering, Kneeing, Pulling, Punching, Pushing, Sidekicking, Slapping.
- **Trial Structure**: Each action recording contains 15 consecutive trials per subject.
- **Total Dataset Size**: $4 \text{ subjects} \times 20 \text{ actions} \times 15 \text{ trials} = 1,200 \text{ total action samples}$.

---

## 3. Signal Processing & Feature Engineering

Because raw sEMG signals are stochastic and high-dimensional, statistical time-domain features are extracted to represent muscle activation patterns:

1. **Trial Slicing**: Each continuous time-series recording is divided into 15 equal-length trial segments.
2. **Sub-Segmentation ($N_{seg} = 4$)**: Each trial is divided into 4 non-overlapping temporal segments to capture the time-varying progression of the action.
3. **Feature Computation**:
   - **Mean ($\mu$)**: $\mu = \frac{1}{L} \sum_{t=1}^{L} x(t)$
   - **Standard Deviation ($\sigma$)**: $\sigma = \sqrt{\frac{1}{L} \sum_{t=1}^{L} (x(t) - \mu)^2}$
4. **Feature Vector Dimensionality**:
   $$\text{Total Features} = 8 \text{ channels} \times 4 \text{ segments} \times 2 \text{ features} = 64 \text{ features per trial}$$
   Resulting dataset matrix shape: $X \in \mathbb{R}^{1200 \times 64}$.

---

## 4. Modeling & Experimental Results

All classification models use $k$-Nearest Neighbors ($k$-NN) encapsulated inside a scikit-learn `Pipeline` with `StandardScaler` to ensure standardization statistics are calculated strictly inside training folds, preventing data leakage.

### A. Binary Classification (`04_binary_classification.ipynb`)

#### Scenario 1: Normal vs. Aggressive (Full Dataset: 1,200 trials, 600 Normal / 600 Aggressive)
- **Model**: $k$-NN ($k=5$, Euclidean distance, `StandardScaler`)
- **Evaluation**: 5-Fold Stratified Cross-Validation
- **Metrics**:
  - **Accuracy**: $84.08\% \pm 2.84\%$
  - **Precision**: $86.21\% \pm 3.10\%$
  - **Recall**: $81.17\% \pm 3.14\%$
  - **F1-Score**: $83.60\% \pm 2.96\%$
  - **AUC-ROC**: $0.91$
- **Hyperparameter Tuning ($k \in [1, 25]$)**:
  - Optimal neighborhood parameter is $k = 1$, achieving an accuracy of **$85.50\% \pm 2.92\%$**. Performance gradually drops as $k$ increases, reaching $\approx 77\%$ at $k = 25$.

#### Scenario 2: Pairwise Two-Class Classification (120 trials: Bowing vs. Elbowing)
- Sample size: $4 \text{ subjects} \times 2 \text{ actions} \times 15 \text{ trials} = 120$ trials.
- **5-Fold Cross-Validation Accuracy**: **$84.17\% \pm 8.90\%$**.

---

### B. Multiclass Analysis & Degradation Study (`05_multiclass_analysis.ipynb`)

To analyze how classification performance scales with task complexity, classification was evaluated across varying numbers of classes $M \in \{2, 3, \dots, 19\}$ using a Monte Carlo protocol:
- For each $M$, 10 random combinations of $M$ classes were drawn from the 20 available actions (`RANDOM_STATE = 42`).
- Each combination was evaluated using 5-Fold Stratified Cross-Validation with $k$-NN ($k=5$).

#### Accuracy Degradation Summary Across $M$ Classes:

| Number of Classes ($M$) | Mean Accuracy (%) | Std Dev (%) | Min Accuracy (%) | Max Accuracy (%) | Chance Level ($100/M$) |
| :---: | :---: | :---: | :---: | :---: | :---: |
| **2** | 83.83% | 6.20% | 76.67% | 97.50% | 50.00% |
| **3** | 71.33% | 7.77% | 53.89% | 81.11% | 33.33% |
| **4** | 60.42% | 7.29% | 48.33% | 74.17% | 25.00% |
| **5** | 59.40% | 3.71% | 55.00% | 67.67% | 20.00% |
| **6** | 56.81% | 4.69% | 50.00% | 63.33% | 16.67% |
| **7** | 54.79% | 3.67% | 48.81% | 58.81% | 14.29% |
| **8** | 49.67% | 2.81% | 44.17% | 53.96% | 12.50% |
| **9** | 49.06% | 2.96% | 44.63% | 54.81% | 11.11% |
| **10** | 48.18% | 2.82% | 42.17% | 51.17% | 10.00% |
| **12** | 44.15% | 3.47% | 37.50% | 50.28% | 8.33% |
| **15** | 39.52% | 1.50% | 36.67% | 41.89% | 6.67% |
| **18** | 36.16% | 1.15% | 33.98% | 38.33% | 5.56% |
| **19** | 35.73% | 1.07% | 33.68% | 36.84% | 5.26% |

#### Full 20-Class Benchmark (All 1,200 trials)
- **5-Fold Cross-Validation Accuracy**: **$35.08\% \pm 2.55\%$**
- **Theoretical Chance Level ($1/20$)**: $5.00\%$
- **Observation**: The 64-dimensional statistical feature representation maintains performance that is **$7\times$ higher than chance**, despite high inter-subject anatomical variability and similar muscular recruitment patterns across related movements.

---

## 5. Extracted Visualizations

All graphs generated across the notebooks have been extracted and saved into the [`extracted_graphs/`](./extracted_graphs/) directory:

| Filename | Source Notebook | Description |
| :--- | :--- | :--- |
| [`01_normal_action_bowing_raw_emg.png`](./extracted_graphs/01_normal_action_bowing_raw_emg.png) | `01_data_exploration.ipynb` | 8-channel raw sEMG voltage signals recorded during the "Bowing" normal action. |
| [`02_aggressive_action_punching_raw_emg.png`](./extracted_graphs/02_aggressive_action_punching_raw_emg.png) | `01_data_exploration.ipynb` | 8-channel raw sEMG voltage signals recorded during the "Punching" aggressive action showing rapid high-amplitude bursts. |
| [`03_feature_profiles_bowing_vs_punching.png`](./extracted_graphs/03_feature_profiles_bowing_vs_punching.png) | `02_feature_extraction.ipynb` | Comparison of 64-dimensional mean feature profiles between Bowing and Punching across all 8 channels and 4 time segments. |
| [`04_dataset_class_distributions.png`](./extracted_graphs/04_dataset_class_distributions.png) | `03_label_generation.ipynb` | Multiclass distribution (balanced 60 trials per action) and binary distribution (balanced 600 Normal vs. 600 Aggressive). |
| [`05_binary_confusion_matrix_and_roc.png`](./extracted_graphs/05_binary_confusion_matrix_and_roc.png) | `04_binary_classification.ipynb` | Confusion matrix and ROC curve (AUC = 0.91) for Normal vs. Aggressive classification. |
| [`06_binary_k_tuning_curve.png`](./extracted_graphs/06_binary_k_tuning_curve.png) | `04_binary_classification.ipynb` | Cross-validation accuracy vs. number of neighbors $k$ ($k=1$ to $25$), showing optimal performance at $k=1$ (85.50%). |
| [`07_pairwise_bowing_vs_elbowing_confusion_matrix.png`](./extracted_graphs/07_pairwise_bowing_vs_elbowing_confusion_matrix.png) | `04_binary_classification.ipynb` | Confusion matrix for pairwise action discrimination between Bowing and Elbowing (84.17% accuracy). |
| [`08_monte_carlo_multiclass_heatmap.png`](./extracted_graphs/08_monte_carlo_multiclass_heatmap.png) | `05_multiclass_analysis.ipynb` | Color-coded heatmap showing cross-validation accuracy across 10 Monte Carlo runs for each class count $M \in [2, 19]$. |
| [`09_multiclass_performance_degradation_curve.png`](./extracted_graphs/09_multiclass_performance_degradation_curve.png) | `05_multiclass_analysis.ipynb` | Mean CV accuracy degradation curve with $\pm 1$ standard deviation error bands compared against random chance level. |
| [`10_full_20_class_confusion_matrix.png`](./extracted_graphs/10_full_20_class_confusion_matrix.png) | `05_multiclass_analysis.ipynb` | Comprehensive 20-by-20 confusion matrix for the full dataset benchmark (35.08% overall accuracy vs. 5% chance). |

---

## 6. Key Scientific Takeaways

1. **High Separability of High-Level Behaviors**: Distinguishing broad aggressive physical actions from daily routine movements is highly viable ($\approx 84.1\% - 85.5\%$ accuracy), because aggressive maneuvers involve synchronized, explosive motor unit recruitment.
2. **Impact of Neighborhood Size ($k$)**: Smaller $k$ values ($k=1, 3$) perform best in the high-dimensional feature space ($\mathbb{R}^{64}$), capturing localized sub-clusters formed by subject-specific motion signatures.
3. **Graceful Performance Degradation**: As classification complexity grows from 2 classes to 19 classes, accuracy scales down smoothly from $83.83\%$ to $35.73\%$, consistently maintaining a strong margin over theoretical chance ($100/M\%$).
4. **Leakage Prevention**: Standardizing feature vectors strictly within cross-validation folds provides an unbiased, generalizable evaluation of action recognition performance across multiple human subjects.

