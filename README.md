# EMG Physical Action Analysis & Classification

This project analyzes 8-channel surface electromyography (sEMG) signals to automatically recognize human physical actions. We classify movements into broad categories (Normal vs. Aggressive) and evaluate multiclass performance across 20 distinct activities.

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue?logo=github)](https://github.com/WebProject05/EMG_PhysicalSignal_Analysis)

---

## What Does This Project Do?

1. **Explores Muscle Signals**: Reads 8-channel muscle recordings from 4 participants performing 20 actions (10 normal daily actions like walking and waving, and 10 aggressive actions like punching and kicking).
2. **Extracts Features**: Chops each action trial into 4 time segments and computes the Mean and Standard Deviation for each sensor channel, creating a compact 64-number profile per trial.
3. **Classifies Actions**: Uses $k$-Nearest Neighbors ($k$-NN) inside a data-leakage-free standardization pipeline with 5-fold cross-validation.
4. **Studies Performance Scaling**: Measures how accuracy changes as the number of actions increases from 2 up to 20 classes.

---

## Project Structure & Notebooks

The workflow is split into 5 step-by-step Jupyter notebooks:

- **`01_data_exploration.ipynb`**: Loads raw sEMG signals, checks sensor channels, and compares waveforms (e.g., Bowing vs. Punching).
- **`02_feature_extraction.ipynb`**: Slices continuous recordings into 15 trials and extracts the 64 statistical features.
- **`03_label_generation.ipynb`**: Assigns multiclass targets (1 to 20) and binary labels (Normal = 0, Aggressive = 1) and verifies class balance.
- **`04_binary_classification.ipynb`**: Evaluates $k$-NN on Normal vs. Aggressive classification and pairwise tasks.
- **`05_multiclass_analysis.ipynb`**: Runs Monte Carlo simulations from $M = 2$ to $19$ classes and benchmarks the full 20-class dataset.

---

## Key Results

- **Normal vs. Aggressive Detection**:
  - **Accuracy**: **84.08%** (Optimal $k=1$ achieves **85.50%**)
  - **AUC-ROC**: **0.91** (Strong separation between routine and violent movements)
- **Pairwise Classification (Bowing vs. Elbowing)**:
  - **Accuracy**: **84.17%**
- **Multiclass Complexity Study**:
  - Accuracy scales gracefully from **83.83%** (for 2 classes) down to **35.73%** (for 19 classes).
  - Full **20-class classification benchmark**: **35.08% accuracy** (nearly $7\times$ higher than the $5.00\%$ random guessing level).

---

## Extracted Graphs & Report

- **Report (PDF)**: Available in [`latex_report/main.pdf`](./latex_report/main.pdf) (built using LaTeX, 6 pages).
- **All Figures**: Stored in [`extracted_graphs/`](./extracted_graphs/) and [`latex_report/images/`](./latex_report/images/).

---

## Repository Link
GitHub: [https://github.com/WebProject05/EMG_PhysicalSignal_Analysis](https://github.com/WebProject05/EMG_PhysicalSignal_Analysis)
