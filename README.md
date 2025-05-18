# Behind the Blackbox: An Overview of Explainable AI

This repository contains a comprehensive pipeline for breast cancer detection using the Wisconsin Diagnostic Breast Cancer dataset, with a focus on model explainability using SHAP and LIME. The project demonstrates how explainable AI (XAI) techniques can be integrated into machine learning workflows to provide transparency and trust in model predictions.

## Features

- **Data Loading & Preprocessing:** Automated fetching and preparation of the UCI dataset.
- **Model Training:** Logistic Regression, Random Forest, SVM, Neural Network (MLP), XGBoost.
- **Hyperparameter Tuning:** Grid search for optimal model parameters.
- **Hybrid Models:** Voting, Stacking, and Combined Ensemble classifiers.
- **Evaluation:** Accuracy, ROC AUC, confusion matrices, and combined ROC curve plots.
- **Explainability:** SHAP for tree-based models, LIME for SVM and neural networks.
- **Visualization:** All results saved in the `results/plots/` directory for easy review.

## Repository Structure

- `models.py` — Main codebase for data processing, model training, evaluation, and explainability.
- `results/plots/` — Generated plots (confusion matrices, ROC curves, SHAP/LIME outputs).
- `results/reports/` — Classification reports for each model.

## Getting Started

### Prerequisites

- Python 3.8+
- pip
- All dependencies listed at the top of `models.py`

### Running the Pipeline

```bash
python [models.py](http://_vscodecontentref_/0)
