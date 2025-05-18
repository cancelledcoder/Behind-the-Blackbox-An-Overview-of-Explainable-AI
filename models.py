import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ucimlrepo import fetch_ucirepo
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, roc_auc_score, roc_curve, auc
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, StackingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from xgboost import XGBClassifier
from lime.lime_tabular import LimeTabularExplainer
import shap
import os
import numpy as np

os.makedirs("results/plots", exist_ok=True)
os.makedirs("results/reports", exist_ok=True)

def load_and_preprocess_data():
    dataset = fetch_ucirepo(id=17)
    X = dataset.data.features
    y = dataset.data.targets.iloc[:, 0]
    le = LabelEncoder()
    y = le.fit_transform(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test

def tune_model(model, param_grid, X_train, y_train):
    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train, y_train)
    return grid_search.best_estimator_

def define_and_tune_models(X_train, y_train):
    tuned_models = {}

    rf_params = {
        'n_estimators': [100, 200],
        'max_depth': [None, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2]
    }
    tuned_models["Random Forest"] = tune_model(RandomForestClassifier(), rf_params, X_train, y_train)

    xgb_params = {
        'n_estimators': [100, 200],
        'max_depth': [3, 6],
        'learning_rate': [0.1, 0.01]
    }
    tuned_models["XGBoost"] = tune_model(XGBClassifier(eval_metric='logloss'), xgb_params, X_train, y_train)

    mlp_params = {
        'hidden_layer_sizes': [(100,), (50, 50)],
        'activation': ['relu', 'tanh'],
        'alpha': [0.0001, 0.001]
    }
    tuned_models["Neural Network"] = tune_model(MLPClassifier(max_iter=1000), mlp_params, X_train, y_train)

    svm_params = {
        'C': [0.1, 1, 10],
        'kernel': ['rbf', 'linear'],
        'gamma': ['scale', 'auto']
    }
    tuned_models["SVM"] = tune_model(SVC(probability=True), svm_params, X_train, y_train)

    tuned_models["Logistic Regression"] = LogisticRegression(max_iter=500)

    return tuned_models

def train_and_evaluate_model(model, X_train, X_test, y_train, y_test, model_name):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    roc_auc = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1]) if hasattr(model, "predict_proba") else None
    report = classification_report(y_test, preds, output_dict=True)
    print(f"{model_name} Accuracy: {acc:.4f}")
    if roc_auc:
        print(f"{model_name} ROC AUC: {roc_auc:.4f}")
    with open(f"results/reports/{model_name}_report.txt", "w") as f:
        f.write(classification_report(y_test, preds))
    plot_roc_curve(y_test, model.predict_proba(X_test)[:, 1], model_name) if roc_auc else None
    return acc, report

def plot_confusion_matrix(y_test, preds, model_name):
    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(8, 6))
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title(f"{model_name} Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    height, width = cm.shape
    tags = np.array([["TN", "FP"], ["FN", "TP"]])
    for i in range(height):
        for j in range(width):
            ax.text(j + 0.1, i + 0.7, tags[i, j], color="black", fontsize=12, fontweight='bold')
    plt.savefig(f"results/plots/{model_name}_confusion_matrix.png")
    plt.close()

def plot_roc_curve(y_test, y_scores, model_name):
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'{model_name} ROC Curve')
    plt.legend(loc="lower right")
    plt.savefig(f"results/plots/{model_name}_roc_curve.png")
    plt.close()

def plot_combined_roc_curve(models, X_test, y_test, filename="results/plots/combined_roc_curve.png"):
    plt.figure(figsize=(10, 8))
    for name, model in models.items():
        if hasattr(model, "predict_proba"):
            y_scores = model.predict_proba(X_test)[:, 1]
        elif hasattr(model, "decision_function"):
            y_scores = model.decision_function(X_test)
        else:
            continue
        fpr, tpr, _ = roc_curve(y_test, y_scores)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Combined ROC Curve')
    plt.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()

def plot_accuracy_comparison(results):
    plt.figure(figsize=(10, 6))
    hybrid_models = ["Voting Classifier", "Stacking Classifier", "Combined Ensemble"]
    data = pd.DataFrame({
        "Model": list(results.keys()),
        "Accuracy": list(results.values()),
        "Type": ["Hybrid" if model in hybrid_models else "Base" for model in results.keys()]
    })
    sns.barplot(data=data, x="Model", y="Accuracy", hue="Type", dodge=False, palette={"Hybrid": "red", "Base": "blue"})
    plt.ylabel("Accuracy")
    plt.title("Model Accuracy Comparison")
    plt.ylim(0.9, 1)
    plt.xticks(rotation=30)
    plt.tight_layout()
    plt.savefig("results/plots/accuracy_comparison.png")
    plt.close()

def define_hybrid_models(models):
    voting = VotingClassifier(estimators=[
        ('lr', models["Logistic Regression"]),
        ('rf', models["Random Forest"]),
        ('svm', models["SVM"]),
        ('mlp', models["Neural Network"]),
        ('xgb', models["XGBoost"])
    ], voting='hard' , weights=[1, 1, 3, 1, 1])

    stacking = StackingClassifier(estimators=[
        ('rf', models["Random Forest"]),
        ('svm', models["SVM"]),
        ('xgb', models["XGBoost"]),
        ('mlp', models["Neural Network"])
    ], final_estimator=LogisticRegression(),
        cv=5,
        passthrough=True)

    combined = VotingClassifier(estimators=[
        ('svm', models["SVM"]),
        ('lr', models["Logistic Regression"]),
        ('mlp', models["Neural Network"])
    ], voting='soft', weights=[3, 2, 1])

    return {
        "Voting Classifier": voting,
        "Stacking Classifier": stacking,
        "Combined Ensemble": combined
    }

def explain_with_shap(model, X_train, X_test, model_name):
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_test)
    shap.summary_plot(shap_values, X_test, show=False)
    plt.savefig(f"results/plots/{model_name}_shap_summary.png")
    plt.close()

def explain_with_lime(model, X_train, X_test, model_name):
    explainer = LimeTabularExplainer(X_train, mode="classification")
    exp = explainer.explain_instance(X_test[0], model.predict_proba)
    exp.save_to_file(f"results/plots/{model_name}_lime.html")

def main():
    X_train, X_test, y_train, y_test = load_and_preprocess_data()
    models = define_and_tune_models(X_train, y_train)
    results = {}

    for name, model in models.items():
        acc, _ = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results[name] = acc
        plot_confusion_matrix(y_test, model.predict(X_test), name)
        if name == "XGBoost" or name == "Random Forest":
            explain_with_shap(model, X_train, X_test, name)
        elif name == "SVM" or name == "Neural Network":
            explain_with_lime(model, X_train, X_test, name)
    
    print("\nEnsemble Model Performance:")
    hybrid_models = define_hybrid_models(models)
    for name, model in hybrid_models.items():
        acc, _ = train_and_evaluate_model(model, X_train, X_test, y_train, y_test, name)
        results[name] = acc
        plot_confusion_matrix(y_test, model.predict(X_test), name)

    all_models = {**models, **hybrid_models}
    plot_combined_roc_curve(all_models, X_test, y_test)
    
    plot_accuracy_comparison(results)
    print("\nFinal Accuracy Scores:")
    for k, v in results.items():
        print(f"{k}: {v:.4f}")

main()
