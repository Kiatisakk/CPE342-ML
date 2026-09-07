"""
MBA Admission Classification Experiment: Decision Tree vs Random Forest vs Gradient Boosting
=============================================================================================
Course: CPE 342 Machine Learning
Dataset: MBA.csv (6,194 records, Wharton MBA Class of 2025 synthetic dataset)

This script implements the consensus design synthesized from a 4-agent panel:
- SubAgent 1 (Dataset Analyst)
- SubAgent 2 (Data Preparation Specialist)
- SubAgent 3 (Model Specialist)
- SubAgent 4 (Evaluation & Analysis Specialist)

Strict reproducibility: random_state=42 throughout.
Zero hyperparameter tuning: all models use standard scikit-learn defaults.
Hermetic pipeline: Preprocessing fitted strictly on training data fold.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
)

# Ensure safe standard output encoding on Windows consoles
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

# ---------------------------------------------------------------------------
# 1. Dataset Loading & Inspection
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MBA.csv')
FIGURES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

print("=" * 80)
print("PHASE 4: EXPERIMENT IMPLEMENTATION - MBA ADMISSION CLASSIFICATION")
print("=" * 80)
print(f"Loading dataset from: {DATA_PATH}")

df = pd.read_csv(DATA_PATH)
n_rows, n_cols = df.shape
print(f"Dataset Dimensions: {n_rows:,} rows, {n_cols} columns\n")

# Target class distribution analysis
raw_counts = df['admission'].value_counts(dropna=False)
raw_percs = df['admission'].value_counts(dropna=False, normalize=True) * 100
print("Raw Target Column ('admission') Distribution:")
for val, count in raw_counts.items():
    label = 'NaN (Denied/Rejected)' if pd.isna(val) else val
    perc = raw_percs[val]
    print(f"  - {label:<24}: {count:>5} ({perc:>5.2f}%)")

# ---------------------------------------------------------------------------
# 2. Target Formulation & Feature Preparation
# ---------------------------------------------------------------------------
# Primary Binary Classification formulation agreed by multi-agent consensus:
# Class 1 (Admit) = 900 (14.53%)
# Class 0 (Non-Admit: Denied + Waitlisted) = 5,294 (85.47%)
y = (df['admission'] == 'Admit').astype(int)

# Dropping application_id to prevent serial index leakage / memorization
X = df.drop(columns=['application_id', 'admission'])

num_cols = ['gpa', 'gmat', 'work_exp']
cat_cols = ['gender', 'international', 'major', 'race', 'work_industry']

print("\nFeature Set Specification:")
print(f"  - Numerical features ({len(num_cols)}): {num_cols}")
print(f"  - Categorical features ({len(cat_cols)}): {cat_cols}")
print(f"  - Excluded features (2): 'application_id' (leakage risk), 'admission' (target)")
print(f"  - Target distribution (Binary): Non-Admit (0) = {(y == 0).sum():,} ({(y == 0).mean()*100:.2f}%), "
      f"Admit (1) = {(y == 1).sum():,} ({(y == 1).mean()*100:.2f}%)")

# ---------------------------------------------------------------------------
# 3. Stratified Train / Test Split (80/20)
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.20,
    random_state=42,
    stratify=y
)

print("\nTrain/Test Partition:")
print(f"  - Training samples : {len(X_train):,} (Admit: {y_train.sum():,}, Non-Admit: {(y_train == 0).sum():,})")
print(f"  - Testing samples  : {len(X_test):,} (Admit: {y_test.sum():,}, Non-Admit: {(y_test == 0).sum():,})")

# ---------------------------------------------------------------------------
# 4. Hermetic Preprocessing Pipeline Design
# ---------------------------------------------------------------------------
# Numerical pipeline: Median imputation (preserves natural units, no scaling for tree models)
num_transformer = SimpleImputer(strategy='median')

# Categorical pipeline: Constant imputation for missingness (MNAR in race) + OneHotEncoding
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='Missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ],
    remainder='drop'
)

# ---------------------------------------------------------------------------
# 5. Model Instantiations (Default Parameters, random_state=42, No Tuning)
# ---------------------------------------------------------------------------
models = {
    'Decision Tree': DecisionTreeClassifier(random_state=42),
    'Random Forest': RandomForestClassifier(random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42)
}

pipelines = {
    name: Pipeline(steps=[
        ('preprocessor', clone(preprocessor)),
        ('classifier', model)
    ])
    for name, model in models.items()
}

# ---------------------------------------------------------------------------
# 6. Training & Evaluation Routine
# ---------------------------------------------------------------------------
results = []
trained_pipelines = {}
predictions = {}
probabilities = {}

print("\n" + "=" * 80)
print("TRAINING AND EVALUATING BASELINE MODELS")
print("=" * 80)

for name, pipe in pipelines.items():
    print(f"Fitting {name}...")
    pipe.fit(X_train, y_train)
    trained_pipelines[name] = pipe

    # Predictions
    y_train_pred = pipe.predict(X_train)
    y_test_pred = pipe.predict(X_test)
    y_train_proba = pipe.predict_proba(X_train)[:, 1]
    y_test_proba = pipe.predict_proba(X_test)[:, 1]

    predictions[name] = (y_train_pred, y_test_pred)
    probabilities[name] = (y_train_proba, y_test_proba)

    # Metrics computation
    train_acc = accuracy_score(y_train, y_train_pred)
    test_acc = accuracy_score(y_test, y_test_pred)
    acc_gap = train_acc - test_acc

    test_prec_binary = precision_score(y_test, y_test_pred, average='binary', zero_division=0)
    test_rec_binary = recall_score(y_test, y_test_pred, average='binary', zero_division=0)
    test_f1_binary = f1_score(y_test, y_test_pred, average='binary', zero_division=0)

    train_f1_binary = f1_score(y_train, y_train_pred, average='binary', zero_division=0)
    f1_gap = train_f1_binary - test_f1_binary

    test_f1_macro = f1_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_prec_macro = precision_score(y_test, y_test_pred, average='macro', zero_division=0)
    test_rec_macro = recall_score(y_test, y_test_pred, average='macro', zero_division=0)

    train_auc = roc_auc_score(y_train, y_train_proba)
    test_auc = roc_auc_score(y_test, y_test_proba)
    auc_gap = train_auc - test_auc

    results.append({
        'Model': name,
        'Train Acc': train_acc,
        'Test Acc': test_acc,
        'Acc Gap (Delta)': acc_gap,
        'Train F1': train_f1_binary,
        'Test F1': test_f1_binary,
        'F1 Gap (Delta)': f1_gap,
        'Test Precision': test_prec_binary,
        'Test Recall': test_rec_binary,
        'Test Macro F1': test_f1_macro,
        'Train ROC-AUC': train_auc,
        'Test ROC-AUC': test_auc,
        'AUC Gap (Delta)': auc_gap,
    })

# Summary Table
results_df = pd.DataFrame(results)

print("\n" + "=" * 80)
print("PHASE 6: EMPIRICAL RESULTS SUMMARY TABLE")
print("=" * 80)

# Format for display matching user requirements
display_df = results_df[[
    'Model', 'Test Acc', 'Test Precision', 'Test Recall', 'Test F1', 'Test Macro F1', 'Test ROC-AUC'
]].copy()
display_df.columns = ['Model', 'Accuracy', 'Precision', 'Recall', 'F1-score', 'Macro F1', 'ROC-AUC']
print(display_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

print("\n" + "-" * 80)
print("OVERFITTING & GENERALIZATION GAP AUDIT (Train vs Test)")
print("-" * 80)
overfit_df = results_df[[
    'Model', 'Train Acc', 'Test Acc', 'Acc Gap (Delta)', 'Train F1', 'Test F1', 'F1 Gap (Delta)', 'Train ROC-AUC', 'Test ROC-AUC'
]].copy()
print(overfit_df.to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# ---------------------------------------------------------------------------
# 7. Confusion Matrices & Classification Reports
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("CONFUSION MATRICES & CLASSIFICATION REPORTS (TEST SET, N=1,239)")
print("=" * 80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))
fig.suptitle('Confusion Matrices on Test Set (MBA Admission Binary Classification)', fontsize=14, fontweight='bold')

for idx, (name, (y_train_pred, y_test_pred)) in enumerate(predictions.items()):
    cm = confusion_matrix(y_test, y_test_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"\n--- {name.upper()} ---")
    print(f"Confusion Matrix (TN={tn}, FP={fp}, FN={fn}, TP={tp}):")
    print(cm)
    print("\nClassification Report:")
    print(classification_report(y_test, y_test_pred, target_names=['Non-Admit', 'Admit'], digits=4))

    # Heatmap
    sns.heatmap(
        cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], cbar=False,
        xticklabels=['Non-Admit (0)', 'Admit (1)'],
        yticklabels=['Non-Admit (0)', 'Admit (1)']
    )
    axes[idx].set_title(f"{name}\nAcc: {accuracy_score(y_test, y_test_pred):.4f} | F1: {f1_score(y_test, y_test_pred):.4f}", fontsize=11)
    axes[idx].set_xlabel('Predicted Label')
    axes[idx].set_ylabel('True Label')

plt.tight_layout()
cm_path = os.path.join(FIGURES_DIR, 'confusion_matrices.png')
plt.savefig(cm_path, dpi=300)
plt.close()
print(f"Saved confusion matrices figure to: {cm_path}")

# ---------------------------------------------------------------------------
# 8. ROC Curves
# ---------------------------------------------------------------------------
plt.figure(figsize=(8, 6))
for name, (y_train_proba, y_test_proba) in probabilities.items():
    fpr, tpr, _ = roc_curve(y_test, y_test_proba)
    auc = roc_auc_score(y_test, y_test_proba)
    plt.plot(fpr, tpr, lw=2, label=f"{name} (AUC = {auc:.4f})")

plt.plot([0, 1], [0, 1], color='gray', lw=1.5, linestyle='--', label='Random Guess (AUC = 0.5000)')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=12)
plt.ylabel('True Positive Rate (Sensitivity / Recall)', fontsize=12)
plt.title('Receiver Operating Characteristic (ROC) Curves on Test Set', fontsize=13, fontweight='bold')
plt.legend(loc='lower right', fontsize=11)
plt.grid(alpha=0.3)
plt.tight_layout()
roc_path = os.path.join(FIGURES_DIR, 'roc_curves.png')
plt.savefig(roc_path, dpi=300)
plt.close()
print(f"Saved ROC curves figure to: {roc_path}")

# ---------------------------------------------------------------------------
# 9. Feature Importance Extraction
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("FEATURE IMPORTANCE AUDIT (Mean Decrease in Impurity / Gini)")
print("=" * 80)

feature_names = trained_pipelines['Decision Tree'].named_steps['preprocessor'].get_feature_names_out()
# Clean prefix names for clarity
cleaned_names = [f.replace('num__', '').replace('cat__', '') for f in feature_names]

feat_df = pd.DataFrame({'Feature': cleaned_names})
for name, pipe in trained_pipelines.items():
    feat_df[name] = pipe.named_steps['classifier'].feature_importances_

feat_df['Mean Importance'] = feat_df[['Decision Tree', 'Random Forest', 'Gradient Boosting']].mean(axis=1)
feat_df = feat_df.sort_values(by='Mean Importance', ascending=False).reset_index(drop=True)

print(feat_df.head(10).to_string(index=False, float_format=lambda x: f"{x:.4f}"))

# Plot Top 10 Features
top_10 = feat_df.head(10).copy()
top_10_melted = top_10.melt(
    id_vars=['Feature'], 
    value_vars=['Decision Tree', 'Random Forest', 'Gradient Boosting'],
    var_name='Model', value_name='Importance'
)

plt.figure(figsize=(12, 6))
sns.barplot(data=top_10_melted, y='Feature', x='Importance', hue='Model', palette='tab10')
plt.title('Top 10 Most Important Features Across Models (Gini Importance)', fontsize=13, fontweight='bold')
plt.xlabel('Feature Importance (Mean Decrease Impurity)', fontsize=11)
plt.ylabel('Feature', fontsize=11)
plt.legend(title='Model', loc='lower right')
plt.tight_layout()
feat_path = os.path.join(FIGURES_DIR, 'feature_importances.png')
plt.savefig(feat_path, dpi=300)
plt.close()
print(f"Saved feature importances figure to: {feat_path}")

# ---------------------------------------------------------------------------
# 10. Overfitting Comparison Plot
# ---------------------------------------------------------------------------
overfit_plot_df = pd.melt(
    results_df, 
    id_vars=['Model'], 
    value_vars=['Train Acc', 'Test Acc'],
    var_name='Split', value_name='Accuracy'
)

plt.figure(figsize=(9, 5))
sns.barplot(data=overfit_plot_df, x='Model', y='Accuracy', hue='Split', palette='Set2')
plt.title('Train vs Test Accuracy (Overfitting Diagnostic)', fontsize=13, fontweight='bold')
plt.ylabel('Accuracy', fontsize=11)
plt.ylim([0.7, 1.05])
for i in range(len(results_df)):
    gap = results_df.loc[i, 'Acc Gap (Delta)']
    plt.text(i, 1.01, f"Gap: -{gap*100:.1f}%", ha='center', fontweight='bold', color='darkred')
plt.tight_layout()
overfit_path = os.path.join(FIGURES_DIR, 'overfitting_gap.png')
plt.savefig(overfit_path, dpi=300)
plt.close()
print(f"Saved overfitting diagnostic figure to: {overfit_path}")

# Save numerical results to CSV for reporting
results_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'experiment_results.csv'), index=False)
feat_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'feature_importances.csv'), index=False)
print("\nAll experiment artifacts successfully generated.")
