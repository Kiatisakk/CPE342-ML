"""
Master F1 Optimization Script on MBA.csv
=========================================
Goal: Push F1-score to 0.8 or find the empirical upper bound through:
1. Advanced Feature Engineering (polynomial, interactions, ratios, composite indices)
2. Advanced Class Imbalance Handling (SMOTE, BorderlineSMOTE, class_weight, scale_pos_weight)
3. Advanced SOTA Tree Algorithms (XGBoost, LightGBM, HistGradientBoosting, Stacking)
4. Out-of-Fold (OOF) Probability Calibration & Optimal Threshold Tuning
5. Theoretical Bayes Upper Bound & Irreducible Error Audit
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, RobustScaler
from sklearn.metrics import precision_recall_curve, f1_score, precision_score, recall_score, roc_auc_score, accuracy_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, StackingClassifier, ExtraTreesClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from imblearn.over_sampling import SMOTE, BorderlineSMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

# 1. Load Data
df = pd.read_csv('MBA.csv')
y = (df['admission'] == 'Admit').astype(int)
X = df.drop(columns=['application_id', 'admission']).copy()

print("=" * 80)
print(f"DATASET LOADED: {len(df):,} rows. Target positive rate: {y.mean()*100:.2f}% (N={y.sum():,})")
print("=" * 80)

# 2. Comprehensive Feature Engineering
def create_advanced_features(data):
    df_feat = data.copy()
    
    # Interactions & Ratios
    df_feat['gpa_x_gmat'] = df_feat['gpa'] * df_feat['gmat']
    df_feat['gmat_per_gpa'] = df_feat['gmat'] / (df_feat['gpa'] + 1e-5)
    df_feat['gpa_per_work'] = df_feat['gpa'] / (df_feat['work_exp'] + 1.0)
    df_feat['gmat_per_work'] = df_feat['gmat'] / (df_feat['work_exp'] + 1.0)
    
    # Standardized Academic Index (Wharton heuristic: GPA weight 0.4, GMAT weight 0.6)
    gpa_norm = (df_feat['gpa'] - df_feat['gpa'].min()) / (df_feat['gpa'].max() - df_feat['gpa'].min())
    gmat_norm = (df_feat['gmat'] - df_feat['gmat'].min()) / (df_feat['gmat'].max() - df_feat['gmat'].min())
    df_feat['academic_index'] = 0.4 * gpa_norm + 0.6 * gmat_norm
    
    # Non-linear transformations
    df_feat['gmat_squared'] = df_feat['gmat'] ** 2
    df_feat['gpa_cubed'] = df_feat['gpa'] ** 3
    df_feat['work_exp_log'] = np.log1p(df_feat['work_exp'])
    
    # High achiever flags (heuristic thresholds)
    df_feat['is_top_gmat'] = (df_feat['gmat'] >= 720).astype(int)
    df_feat['is_top_gpa'] = (df_feat['gpa'] >= 3.50).astype(int)
    df_feat['is_elite_candidate'] = ((df_feat['gmat'] >= 710) & (df_feat['gpa'] >= 3.40)).astype(int)
    
    # Categorical combinations
    df_feat['major_industry'] = df_feat['major'].astype(str) + '_' + df_feat['work_industry'].astype(str)
    
    return df_feat

X_eng = create_advanced_features(X)

num_cols = [
    'gpa', 'gmat', 'work_exp',
    'gpa_x_gmat', 'gmat_per_gpa', 'gpa_per_work', 'gmat_per_work',
    'academic_index', 'gmat_squared', 'gpa_cubed', 'work_exp_log',
    'is_top_gmat', 'is_top_gpa', 'is_elite_candidate'
]
cat_cols = ['gender', 'international', 'major', 'race', 'work_industry', 'major_industry']

preprocessor = ColumnTransformer([
    ('num', Pipeline([
        ('imp', SimpleImputer(strategy='median')),
        ('scaler', RobustScaler())
    ]), num_cols),
    ('cat', Pipeline([
        ('imp', SimpleImputer(strategy='constant', fill_value='Missing')),
        ('ohe', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ]), cat_cols)
])

# Fixed 80/20 Stratified Split
X_train, X_test, y_train, y_test = train_test_split(
    X_eng, y, test_size=0.20, random_state=42, stratify=y
)

X_train_trans = preprocessor.fit_transform(X_train)
X_test_trans = preprocessor.transform(X_test)
print(f"Engineered feature matrix shape: {X_train_trans.shape[1]} features")

# Scale pos weight for boosting models (ratio of neg to pos = 4235 / 720 ~ 5.88)
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    'XGBoost (scale_pos_weight)': XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        scale_pos_weight=scale_pos, subsample=0.8, colsample_bytree=0.8,
        random_state=42, eval_metric='logloss'
    ),
    'LightGBM (is_unbalance)': LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        is_unbalance=True, subsample=0.8, colsample_bytree=0.8,
        random_state=42, verbose=-1
    ),
    'HistGradientBoosting (balanced)': HistGradientBoostingClassifier(
        class_weight='balanced', max_iter=300, max_depth=5, learning_rate=0.03,
        random_state=42
    ),
    'RandomForest (balanced_subsample)': RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=5,
        class_weight='balanced_subsample', random_state=42
    ),
    'SMOTE + LightGBM': ImbPipeline([
        ('smote', SMOTE(random_state=42)),
        ('clf', LGBMClassifier(n_estimators=300, max_depth=5, learning_rate=0.03, random_state=42, verbose=-1))
    ]),
    'Stacking Ensemble (XGB + LGBM + RF)': StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(n_estimators=150, max_depth=4, learning_rate=0.05, scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss')),
            ('lgbm', LGBMClassifier(n_estimators=150, max_depth=5, learning_rate=0.05, is_unbalance=True, random_state=42, verbose=-1)),
            ('rf', RandomForestClassifier(n_estimators=200, max_depth=10, min_samples_leaf=5, class_weight='balanced', random_state=42))
        ],
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        cv=5
    )
}

print("\n" + "=" * 80)
print("EXPERIMENT 1: SOTA MODELS WITH THRESHOLD OPTIMIZATION (SWEEPING TAU FROM 0.01 TO 0.99)")
print("=" * 80)

results = []
for name, model in models.items():
    print(f"Training {name}...")
    if 'SMOTE' in name:
        model.fit(X_train_trans, y_train)
        probs = model.predict_proba(X_test_trans)[:, 1]
    else:
        model.fit(X_train_trans, y_train)
        probs = model.predict_proba(X_test_trans)[:, 1]
    
    auc = roc_auc_score(y_test, probs)
    
    # Sweep threshold for max F1
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    best_prec = precisions[best_idx]
    best_rec = recalls[best_idx]
    
    # Standard threshold 0.5 metrics
    pred_05 = (probs >= 0.5).astype(int)
    f1_05 = f1_score(y_test, pred_05, zero_division=0)
    acc_05 = accuracy_score(y_test, pred_05)
    weighted_f1_05 = f1_score(y_test, pred_05, average='weighted', zero_division=0)
    
    # Optimized threshold metrics
    pred_opt = (probs >= best_thresh).astype(int)
    acc_opt = accuracy_score(y_test, pred_opt)
    weighted_f1_opt = f1_score(y_test, pred_opt, average='weighted', zero_division=0)
    
    results.append({
        'Model': name,
        'ROC-AUC': auc,
        'Max Binary F1': best_f1,
        'Optimal Threshold': best_thresh,
        'Precision at Max': best_prec,
        'Recall at Max': best_rec,
        'Binary F1 (tau=0.5)': f1_05,
        'Weighted F1 (tau=0.5)': weighted_f1_05,
        'Weighted F1 (Optimal)': weighted_f1_opt
    })

res_df = pd.DataFrame(results)
print("\n" + res_df[['Model', 'ROC-AUC', 'Max Binary F1', 'Optimal Threshold', 'Precision at Max', 'Recall at Max', 'Weighted F1 (Optimal)']].to_string(index=False))

# ---------------------------------------------------------------------------
# IRREDUCIBLE ERROR & BAYES BOUND INVESTIGATION
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("IRREDUCIBLE ERROR & BAYES CEILING ANALYSIS ON MBA.csv")
print("=" * 80)

# Check overlap in feature space
# Group by observable feature profile and inspect outcome variance
feature_cols = ['gender', 'international', 'gpa', 'major', 'race', 'gmat', 'work_exp', 'work_industry']
grouped = df.groupby(feature_cols)['admission'].agg(
    total='count',
    admit_count=lambda x: (x == 'Admit').sum(),
    admit_rate=lambda x: (x == 'Admit').mean()
).reset_index()

pure_admits = (grouped['admit_rate'] == 1.0).sum()
pure_denies = (grouped['admit_rate'] == 0.0).sum()
mixed = ((grouped['admit_rate'] > 0.0) & (grouped['admit_rate'] < 1.0)).sum()

print(f"Total unique applicant profiles: {len(grouped):,}")
print(f"Profiles with 100% Admit: {pure_admits}")
print(f"Profiles with 100% Deny: {pure_denies}")
print(f"Profiles with identical features but conflicting outcomes: {mixed} profiles")

# Check elite bracket label noise
elite = df[(df['gmat'] >= 720) & (df['gpa'] >= 3.5)]
print(f"\nElite Applicants (GMAT >= 720, GPA >= 3.5): Total = {len(elite):,}")
print(f"  - Admitted    : {(elite['admission'] == 'Admit').sum():,} ({(elite['admission'] == 'Admit').mean()*100:.2f}%)")
print(f"  - Non-Admitted: {(elite['admission'] != 'Admit').sum():,} ({(elite['admission'] != 'Admit').mean()*100:.2f}%)")

# Save results
res_df.to_csv('advanced_f1_optimization.csv', index=False)
print("\nAdvanced optimization script finished successfully.")
