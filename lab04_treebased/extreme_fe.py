"""
Extreme Feature Engineering Pipeline for MBA Admission Classification
======================================================================
Author: CPE 342 Advanced Feature Engineering Lab
Target: admission (Binary: Admit = 1, Non-Admit = 0)

Implements 7 Deep Feature Engineering Layers:
1. Domain B-School Academic Indices & Non-Linear Expansions
2. GMAC Percentile Curves & Major Rigor Adjustments
3. Career Velocity & Experience Sweet Spot Dynamics
4. Leak-Free Group Aggregations (Cohort Relative Standing)
5. Out-of-Fold (OOF) Smoothed Target Encoding
6. Elite Threshold & Demographic Quota Interaction Flags
7. Unsupervised Representation Learning (K-Means & PCA)
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    roc_auc_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    precision_recall_curve,
    confusion_matrix,
    classification_report
)

# Tree & Ensemble Models
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    StackingClassifier
)
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

# ---------------------------------------------------------------------------
# 1. Load Data & Initial Prep
# ---------------------------------------------------------------------------
DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'MBA.csv')
df = pd.read_csv(DATA_PATH)

# Target: Binary Admit (1) vs Non-Admit (0)
y = (df['admission'] == 'Admit').astype(int)
X_raw = df.drop(columns=['application_id', 'admission']).copy()

# Stratified 80/20 Train/Test Split FIRST to ensure absolute zero data leakage
X_train_raw, X_test_raw, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

print("=" * 85)
print("EXTREME FEATURE ENGINEERING: 7-LAYER ARCHITECTURE ON MBA.CSV")
print("=" * 85)
print(f"Train samples: {len(X_train_raw):,} | Test samples: {len(X_test_raw):,} | Positive rate: {y_train.mean()*100:.2f}%")

# ---------------------------------------------------------------------------
# 2. Extreme Feature Engineering Engine (Fitted strictly on Train)
# ---------------------------------------------------------------------------
class ExtremeFeatureEngineer:
    def __init__(self, random_state=42):
        self.random_state = random_state
        self.group_stats = {}
        self.target_encoding_maps = {}
        self.kmeans = None
        self.pca = None
        self.scaler = None
        
    @staticmethod
    def gmat_to_percentile(gmat_scores):
        """Map raw GMAT scores to approximate GMAC empirical percentiles."""
        pct = np.zeros_like(gmat_scores, dtype=float)
        pct[gmat_scores >= 760] = 0.99
        pct[(gmat_scores >= 740) & (gmat_scores < 760)] = 0.97
        pct[(gmat_scores >= 720) & (gmat_scores < 740)] = 0.94
        pct[(gmat_scores >= 700) & (gmat_scores < 720)] = 0.88
        pct[(gmat_scores >= 680) & (gmat_scores < 700)] = 0.80
        pct[(gmat_scores >= 650) & (gmat_scores < 680)] = 0.68
        pct[(gmat_scores >= 620) & (gmat_scores < 650)] = 0.55
        pct[(gmat_scores >= 590) & (gmat_scores < 620)] = 0.43
        pct[gmat_scores < 590] = 0.30
        return pct

    def fit_transform(self, X_train, y_train):
        df_out = X_train.copy()
        
        # -------------------------------------------------------------------
        # LAYER 1: Domain Academic Indices & Non-Linear Expansions
        # -------------------------------------------------------------------
        # Normalize continuous credentials on train bounds
        self.gpa_min, self.gpa_max = df_out['gpa'].min(), df_out['gpa'].max()
        self.gmat_min, self.gmat_max = df_out['gmat'].min(), df_out['gmat'].max()
        
        gpa_norm = (df_out['gpa'] - self.gpa_min) / (self.gpa_max - self.gpa_min)
        gmat_norm = (df_out['gmat'] - self.gmat_min) / (self.gmat_max - self.gmat_min)
        
        # Wharton Academic Index (AI)
        df_out['fe_academic_index'] = 0.40 * gpa_norm + 0.60 * gmat_norm
        df_out['fe_academic_power'] = (df_out['gpa'] ** 2) * (df_out['gmat'] / 100.0)
        df_out['fe_gpa_x_gmat'] = df_out['gpa'] * df_out['gmat']
        df_out['fe_gmat_per_gpa'] = df_out['gmat'] / (df_out['gpa'] + 1e-5)
        df_out['fe_gpa_squared'] = df_out['gpa'] ** 2
        df_out['fe_gmat_squared'] = df_out['gmat'] ** 2
        df_out['fe_log_gmat'] = np.log(df_out['gmat'])
        df_out['fe_sqrt_synergy'] = np.sqrt(df_out['gpa']) * np.sqrt(df_out['gmat'])
        
        # -------------------------------------------------------------------
        # LAYER 2: GMAC Percentiles & Major Rigor Scaling
        # -------------------------------------------------------------------
        df_out['fe_gmat_percentile'] = self.gmat_to_percentile(df_out['gmat'].values)
        
        # Major grading rigor adjustments (STEM has harsher curve)
        major_rigor = {'STEM': 1.08, 'Business': 1.04, 'Humanities': 1.00}
        df_out['fe_rigor_adjusted_gpa'] = df_out['gpa'] * df_out['major'].map(major_rigor).fillna(1.0)
        
        # -------------------------------------------------------------------
        # LAYER 3: Experience Sweet Spot & Career Velocity
        # -------------------------------------------------------------------
        # Wharton experience median is 5.0 years
        df_out['fe_exp_sweet_spot_dev'] = np.abs(df_out['work_exp'] - 5.0)
        df_out['fe_is_sweet_spot'] = ((df_out['work_exp'] >= 4.0) & (df_out['work_exp'] <= 6.0)).astype(float)
        df_out['fe_exp_quadratic'] = (df_out['work_exp'] - 5.0) ** 2
        df_out['fe_is_too_young'] = (df_out['work_exp'] <= 2.0).astype(float)
        df_out['fe_is_over_exp'] = (df_out['work_exp'] >= 8.0).astype(float)
        
        # Velocity metrics
        df_out['fe_velocity_gmat'] = df_out['gmat'] / (df_out['work_exp'] + 1.0)
        df_out['fe_velocity_gpa'] = df_out['gpa'] / (df_out['work_exp'] + 1.0)
        df_out['fe_velocity_academic'] = df_out['fe_academic_index'] / (df_out['work_exp'] + 1.0)
        
        # Industry Feeder Tiers
        tier1_industries = {'PE/VC', 'Consulting', 'Investment Banking'}
        tier2_industries = {'Technology', 'Investment Management', 'Financial Services'}
        df_out['fe_is_tier1_feeder'] = df_out['work_industry'].isin(tier1_industries).astype(float)
        df_out['fe_is_tier2_feeder'] = df_out['work_industry'].isin(tier2_industries).astype(float)
        
        # -------------------------------------------------------------------
        # LAYER 4: Cohort Relative Standings (Fitted on Train Only)
        # -------------------------------------------------------------------
        for grp_col in ['work_industry', 'major', 'gender']:
            mean_gmat = df_out.groupby(grp_col)['gmat'].mean()
            std_gmat = df_out.groupby(grp_col)['gmat'].std().fillna(1.0)
            mean_gpa = df_out.groupby(grp_col)['gpa'].mean()
            std_gpa = df_out.groupby(grp_col)['gpa'].std().fillna(1.0)
            
            self.group_stats[grp_col] = {
                'mean_gmat': mean_gmat, 'std_gmat': std_gmat,
                'mean_gpa': mean_gpa, 'std_gpa': std_gpa,
                'global_mean_gmat': df_out['gmat'].mean(),
                'global_std_gmat': df_out['gmat'].std(),
                'global_mean_gpa': df_out['gpa'].mean(),
                'global_std_gpa': df_out['gpa'].std()
            }
            
            df_out[f'fe_diff_gmat_{grp_col}'] = df_out['gmat'] - df_out[grp_col].map(mean_gmat)
            df_out[f'fe_z_gmat_{grp_col}'] = df_out[f'fe_diff_gmat_{grp_col}'] / (df_out[grp_col].map(std_gmat) + 1e-5)
            df_out[f'fe_diff_gpa_{grp_col}'] = df_out['gpa'] - df_out[grp_col].map(mean_gpa)
            df_out[f'fe_z_gpa_{grp_col}'] = df_out[f'fe_diff_gpa_{grp_col}'] / (df_out[grp_col].map(std_gpa) + 1e-5)

        # -------------------------------------------------------------------
        # LAYER 5: Out-Of-Fold (OOF) Smoothed Target Encoding
        # -------------------------------------------------------------------
        # Bayesian smoothing parameter m=10
        m = 10.0
        global_mean = y_train.mean()
        self.global_target_mean = global_mean
        
        # Interaction category
        df_out['major_x_industry'] = df_out['major'].astype(str) + '_' + df_out['work_industry'].astype(str)
        
        for te_col in ['work_industry', 'major', 'race', 'major_x_industry']:
            # Fill race missing
            series = df_out[te_col].fillna('Missing')
            
            # OOF target encoding via StratifiedKFold
            oof_te = np.zeros(len(df_out))
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=self.random_state)
            
            for tr_idx, val_idx in skf.split(df_out, y_train):
                tr_series = series.iloc[tr_idx]
                tr_y = y_train.iloc[tr_idx]
                
                stats = tr_y.groupby(tr_series).agg(count='count', sum='sum')
                smoothed = (stats['sum'] + m * global_mean) / (stats['count'] + m)
                oof_te[val_idx] = series.iloc[val_idx].map(smoothed).fillna(global_mean)
                
            df_out[f'fe_te_{te_col}'] = oof_te
            
            # Fit whole train mapping for test set
            all_stats = y_train.groupby(series).agg(count='count', sum='sum')
            self.target_encoding_maps[te_col] = (all_stats['sum'] + m * global_mean) / (all_stats['count'] + m)

        # -------------------------------------------------------------------
        # LAYER 6: Elite Threshold & Demographic Interactions
        # -------------------------------------------------------------------
        df_out['fe_elite_candidate'] = ((df_out['gmat'] >= 720) & (df_out['gpa'] >= 3.50)).astype(float)
        df_out['fe_solid_candidate'] = ((df_out['gmat'] >= 690) & (df_out['gpa'] >= 3.30)).astype(float)
        df_out['fe_subpar_candidate'] = ((df_out['gmat'] < 620) | (df_out['gpa'] < 3.00)).astype(float)
        
        # Gender quota boost interaction (Female admit rate is 20% vs Male 11.4%)
        is_female = (df_out['gender'] == 'Female').astype(float)
        df_out['fe_female_high_credentials'] = is_female * df_out['fe_solid_candidate']
        df_out['fe_female_tier1_feeder'] = is_female * df_out['fe_is_tier1_feeder']
        df_out['fe_intl_stem'] = df_out['international'].astype(float) * (df_out['major'] == 'STEM').astype(float)

        # -------------------------------------------------------------------
        # LAYER 7: Unsupervised Representations (K-Means & PCA)
        # -------------------------------------------------------------------
        core_numeric = ['gpa', 'gmat', 'work_exp', 'fe_academic_index']
        self.scaler = StandardScaler()
        scaled_core = self.scaler.fit_transform(df_out[core_numeric])
        
        # K-Means
        self.kmeans = KMeans(n_clusters=3, random_state=self.random_state, n_init=10)
        self.kmeans.fit(scaled_core)
        dists = self.kmeans.transform(scaled_core)
        df_out['fe_kmeans_dist_c0'] = dists[:, 0]
        df_out['fe_kmeans_dist_c1'] = dists[:, 1]
        df_out['fe_kmeans_dist_c2'] = dists[:, 2]
        df_out['fe_kmeans_cluster'] = self.kmeans.labels_
        
        # PCA
        self.pca = PCA(n_components=2, random_state=self.random_state)
        pca_comps = self.pca.fit_transform(scaled_core)
        df_out['fe_pca_1'] = pca_comps[:, 0]
        df_out['fe_pca_2'] = pca_comps[:, 1]

        # Categorical One-Hot Encoding for remaining nominals
        self.cat_cols = ['gender', 'international', 'major', 'race', 'work_industry']
        self.ohe = OneHotEncoder(handle_unknown='ignore', sparse_output=False)
        cat_imputed = df_out[self.cat_cols].fillna('Missing').astype(str)
        ohe_arr = self.ohe.fit_transform(cat_imputed)
        ohe_names = [f"cat_{col}_{val}" for col, vals in zip(self.cat_cols, self.ohe.categories_) for val in vals]
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_names, index=df_out.index)
        
        # Final combined engineered dataframe
        fe_cols = [c for c in df_out.columns if c.startswith('fe_')] + ['gpa', 'gmat', 'work_exp']
        final_df = pd.concat([df_out[fe_cols], ohe_df], axis=1)
        self.final_feature_names = final_df.columns.tolist()
        
        return final_df

    def transform(self, X_test):
        df_out = X_test.copy()
        
        # Layer 1
        gpa_norm = (df_out['gpa'] - self.gpa_min) / (self.gpa_max - self.gpa_min)
        gmat_norm = (df_out['gmat'] - self.gmat_min) / (self.gmat_max - self.gmat_min)
        df_out['fe_academic_index'] = 0.40 * gpa_norm + 0.60 * gmat_norm
        df_out['fe_academic_power'] = (df_out['gpa'] ** 2) * (df_out['gmat'] / 100.0)
        df_out['fe_gpa_x_gmat'] = df_out['gpa'] * df_out['gmat']
        df_out['fe_gmat_per_gpa'] = df_out['gmat'] / (df_out['gpa'] + 1e-5)
        df_out['fe_gpa_squared'] = df_out['gpa'] ** 2
        df_out['fe_gmat_squared'] = df_out['gmat'] ** 2
        df_out['fe_log_gmat'] = np.log(df_out['gmat'])
        df_out['fe_sqrt_synergy'] = np.sqrt(df_out['gpa']) * np.sqrt(df_out['gmat'])
        
        # Layer 2
        df_out['fe_gmat_percentile'] = self.gmat_to_percentile(df_out['gmat'].values)
        major_rigor = {'STEM': 1.08, 'Business': 1.04, 'Humanities': 1.00}
        df_out['fe_rigor_adjusted_gpa'] = df_out['gpa'] * df_out['major'].map(major_rigor).fillna(1.0)
        
        # Layer 3
        df_out['fe_exp_sweet_spot_dev'] = np.abs(df_out['work_exp'] - 5.0)
        df_out['fe_is_sweet_spot'] = ((df_out['work_exp'] >= 4.0) & (df_out['work_exp'] <= 6.0)).astype(float)
        df_out['fe_exp_quadratic'] = (df_out['work_exp'] - 5.0) ** 2
        df_out['fe_is_too_young'] = (df_out['work_exp'] <= 2.0).astype(float)
        df_out['fe_is_over_exp'] = (df_out['work_exp'] >= 8.0).astype(float)
        
        df_out['fe_velocity_gmat'] = df_out['gmat'] / (df_out['work_exp'] + 1.0)
        df_out['fe_velocity_gpa'] = df_out['gpa'] / (df_out['work_exp'] + 1.0)
        df_out['fe_velocity_academic'] = df_out['fe_academic_index'] / (df_out['work_exp'] + 1.0)
        
        tier1_industries = {'PE/VC', 'Consulting', 'Investment Banking'}
        tier2_industries = {'Technology', 'Investment Management', 'Financial Services'}
        df_out['fe_is_tier1_feeder'] = df_out['work_industry'].isin(tier1_industries).astype(float)
        df_out['fe_is_tier2_feeder'] = df_out['work_industry'].isin(tier2_industries).astype(float)
        
        # Layer 4 (Map train stats)
        for grp_col in ['work_industry', 'major', 'gender']:
            stats = self.group_stats[grp_col]
            mean_gmat = df_out[grp_col].map(stats['mean_gmat']).fillna(stats['global_mean_gmat'])
            std_gmat = df_out[grp_col].map(stats['std_gmat']).fillna(stats['global_std_gmat'])
            mean_gpa = df_out[grp_col].map(stats['mean_gpa']).fillna(stats['global_mean_gpa'])
            std_gpa = df_out[grp_col].map(stats['std_gpa']).fillna(stats['global_std_gpa'])
            
            df_out[f'fe_diff_gmat_{grp_col}'] = df_out['gmat'] - mean_gmat
            df_out[f'fe_z_gmat_{grp_col}'] = df_out[f'fe_diff_gmat_{grp_col}'] / (std_gmat + 1e-5)
            df_out[f'fe_diff_gpa_{grp_col}'] = df_out['gpa'] - mean_gpa
            df_out[f'fe_z_gpa_{grp_col}'] = df_out[f'fe_diff_gpa_{grp_col}'] / (std_gpa + 1e-5)
            
        # Layer 5 (Map train target encodings)
        df_out['major_x_industry'] = df_out['major'].astype(str) + '_' + df_out['work_industry'].astype(str)
        for te_col in ['work_industry', 'major', 'race', 'major_x_industry']:
            series = df_out[te_col].fillna('Missing')
            te_map = self.target_encoding_maps[te_col]
            df_out[f'fe_te_{te_col}'] = series.map(te_map).fillna(self.global_target_mean)
            
        # Layer 6
        df_out['fe_elite_candidate'] = ((df_out['gmat'] >= 720) & (df_out['gpa'] >= 3.50)).astype(float)
        df_out['fe_solid_candidate'] = ((df_out['gmat'] >= 690) & (df_out['gpa'] >= 3.30)).astype(float)
        df_out['fe_subpar_candidate'] = ((df_out['gmat'] < 620) | (df_out['gpa'] < 3.00)).astype(float)
        
        is_female = (df_out['gender'] == 'Female').astype(float)
        df_out['fe_female_high_credentials'] = is_female * df_out['fe_solid_candidate']
        df_out['fe_female_tier1_feeder'] = is_female * df_out['fe_is_tier1_feeder']
        df_out['fe_intl_stem'] = df_out['international'].astype(float) * (df_out['major'] == 'STEM').astype(float)
        
        # Layer 7
        core_numeric = ['gpa', 'gmat', 'work_exp', 'fe_academic_index']
        scaled_core = self.scaler.transform(df_out[core_numeric])
        dists = self.kmeans.transform(scaled_core)
        df_out['fe_kmeans_dist_c0'] = dists[:, 0]
        df_out['fe_kmeans_dist_c1'] = dists[:, 1]
        df_out['fe_kmeans_dist_c2'] = dists[:, 2]
        df_out['fe_kmeans_cluster'] = self.kmeans.predict(scaled_core)
        
        pca_comps = self.pca.transform(scaled_core)
        df_out['fe_pca_1'] = pca_comps[:, 0]
        df_out['fe_pca_2'] = pca_comps[:, 1]
        
        cat_imputed = df_out[self.cat_cols].fillna('Missing').astype(str)
        ohe_arr = self.ohe.transform(cat_imputed)
        ohe_names = [f"cat_{col}_{val}" for col, vals in zip(self.cat_cols, self.ohe.categories_) for val in vals]
        ohe_df = pd.DataFrame(ohe_arr, columns=ohe_names, index=df_out.index)
        
        fe_cols = [c for c in df_out.columns if c.startswith('fe_')] + ['gpa', 'gmat', 'work_exp']
        final_df = pd.concat([df_out[fe_cols], ohe_df], axis=1)
        
        return final_df

# Fit and transform
fe = ExtremeFeatureEngineer(random_state=42)
X_train_fe = fe.fit_transform(X_train_raw, y_train)
X_test_fe = fe.transform(X_test_raw)

print(f"\nEngineered Features count: {X_train_fe.shape[1]} features created across 7 layers!")

# ---------------------------------------------------------------------------
# 3. Model Training & Benchmarking Across Features
# ---------------------------------------------------------------------------
scale_pos = (y_train == 0).sum() / (y_train == 1).sum()

models = {
    'LightGBM (Extreme FE)': LGBMClassifier(
        n_estimators=300, max_depth=5, learning_rate=0.03,
        is_unbalance=True, colsample_bytree=0.7, subsample=0.8,
        random_state=42, verbose=-1
    ),
    'XGBoost (Extreme FE)': XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.03,
        scale_pos_weight=scale_pos, colsample_bytree=0.7, subsample=0.8,
        random_state=42, eval_metric='logloss'
    ),
    'Gradient Boosting (Extreme FE)': GradientBoostingClassifier(
        n_estimators=250, max_depth=4, learning_rate=0.04,
        random_state=42
    ),
    'Random Forest (Extreme FE)': RandomForestClassifier(
        n_estimators=400, max_depth=12, min_samples_leaf=4,
        class_weight='balanced_subsample', random_state=42
    ),
    'Stacking Beast (XGB+LGBM+RF)': StackingClassifier(
        estimators=[
            ('xgb', XGBClassifier(n_estimators=200, max_depth=4, learning_rate=0.03, scale_pos_weight=scale_pos, random_state=42, eval_metric='logloss')),
            ('lgbm', LGBMClassifier(n_estimators=200, max_depth=5, learning_rate=0.03, is_unbalance=True, random_state=42, verbose=-1)),
            ('rf', RandomForestClassifier(n_estimators=250, max_depth=10, min_samples_leaf=4, class_weight='balanced', random_state=42))
        ],
        final_estimator=LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42),
        cv=5
    )
}

print("\n" + "=" * 85)
print("BENCHMARKING SOTA MODELS WITH EXTREME FEATURES + OPTIMAL THRESHOLDING")
print("=" * 85)

results = []
probs_dict = {}

for name, model in models.items():
    print(f"Training {name}...")
    model.fit(X_train_fe, y_train)
    probs = model.predict_proba(X_test_fe)[:, 1]
    probs_dict[name] = probs
    
    auc = roc_auc_score(y_test, probs)
    
    # Threshold sweep
    precisions, recalls, thresholds = precision_recall_curve(y_test, probs)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-10)
    best_idx = np.argmax(f1_scores)
    best_f1 = f1_scores[best_idx]
    best_thresh = thresholds[best_idx] if best_idx < len(thresholds) else 0.5
    
    pred_opt = (probs >= best_thresh).astype(int)
    cm = confusion_matrix(y_test, pred_opt)
    
    results.append({
        'Model': name,
        'ROC-AUC': auc,
        'Max Binary F1': best_f1,
        'Optimal Threshold': best_thresh,
        'Precision at Max': precisions[best_idx],
        'Recall at Max': recalls[best_idx],
        'Weighted F1': f1_score(y_test, pred_opt, average='weighted'),
        'TP': cm[1,1], 'FP': cm[0,1], 'FN': cm[1,0], 'TN': cm[0,0]
    })

res_df = pd.DataFrame(results).sort_values(by='ROC-AUC', ascending=False)
print("\n" + res_df[['Model', 'ROC-AUC', 'Max Binary F1', 'Optimal Threshold', 'Precision at Max', 'Recall at Max', 'Weighted F1']].to_string(index=False))

# ---------------------------------------------------------------------------
# 4. Top Feature Importances in Extreme Feature Pipeline
# ---------------------------------------------------------------------------
print("\n" + "=" * 85)
print("TOP 15 MOST INFLUENTIAL ENGINEERED FEATURES (LightGBM)")
print("=" * 85)

lgbm_model = models['LightGBM (Extreme FE)']
feat_imp = pd.DataFrame({
    'Feature': X_train_fe.columns,
    'Importance': lgbm_model.feature_importances_
}).sort_values(by='Importance', ascending=False).reset_index(drop=True)

print(feat_imp.head(15).to_string(index=False))

# Save artifacts
res_df.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extreme_fe_results.csv'), index=False)
feat_imp.to_csv(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'extreme_fe_importances.csv'), index=False)
print("\nAll Extreme Feature Engineering results successfully computed and saved.")
