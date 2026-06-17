"""
dga_stress_test.py — Transformer DGA Stress-Testing & Boundary Mapping
========================================================================
Performs:
1. Deduplication and cleaning of the 2321-sample DGA dataset (data.xlsx).
2. Exact Duval Triangle 1 classification via 2D Cartesian polygon inclusion.
3. Level 1 Anomaly Detection (Normal vs Faulty) Stratified 5-Fold CV.
4. Level 2 Fault Classification (PD, D1, D2, T1, T2, T3) comparing:
   - Physical Duval Triangle 1 (Baseline)
   - 3-Gas Random Forest Classifier
   - 5-Gas + Ratios Random Forest Classifier
5. Confidence Calibration of the 5-Gas ML Model.
6. Sensor Noise Stress-Testing (IEEE C57.104 repeatability limit: ±20%).
7. Generates a premium 3-panel diagnostic visualization.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.path import Path
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Set seed for reproducibility
np.random.seed(42)

# --- 1. Load and Clean Dataset ---
excel_path = "data.xlsx"
df = pd.read_excel(excel_path)
print(f"Raw dataset loaded. Shape: {df.shape}")

# Deduplicate to prevent train-test leakage
df_dedup = df.drop_duplicates().copy()
print(f"Deduplicated dataset shape: {df_dedup.shape} (Removed {len(df) - len(df_dedup)} duplicates)")

# Map Chinese labels to Western codes
label_map = {
    '正常': 'Normal',
    '局部放电': 'PD',
    '低能放电': 'D1',
    '高能放电': 'D2',
    '低温过热': 'T1',
    '中温过热': 'T2',
    '高温过热': 'T3'
}
df_dedup['Fault'] = df_dedup['故障类型'].map(label_map)

# Split into normal and fault-only subsets
df_fault = df_dedup[df_dedup['Fault'] != 'Normal'].copy()
print(f"Normal samples: {sum(df_dedup['Fault'] == 'Normal')}")
print(f"Fault samples: {len(df_fault)}")

# --- 2. Precise Duval Triangle 1 Implementation ---
# Coordinates derived from standard IEC 60599 / duvals_triangle_plotter constants
# a: Methane (CH4), b: Acetylene (C2H2), c: Ethylene (C2H4)
regions = {
    'PD': {'a': [98, 1, 98], 'b': [0, 0, 2], 'c': [2, 0, 0]},
    'D1': {'a': [0, 0, 64, 87], 'b': [1, 77, 13, 13], 'c': [0, 23, 23, 0]},
    'D2': {'a': [0, 0, 31, 47, 64], 'b': [77, 29, 29, 13, 13], 'c': [23, 71, 40, 40, 23]},
    'DT': {'a': [0, 0, 35, 46, 96, 87, 47, 31], 'b': [29, 15, 15, 4, 4, 13, 13, 29], 'c': [71, 85, 50, 50, 0, 0, 40, 40]},
    'T1': {'a': [76, 80, 98, 98, 96], 'b': [4, 0, 0, 2, 4], 'c': [20, 20, 2, 0, 0]},
    'T2': {'a': [46, 50, 80, 76], 'b': [4, 0, 0, 4], 'c': [50, 50, 20, 20]},
    'T3': {'a': [0, 0, 50, 35], 'b': [15, 0, 0, 15], 'c': [85, 1, 50, 50]}
}

def to_cartesian_coords(a_ch4, b_c2h2, c_c2h4):
    a = np.array(a_ch4)
    b = np.array(b_c2h2)
    c = np.array(c_c2h4)
    tot = a + b + c
    pct_ch4 = a / tot * 100
    pct_c2h2 = b / tot * 100
    pct_c2h4 = c / tot * 100
    x = pct_c2h4 + 0.5 * pct_ch4
    y = pct_ch4 * (np.sqrt(3)/2)
    return np.column_stack((x, y))

# Pre-build Path objects
polygon_paths = {}
for name, coords in regions.items():
    poly_pts = to_cartesian_coords(coords['a'], coords['b'], coords['c'])
    polygon_paths[name] = Path(poly_pts)

def classify_duval_precise(ch4, c2h4, c2h2):
    tot = ch4 + c2h4 + c2h2
    if tot < 1e-6:
        return 'undefined'
    pct_ch4 = ch4 / tot * 100
    pct_c2h2 = c2h2 / tot * 100
    pct_c2h4 = c2h4 / tot * 100
    x = pct_c2h4 + 0.5 * pct_ch4
    y = pct_ch4 * (np.sqrt(3)/2)
    
    # Try strict containment first
    for name, path in polygon_paths.items():
        if path.contains_point((x, y)):
            return name
            
    # Try with small tolerances to handle points resting on boundaries
    for tol in [1e-5, 1e-3, 0.1, 0.5, 1.0, 2.0]:
        for name, path in polygon_paths.items():
            if path.contains_point((x, y), radius=tol) or path.contains_point((x, y), radius=-tol):
                return name
    return 'undefined'

# Verify exact physical Duval accuracy
duval_correct = 0
for _, row in df_fault.iterrows():
    pred = classify_duval_precise(row['CH4'], row['C2H4'], row['C2H2'])
    if pred == row['Fault']:
        duval_correct += 1
duval_baseline_acc = duval_correct / len(df_fault)
print(f"Verified Duval Triangle 1 Baseline Accuracy: {duval_baseline_acc:.4%}")

# --- 3. Feature Engineering ---
# Normalized percentages and standard diagnostic ratios
def add_features(df_sub):
    df_out = df_sub.copy()
    sum_gases = df_out['H2'] + df_out['CH4'] + df_out['C2H6'] + df_out['C2H4'] + df_out['C2H2']
    for g in ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']:
        df_out[f'pct_{g}_of_5'] = (df_out[g] / sum_gases * 100).fillna(0.0)
        
    tot_duval = df_out['CH4'] + df_out['C2H4'] + df_out['C2H2']
    df_out['pct_CH4'] = (df_out['CH4'] / tot_duval * 100).fillna(0.0)
    df_out['pct_C2H4'] = (df_out['C2H4'] / tot_duval * 100).fillna(0.0)
    df_out['pct_C2H2'] = (df_out['C2H2'] / tot_duval * 100).fillna(0.0)
    
    # Core ratios (Rogers / IEC / Doernenburg)
    df_out['ratio_ch4_h2'] = df_out['CH4'] / (df_out['H2'] + 1e-5)
    df_out['ratio_c2h2_ch4'] = df_out['C2H2'] / (df_out['CH4'] + 1e-5)
    df_out['ratio_c2h4_c2h6'] = df_out['C2H4'] / (df_out['C2H6'] + 1e-5)
    df_out['ratio_c2h2_c2h4'] = df_out['C2H2'] / (df_out['C2H4'] + 1e-5)
    df_out['ratio_c2h6_ch4'] = df_out['C2H6'] / (df_out['CH4'] + 1e-5)
    return df_out

df_fault_feat = add_features(df_fault)

# Column subsets
cols_3gas = ['CH4', 'C2H4', 'C2H2', 'pct_CH4', 'pct_C2H4', 'pct_C2H2']
cols_5gas = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2',
             'pct_H2_of_5', 'pct_CH4_of_5', 'pct_C2H6_of_5', 'pct_C2H4_of_5', 'pct_C2H2_of_5',
             'pct_CH4', 'pct_C2H4', 'pct_C2H2',
             'ratio_ch4_h2', 'ratio_c2h2_ch4', 'ratio_c2h4_c2h6', 'ratio_c2h2_c2h4', 'ratio_c2h6_ch4']

# --- 4. Cross Validation Evaluator ---
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Level 1 Anomaly Detection (Normal vs Fault)
y_lvl1 = (df_dedup['Fault'] != 'Normal').astype(int)
X_lvl1 = df_dedup[['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']]
lvl1_scores = []
for tr, te in cv.split(X_lvl1, y_lvl1):
    clf_lvl1 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_lvl1.fit(X_lvl1.iloc[tr], y_lvl1.iloc[tr])
    lvl1_scores.append(accuracy_score(y_lvl1.iloc[te], clf_lvl1.predict(X_lvl1.iloc[te])))
lvl1_cv_acc = np.mean(lvl1_scores)
print(f"Level 1 (Normal vs Faulty) CV Accuracy: {lvl1_cv_acc:.4%}")

# Level 2 Fault Classification
y_lvl2 = df_fault_feat['Fault']
X_3g = df_fault_feat[cols_3gas]
X_5g = df_fault_feat[cols_5gas]

rf3_scores = []
rf5_scores = []
for tr, te in cv.split(X_5g, y_lvl2):
    # Train 3-Gas RF
    clf3 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf3.fit(X_3g.iloc[tr], y_lvl2.iloc[tr])
    rf3_scores.append(accuracy_score(y_lvl2.iloc[te], clf3.predict(X_3g.iloc[te])))
    
    # Train 5-Gas + Ratios RF
    clf5 = RandomForestClassifier(n_estimators=100, random_state=42)
    clf5.fit(X_5g.iloc[tr], y_lvl2.iloc[tr])
    rf5_scores.append(accuracy_score(y_lvl2.iloc[te], clf5.predict(X_5g.iloc[te])))

rf3_cv_acc = np.mean(rf3_scores)
rf5_cv_acc = np.mean(rf5_scores)
print(f"Level 2 RF 3-Gas CV Accuracy: {rf3_cv_acc:.4%}")
print(f"Level 2 RF 5-Gas + Ratios CV Accuracy: {rf5_cv_acc:.4%}")

# --- 5. Confidence Calibration Analysis ---
# Generate out-of-fold predicted probabilities for 5-Gas model
oof_preds = []
oof_probs = []
oof_true = []
for tr, te in cv.split(X_5g, y_lvl2):
    clf_calib = RandomForestClassifier(n_estimators=100, random_state=42)
    clf_calib.fit(X_5g.iloc[tr], y_lvl2.iloc[tr])
    probs = clf_calib.predict_proba(X_5g.iloc[te])
    max_probs = np.max(probs, axis=1)
    preds = clf_calib.classes_[np.argmax(probs, axis=1)]
    
    oof_preds.extend(preds)
    oof_probs.extend(max_probs)
    oof_true.extend(y_lvl2.iloc[te])

oof_preds = np.array(oof_preds)
oof_probs = np.array(oof_probs)
oof_true = np.array(oof_true)

# Bin confidence values
bins = [0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
calib_results = []
for idx in range(len(bins)-1):
    lower = bins[idx]
    upper = bins[idx+1]
    # Handle upper bound inclusive
    if upper == 1.0:
        mask = (oof_probs >= lower) & (oof_probs <= upper)
    else:
        mask = (oof_probs >= lower) & (oof_probs < upper)
        
    bin_samples = sum(mask)
    if bin_samples > 0:
        bin_acc = accuracy_score(oof_true[mask], oof_preds[mask])
        mean_conf = np.mean(oof_probs[mask])
    else:
        bin_acc = 0.0
        mean_conf = 0.0
    calib_results.append({
        'bin': f"[{lower:.1f}, {upper:.1f}]",
        'samples': int(bin_samples),
        'expected_conf': float(mean_conf),
        'actual_acc': float(bin_acc)
    })

print("\nConfidence Calibration Analysis:")
for b in calib_results:
    print(f"  Bin {b['bin']} | Samples = {b['samples']:3d} | Expected Conf = {b['expected_conf']:.3f} | Actual Acc = {b['actual_acc']:.3f}")

# --- 6. Repeatability & Sensor Noise Stress-Test ---
# Multiplicative noise sweep. Repeatability limit ±20% corresponds to 3-sigma at 0.20 (sigma = 0.067)
sigmas = np.linspace(0.0, 0.20, 11)
noise_results = {
    'noise_sigmas': sigmas.tolist(),
    'rf_accuracies': [],
    'duval_accuracies': []
}

rng = np.random.default_rng(42)
gases_list = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']

for s in sigmas:
    rf_fold_scores = []
    duval_fold_scores = []
    
    for tr, te in cv.split(X_5g, y_lvl2):
        X_train_fold = X_5g.iloc[tr].copy()
        X_test_fold = X_5g.iloc[te].copy()
        
        # Add noise to test concentrations
        test_noisy = X_test_fold[gases_list].copy()
        for g in gases_list:
            noise_factor = rng.normal(0, s, size=len(test_noisy))
            test_noisy[g] = test_noisy[g] * (1.0 + noise_factor)
        test_noisy = np.clip(test_noisy, 1e-4, None)
        
        # Build features for train/test folds
        # Compute percentages
        sum_train = X_train_fold[gases_list].sum(axis=1)
        sum_test = test_noisy.sum(axis=1)
        for g in gases_list:
            X_train_fold[f'pct_{g}_of_5'] = (X_train_fold[g] / sum_train * 100).fillna(0.0)
            X_test_fold[f'pct_{g}_of_5'] = (test_noisy[g] / sum_test * 100).fillna(0.0)
            X_test_fold[g] = test_noisy[g]
            
        tot_d_train = X_train_fold['CH4'] + X_train_fold['C2H4'] + X_train_fold['C2H2']
        tot_d_test = X_test_fold['CH4'] + X_test_fold['C2H4'] + X_test_fold['C2H2']
        
        X_train_fold['pct_CH4'] = (X_train_fold['CH4'] / tot_d_train * 100).fillna(0.0)
        X_train_fold['pct_C2H4'] = (X_train_fold['C2H4'] / tot_d_train * 100).fillna(0.0)
        X_train_fold['pct_C2H2'] = (X_train_fold['C2H2'] / tot_d_train * 100).fillna(0.0)
        
        X_test_fold['pct_CH4'] = (X_test_fold['CH4'] / tot_d_test * 100).fillna(0.0)
        X_test_fold['pct_C2H4'] = (X_test_fold['C2H4'] / tot_d_test * 100).fillna(0.0)
        X_test_fold['pct_C2H2'] = (X_test_fold['C2H2'] / tot_d_test * 100).fillna(0.0)
        
        # Compute ratios
        for df_tmp in [X_train_fold, X_test_fold]:
            df_tmp['ratio_ch4_h2'] = df_tmp['CH4'] / (df_tmp['H2'] + 1e-5)
            df_tmp['ratio_c2h2_ch4'] = df_tmp['C2H2'] / (df_tmp['CH4'] + 1e-5)
            df_tmp['ratio_c2h4_c2h6'] = df_tmp['C2H4'] / (df_tmp['C2H6'] + 1e-5)
            df_tmp['ratio_c2h2_c2h4'] = df_tmp['C2H2'] / (df_tmp['C2H4'] + 1e-5)
            df_tmp['ratio_c2h6_ch4'] = df_tmp['C2H6'] / (df_tmp['CH4'] + 1e-5)
            
        # Evaluate RF 5-Gas Model
        clf_noisy = RandomForestClassifier(n_estimators=100, random_state=42)
        clf_noisy.fit(X_train_fold[cols_5gas], y_lvl2.iloc[tr])
        rf_fold_scores.append(accuracy_score(y_lvl2.iloc[te], clf_noisy.predict(X_test_fold[cols_5gas])))
        
        # Evaluate Physical Duval
        duval_preds = []
        for _, row in X_test_fold.iterrows():
            duval_preds.append(classify_duval_precise(row['CH4'], row['C2H4'], row['C2H2']))
        duval_fold_scores.append(accuracy_score(y_lvl2.iloc[te], duval_preds))
        
    noise_results['rf_accuracies'].append(float(np.mean(rf_fold_scores)))
    noise_results['duval_accuracies'].append(float(np.mean(duval_fold_scores)))

print("\nSensor Noise Sweep Results:")
for idx, s in enumerate(sigmas):
    print(f"  Sigma = {s:.3f} | RF Acc = {noise_results['rf_accuracies'][idx]:.4f} | Duval Acc = {noise_results['duval_accuracies'][idx]:.4f}")

# --- 7. Save Verified Results to JSON ---
os.makedirs("results", exist_ok=True)
output_payload = {
    'level1_anomaly_cv_accuracy': float(lvl1_cv_acc),
    'level2_duval_baseline_accuracy': float(duval_baseline_acc),
    'level2_rf_3gas_cv_accuracy': float(rf3_cv_acc),
    'level2_rf_5gas_cv_accuracy': float(rf5_cv_acc),
    'confidence_calibration': calib_results,
    'noise_stress_sweep': noise_results
}
with open("results/verified_dga_results.json", "w") as f:
    json.dump(output_payload, f, indent=2)

# --- 8. Premium 3-Panel Visualizations ---
fig = plt.figure(figsize=(22, 7), dpi=150)
plt.rcParams['font.family'] = 'sans-serif'
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']

# Colors
color_rf = "#3182bd"
color_duval = "#de2d26"

# 1. Noise Degradation Subplot
ax1 = fig.add_subplot(1, 3, 1)
ax1.plot(sigmas, noise_results['rf_accuracies'], marker='o', lw=2.5, color=color_rf, label='5-Gas + Ratios Random Forest (ML)')
ax1.plot(sigmas, noise_results['duval_accuracies'], marker='s', lw=2.5, color=color_duval, label='Duval Triangle 1 (Physical)')
# Repeatability limit line
ax1.axvline(x=0.067, color='#737373', linestyle='--', alpha=0.8, label='IEEE C57.104 Limit (±20% at 3σ)')
ax1.set_title("Classifier Robustness vs Sensor Noise", fontsize=13, fontweight='bold', pad=12)
ax1.set_xlabel(r"Sensor Noise Standard Deviation ($\sigma$)", fontsize=11)
ax1.set_ylabel("Classification Accuracy", fontsize=11)
ax1.grid(True, alpha=0.3, linestyle=':')
ax1.set_xlim(-0.01, 0.21)
ax1.set_ylim(0.35, 0.85)
ax1.legend(fontsize=9, loc='lower left')

# 2. Calibration Subplot
ax2 = fig.add_subplot(1, 3, 2)
expected_c = [b['expected_conf'] for b in calib_results]
actual_a = [b['actual_acc'] for b in calib_results]
bin_labels = [b['bin'] for b in calib_results]
ax2.plot(expected_c, actual_a, marker='D', color='#31a354', lw=2.5, label='RF Model Calibration')
ax2.plot([0.5, 1.0], [0.5, 1.0], linestyle='--', color='#969696', label='Perfect Calibration')
ax2.set_title("ML Model Confidence Calibration", fontsize=13, fontweight='bold', pad=12)
ax2.set_xlabel("Mean Predicted Confidence", fontsize=11)
ax2.set_ylabel("Actual Bin Accuracy", fontsize=11)
ax2.grid(True, alpha=0.3, linestyle=':')
ax2.set_xlim(0.48, 1.02)
ax2.set_ylim(0.48, 1.02)
ax2.legend(fontsize=9, loc='upper left')

# 3. Duval Triangle Boundary Projection Map
ax3 = fig.add_subplot(1, 3, 3)

# Define Cartesian triangle frame
# Vertices: C2H2 (0,0), C2H4 (100,0), CH4 (50, 100 * sqrt(3)/2)
h_tri = 100.0 * (np.sqrt(3)/2.0)
triangle = patches.Polygon([[0.0, 0.0], [100.0, 0.0], [50.0, h_tri]], closed=True, fill=None, edgecolor='#252525', lw=2)
ax3.add_patch(triangle)

# Plot polygon zones with clean styling
zone_colors = {
    'PD': '#ccebc5',
    'D1': '#ffffb3',
    'D2': '#bebada',
    'DT': '#fb8072',
    'T1': '#80b1d3',
    'T2': '#fdb462',
    'T3': '#8dd3c7'
}

for name, coords in regions.items():
    poly_pts = to_cartesian_coords(coords['a'], coords['b'], coords['c'])
    patch = patches.Polygon(poly_pts, closed=True, facecolor=zone_colors[name], edgecolor='#737373', alpha=0.4, label=name)
    ax3.add_patch(patch)

# Project and scatter data points (Fault samples)
class_names = ['PD', 'D1', 'D2', 'T1', 'T2', 'T3']
scatter_colors = {
    'PD': '#4daf4a',
    'D1': '#ffff33',
    'D2': '#984ea3',
    'T1': '#377eb8',
    'T2': '#ff7f00',
    'T3': '#e41a1c'
}

tot_df = df_fault['CH4'] + df_fault['C2H4'] + df_fault['C2H2']
df_fault['pct_CH4'] = df_fault['CH4'] / tot_df * 100
df_fault['pct_C2H4'] = df_fault['C2H4'] / tot_df * 100
df_fault['pct_C2H2'] = df_fault['C2H2'] / tot_df * 100
df_fault['x_proj'] = df_fault['pct_C2H4'] + 0.5 * df_fault['pct_CH4']
df_fault['y_proj'] = df_fault['pct_CH4'] * (np.sqrt(3)/2)

for cls in class_names:
    sub = df_fault[df_fault['Fault'] == cls]
    ax3.scatter(sub['x_proj'], sub['y_proj'], label=f"{cls} Sample", color=scatter_colors[cls], edgecolors='#252525', s=25, alpha=0.85, zorder=5)

ax3.text(-4.0, -4.0, "%C2H2 = 100", fontsize=9, fontweight='bold', ha='center')
ax3.text(104.0, -4.0, "%C2H4 = 100", fontsize=9, fontweight='bold', ha='center')
ax3.text(50.0, h_tri + 2.0, "%CH4 = 100", fontsize=9, fontweight='bold', ha='center')

ax3.set_xlim(-10, 110)
ax3.set_ylim(-10, 95)
ax3.set_aspect('equal')
ax3.set_title("Duval Triangle 1 Boundary Projection Map", fontsize=13, fontweight='bold', pad=12)
ax3.axis('off')

# Single legend for the triangle zones and markers
# Dedup legend handles
handles, labels = ax3.get_legend_handles_labels()
by_label = dict(zip(labels, handles))
ax3.legend(by_label.values(), by_label.keys(), loc='center left', bbox_to_anchor=(0.95, 0.5), fontsize=8, frameon=True)

plt.suptitle("Transformer Dissolved Gas Analysis (DGA) Stress-Testing & ML Boundary Mapping\n"
             "Evaluating Exact Duval Triangle 1 vs. Random Forest Classifiers under Sensor Noise",
             fontsize=15, fontweight='bold', y=0.98)
plt.subplots_adjust(top=0.85, wspace=0.25)
plt.savefig("results/dga_stress_test.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nSUCCESS: Saved results/dga_stress_test.png and results/verified_dga_results.json")
