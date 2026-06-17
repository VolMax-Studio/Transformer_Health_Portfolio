"""
dga_stress_test.py — Transformer DGA Stress-Testing & Boundary Mapping
Loads public 589-sample DGA dataset, trains Random Forest, runs noise stress test,
and projects decision boundaries onto standard 2D triangular coordinates.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Import physical Duval rule from src
from src.dga_analyzer import duval_triangle_zone

# Set random seed for reproducibility
np.random.seed(42)

# Load and clean dataset
excel_path = "dataset_589.xlsx"
df = pd.read_excel(excel_path)

# Map Chinese labels to standard Western DGA codes
label_map = {
    '局部放电': 'PD',
    '低能放电': 'D1',
    '高能放电': 'D2',
    '低温过热': 'T1',
    '中温过热': 'T2',
    '高温过热': 'T3'
}
df['Fault'] = df['故障类型'].map(label_map)

# Features and target
gases = ['H2', 'CH4', 'C2H6', 'C2H4', 'C2H2']
X_raw = df[gases].copy()
y = df['Fault'].copy()

# Add Duval ratios as features for the ML model to assist training
tot_duval = X_raw['CH4'] + X_raw['C2H4'] + X_raw['C2H2']
X_raw['pct_CH4'] = (X_raw['CH4'] / tot_duval * 100).fillna(0.0)
X_raw['pct_C2H4'] = (X_raw['C2H4'] / tot_duval * 100).fillna(0.0)
X_raw['pct_C2H2'] = (X_raw['C2H2'] / tot_duval * 100).fillna(0.0)

# Train-test split (80-20, stratified)
X_train, X_test, y_train, y_test = train_test_split(
    X_raw, y, test_size=0.20, random_state=42, stratify=y
)

# Train ML Classifier (Random Forest)
clf = RandomForestClassifier(n_estimators=100, max_depth=6, random_state=42)
clf.fit(X_train, y_train)

# Physical Duval Classifier helper
def classify_physical_duval(X_df):
    preds = []
    for _, row in X_df.iterrows():
        zone = duval_triangle_zone(row['CH4'], row['C2H4'], row['C2H2'])
        # Map physical zone names if necessary, otherwise use returned zone
        preds.append(zone)
    return np.array(preds)

# Stress test loop over multiplicative Gaussian noise levels
sigmas = [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5]
results = {
    'noise_levels': sigmas,
    'rf_accuracy': [],
    'duval_accuracy': []
}

rng = np.random.default_rng(42)

for s in sigmas:
    # Clone test features
    X_test_noisy = X_test.copy()
    
    # Inject multiplicative noise only to the raw gas concentrations
    for gas in gases:
        noise = rng.normal(0, s, size=len(X_test))
        X_test_noisy[gas] = X_test[gas] * (1.0 + noise)
    
    # Clip to avoid negative or zero concentrations
    X_test_noisy[gases] = np.clip(X_test_noisy[gases], 1e-4, None)
    
    # Recompute percentages for the noisy gases
    tot_noisy = X_test_noisy['CH4'] + X_test_noisy['C2H4'] + X_test_noisy['C2H2']
    X_test_noisy['pct_CH4'] = (X_test_noisy['CH4'] / tot_noisy * 100).fillna(0.0)
    X_test_noisy['pct_C2H4'] = (X_test_noisy['C2H4'] / tot_noisy * 100).fillna(0.0)
    X_test_noisy['pct_C2H2'] = (X_test_noisy['C2H2'] / tot_noisy * 100).fillna(0.0)
    
    # Evaluate ML Classifier
    y_pred_rf = clf.predict(X_test_noisy)
    acc_rf = accuracy_score(y_test, y_pred_rf)
    results['rf_accuracy'].append(float(acc_rf))
    
    # Evaluate Physical Duval Classifier
    y_pred_duval = classify_physical_duval(X_test_noisy)
    acc_duval = accuracy_score(y_test, y_pred_duval)
    results['duval_accuracy'].append(float(acc_duval))

print("Stress Test Results:")
for i, s in enumerate(sigmas):
    print(f"  Noise level (sigma) = {s:4.2f} | RF Acc = {results['rf_accuracy'][i]:.3f} | Duval Acc = {results['duval_accuracy'][i]:.3f}")

# Save results to JSON
os.makedirs("results", exist_ok=True)
with open("results/verified_dga_results.json", "w") as f:
    json.dump(results, f, indent=2)

# --- 2D Triangular Coordinate Projection Helpers ---
def to_cartesian(pct_c2h4, pct_ch4, pct_c2h2):
    """
    Project 3 components summing to 100 onto 2D Cartesian triangle space.
    Let C2H2 be at (0, 0), C2H4 at (100, 0), and CH4 at (50, 50 * sqrt(3)).
    """
    tot = pct_c2h4 + pct_ch4 + pct_c2h2
    # Ensure they sum to 100 for safety
    c2h4 = pct_c2h4 / tot * 100
    ch4 = pct_ch4 / tot * 100
    x = c2h4 + 0.5 * ch4
    y = ch4 * (np.sqrt(3) / 2.0)
    return x, y

def to_barycentric(x, y):
    """Invert Cartesian triangle coordinates back to percentages."""
    pct_ch4 = y / (np.sqrt(3) / 2.0)
    pct_c2h4 = x - 0.5 * pct_ch4
    pct_c2h2 = 100.0 - pct_ch4 - pct_c2h4
    return pct_c2h4, pct_ch4, pct_c2h2

# Generate a grid over the bounding box of the triangle to map decision boundaries
x_grid = np.linspace(0, 100, 250)
y_grid = np.linspace(0, 100 * (np.sqrt(3)/2.0), 250)
xx, yy = np.meshgrid(x_grid, y_grid)

# Map labels to integer index for coloring
class_names = ['PD', 'D1', 'D2', 'T1', 'T2', 'T3']
class_to_idx = {name: idx for idx, name in enumerate(class_names)}
class_colors = ['lightblue', 'tan', 'orange', 'lightgreen', 'yellowgreen', 'tomato']

grid_duval = np.full(xx.shape, np.nan)
grid_rf = np.full(xx.shape, np.nan)

# Fill the decision grid inside the triangle boundaries
for i in range(xx.shape[0]):
    for j in range(xx.shape[1]):
        x, y = xx[i, j], yy[i, j]
        c2h4, ch4, c2h2 = to_barycentric(x, y)
        if c2h4 >= -1e-5 and ch4 >= -1e-5 and c2h2 >= -1e-5:
            # Physical Duval prediction
            z = duval_triangle_zone(ch4, c2h4, c2h2)
            if z in class_to_idx:
                grid_duval[i, j] = class_to_idx[z]
            
            # Random Forest prediction (requires synthetic feature structure)
            # We assume dummy absolute values matching the ratio for the RF model
            row_features = pd.DataFrame([{
                'H2': 100.0, 'CH4': float(ch4), 'C2H6': 50.0, 'C2H4': float(c2h4), 'C2H2': float(c2h2),
                'pct_CH4': float(ch4), 'pct_C2H4': float(c2h4), 'pct_C2H2': float(c2h2)
            }])
            pred = clf.predict(row_features)[0]
            if pred in class_to_idx:
                grid_rf[i, j] = class_to_idx[pred]

# Project all dataset points onto Cartesian coordinates
xs_data, ys_data = [], []
for _, row in df.iterrows():
    tot = row['CH4'] + row['C2H4'] + row['C2H2']
    if tot > 0:
        x, y = to_cartesian(row['C2H4'], row['CH4'], row['C2H2'])
        xs_data.append(x)
        ys_data.append(y)
    else:
        xs_data.append(np.nan)
        ys_data.append(np.nan)
df['x_proj'] = xs_data
df['y_proj'] = ys_data
df = df.dropna(subset=['x_proj'])

# Plotting the diagnostic figure
fig = plt.figure(figsize=(18, 8))
plt.rcParams['font.family'] = 'sans-serif'

# Subplot 1: Stress-Test Curves
ax1 = fig.add_subplot(1, 2, 1)
ax1.plot(sigmas, results['rf_accuracy'], marker='o', lw=2.5, color='#1f77b4', label='Random Forest Classifier (ML)')
ax1.plot(sigmas, results['duval_accuracy'], marker='s', lw=2.5, color='#d62728', label='Duval Triangle (Physical Baseline)')
ax1.set_title("DGA Classifier Accuracy vs Sensor Noise Level", fontsize=14, fontweight='bold', pad=15)
ax1.set_xlabel(r"Noise Severity ($\sigma$ - fraction of gas level)", fontsize=12)
ax1.set_ylabel("Classification Accuracy", fontsize=12)
ax1.grid(True, alpha=0.3)
ax1.legend(fontsize=11)
ax1.set_ylim(0.4, 1.02)

# Subplot 2: Decision Boundaries & Mapped Samples
ax2 = fig.add_subplot(1, 2, 2)
# Draw decision regions of physical Duval model
ax2.contourf(xx, yy, grid_duval, levels=np.arange(len(class_names)+1)-0.5,
             colors=class_colors, alpha=0.35)

# Plot actual data points grouped by fault class
for name in class_names:
    sub = df[df['Fault'] == name]
    ax2.scatter(sub['x_proj'], sub['y_proj'], label=name, edgecolors='black', linewidths=0.5, s=40)

# Draw triangle frame
triangle = patches.Polygon([[0, 0], [100, 0], [50, 100 * (np.sqrt(3)/2.0)]], closed=True, fill=None, edgecolor='black', lw=2)
ax2.add_patch(triangle)

# Add vertex labels
ax2.text(-3, -4, "%C2H2 = 100", fontsize=11, fontweight='bold', ha='center')
ax2.text(103, -4, "%C2H4 = 100", fontsize=11, fontweight='bold', ha='center')
ax2.text(50, 100 * (np.sqrt(3)/2.0) + 2, "%CH4 = 100", fontsize=11, fontweight='bold', ha='center')

ax2.set_xlim(-10, 110)
ax2.set_ylim(-10, 100)
ax2.set_aspect('equal')
ax2.set_title("Duval Triangle Physical Zones & DGA Fault Samples", fontsize=14, fontweight='bold', pad=15)
ax2.axis('off')
ax2.legend(title="Fault Type", bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=11)

plt.suptitle("Transformer Dissolved Gas Analysis (DGA) Stress-Testing & Mapping\n"
             "Comparing Physical Duval Boundaries vs Random Forest (ML) under Sensor Noise",
             fontsize=16, fontweight='bold', y=0.98)
plt.savefig("results/dga_stress_test.png", dpi=150, bbox_inches='tight')
plt.close()

print("\nSUCCESS: Saved results/dga_stress_test.png and results/verified_dga_results.json")
