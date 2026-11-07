import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

# Set overall plot styling
plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')
plt.rcParams['font.sans-serif'] = 'Helvetica, Arial, DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

os.makedirs('figures', exist_ok=True)

# Load dataset
df = pd.read_csv('data/Wholesale_customers_data.csv')
features = ['Fresh', 'Milk', 'Grocery', 'Frozen', 'Detergents_Paper', 'Delicassen']
X = df[features]

# 1. Spending Distributions (Boxplots + Log-scale KDE)
fig, axes = plt.subplots(2, 3, figsize=(16, 9))
palette = sns.color_palette("mako", 6)

for idx, (col, ax) in enumerate(zip(features, axes.flatten())):
    sns.boxplot(y=df[col], ax=ax, color=palette[idx], width=0.4, fliersize=3)
    ax.set_title(f'Spending Distribution: {col}', fontsize=12, fontweight='bold', pad=8)
    ax.set_ylabel('Annual Spending (m.u.)', fontsize=10)
    ax.ticklabel_format(style='plain', axis='y')
    ax.grid(True, linestyle='--', alpha=0.6)

plt.suptitle('Wholesale Customers - Annual Spending Distributions by Category', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout(rect=[0, 0, 1, 0.96])
plt.savefig('figures/01_spending_distributions.png', dpi=300)
plt.close()

# 2. Correlation Heatmap
plt.figure(figsize=(9, 7))
corr = df[features].corr()
mask = np.triu(np.ones_like(corr, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(corr, annot=True, fmt='.2f', cmap=cmap, vmin=-0.2, vmax=1.0, 
            square=True, linewidths=1.2, cbar_kws={"shrink": .8},
            annot_kws={"size": 11, "weight": "bold"})
plt.title('Feature Correlation Heatmap (Product Categories)', fontsize=14, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig('figures/02_correlation_heatmap.png', dpi=300)
plt.close()

# Scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 3. Optimal K: Elbow Method & Silhouette Scores
inertias = []
sil_scores = []
k_range = range(2, 11)

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(X_scaled)
    inertias.append(km.inertia_)
    sil_scores.append(silhouette_score(X_scaled, labels))

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

# Elbow Curve
k_range_full = range(1, 11)
inertias_full = []
for k in k_range_full:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_scaled)
    inertias_full.append(km.inertia_)

ax1.plot(k_range_full, inertias_full, marker='o', markersize=8, color='#1f77b4', linewidth=2.2, label='Inertia')
ax1.axvline(x=3, color='#d62728', linestyle='--', linewidth=1.5, label='Elbow Point (k=3)')
ax1.set_title('Elbow Method: Inertia vs. Number of Clusters', fontsize=13, fontweight='bold')
ax1.set_xlabel('Number of Clusters (K)', fontsize=11)
ax1.set_ylabel('Within-Cluster Sum of Squares (Inertia)', fontsize=11)
ax1.set_xticks(k_range_full)
ax1.grid(True, linestyle='--', alpha=0.6)
ax1.legend(frameon=True)

# Silhouette Score
ax2.plot(k_range, sil_scores, marker='s', markersize=8, color='#2ca02c', linewidth=2.2, label='Silhouette Score')
ax2.axvline(x=3, color='#d62728', linestyle='--', linewidth=1.5, label='Selected k=3 (Score: 0.55)')
ax2.set_title('Silhouette Analysis across Cluster Numbers', fontsize=13, fontweight='bold')
ax2.set_xlabel('Number of Clusters (K)', fontsize=11)
ax2.set_ylabel('Silhouette Coefficient', fontsize=11)
ax2.set_xticks(k_range)
ax2.grid(True, linestyle='--', alpha=0.6)
ax2.legend(frameon=True)

plt.suptitle('Model Evaluation: Identifying Optimal Number of Customer Segments', fontsize=15, fontweight='bold', y=1.03)
plt.tight_layout()
plt.savefig('figures/03_elbow_silhouette_analysis.png', dpi=300, bbox_inches='tight')
plt.close()

# Fit K-Means with k=3
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# PCA for 2D visualization
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)
df['PCA1'] = X_pca[:, 0]
df['PCA2'] = X_pca[:, 1]
centroids_pca = pca.transform(kmeans.cluster_centers_)

var_ratio = pca.explained_variance_ratio_

# 4. PCA Clusters 2D Scatter Plot
plt.figure(figsize=(11, 7))
colors = ['#2b5c8f', '#e6550d', '#2ca02c']
cluster_names = {
    0: 'Cluster 0: Retailers & Grocery Wholesalers (N=45)',
    1: 'Cluster 1: Standard Hospitality & Horeca (N=393)',
    2: 'Cluster 2: High-Volume Bulk Buyers (N=2)'
}

for c_idx in sorted(df['Cluster'].unique()):
    sub = df[df['Cluster'] == c_idx]
    plt.scatter(sub['PCA1'], sub['PCA2'], 
                label=cluster_names[c_idx], 
                s=90, alpha=0.75, edgecolors='w', linewidth=0.6)

plt.scatter(centroids_pca[:, 0], centroids_pca[:, 1], 
            s=250, c='red', marker='X', edgecolors='black', linewidth=1.5, 
            label='Cluster Centroids', zorder=10)

plt.title('Customer Segments Visualized via PCA (Principal Components 1 & 2)', fontsize=14, fontweight='bold', pad=12)
plt.xlabel(f'Principal Component 1 ({var_ratio[0]*100:.1f}% Variance Explained)', fontsize=11)
plt.ylabel(f'Principal Component 2 ({var_ratio[1]*100:.1f}% Variance Explained)', fontsize=11)
plt.legend(frameon=True, fontsize=10, loc='best')
plt.grid(True, linestyle='--', alpha=0.6)
plt.tight_layout()
plt.savefig('figures/04_pca_clusters_2d.png', dpi=300)
plt.close()

# 5. Cluster Spending Comparison Bar Chart
cluster_summary = df.groupby('Cluster')[features].mean()
fig, ax = plt.subplots(figsize=(13, 7))

x = np.arange(len(features))
width = 0.26

rects0 = ax.bar(x - width, cluster_summary.loc[0], width, label='Cluster 0 (Retail/Grocery)', color='#3182bd', edgecolor='black', alpha=0.9)
rects1 = ax.bar(x, cluster_summary.loc[1], width, label='Cluster 1 (Horeca/Standard)', color='#74c476', edgecolor='black', alpha=0.9)
rects2 = ax.bar(x + width, cluster_summary.loc[2], width, label='Cluster 2 (Bulk Mega-Buyers)', color='#fd8d3c', edgecolor='black', alpha=0.9)

ax.set_ylabel('Average Annual Spending (m.u.)', fontsize=12, fontweight='bold')
ax.set_title('Mean Annual Spending per Product Category Across Customer Segments', fontsize=14, fontweight='bold', pad=15)
ax.set_xticks(x)
ax.set_xticklabels(features, fontsize=11, fontweight='bold')
ax.legend(frameon=True, fontsize=11)
ax.grid(True, axis='y', linestyle='--', alpha=0.6)

# Format y axis with commas
ax.get_yaxis().set_major_formatter(plt.FuncFormatter(lambda val, loc: "{:,}".format(int(val))))

plt.tight_layout()
plt.savefig('figures/05_cluster_spending_comparison.png', dpi=300)
plt.close()

# 6. Channel & Region Breakdown per Cluster
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Channel mapping: 1 -> Horeca, 2 -> Retail
df['Channel_Name'] = df['Channel'].map({1: 'Horeca (Hotel/Rest/Café)', 2: 'Retail'})
channel_ct = pd.crosstab(df['Cluster'], df['Channel_Name'], normalize='index') * 100

channel_ct.plot(kind='bar', stacked=True, ax=ax1, color=['#1f77b4', '#ff7f0e'], edgecolor='black', alpha=0.85)
ax1.set_title('Sales Channel Composition by Cluster (%)', fontsize=12, fontweight='bold')
ax1.set_xlabel('Customer Cluster', fontsize=11)
ax1.set_ylabel('Percentage (%)', fontsize=11)
ax1.set_xticklabels(['Cluster 0\n(Retail Wholesalers)', 'Cluster 1\n(Standard Horeca)', 'Cluster 2\n(Bulk Buyers)'], rotation=0)
ax1.legend(title='Channel', frameon=True)
ax1.grid(True, axis='y', linestyle='--', alpha=0.5)

# Region mapping: 1 -> Lisbon, 2 -> Oporto, 3 -> Other Region
df['Region_Name'] = df['Region'].map({1: 'Lisbon', 2: 'Oporto', 3: 'Other'})
region_ct = pd.crosstab(df['Cluster'], df['Region_Name'], normalize='index') * 100

region_ct.plot(kind='bar', stacked=True, ax=ax2, color=['#2ca02c', '#d62728', '#9467bd'], edgecolor='black', alpha=0.85)
ax2.set_title('Geographic Region Composition by Cluster (%)', fontsize=12, fontweight='bold')
ax2.set_xlabel('Customer Cluster', fontsize=11)
ax2.set_ylabel('Percentage (%)', fontsize=11)
ax2.set_xticklabels(['Cluster 0\n(Retail Wholesalers)', 'Cluster 1\n(Standard Horeca)', 'Cluster 2\n(Bulk Buyers)'], rotation=0)
ax2.legend(title='Region', frameon=True)
ax2.grid(True, axis='y', linestyle='--', alpha=0.5)

plt.suptitle('Demographic Distribution of Clusters across Channel and Region', fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('figures/06_channel_region_breakdown.png', dpi=300, bbox_inches='tight')
plt.close()

print("All 6 figures generated successfully in figures/")
