"""Wholesale Customer Segmentation Pipeline.

This module provides an end-to-end unsupervised machine learning pipeline
to segment wholesale clients based on their annual spending patterns across
diverse product categories (Fresh, Milk, Grocery, Frozen, Detergents_Paper, Delicassen).
"""

import argparse
import os
import sys
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


def load_data(file_path: str = "data/Wholesale_customers_data.csv") -> pd.DataFrame:
    """Load wholesale customer dataset from local path or fallback URL."""
    if os.path.exists(file_path):
        print(f"[INFO] Loading dataset from local file: {file_path}")
        return pd.read_csv(file_path)
    
    fallback_url = (
        "https://raw.githubusercontent.com/alishermutalov/praktikum-datasets/"
        "refs/heads/praktikum/Wholesale_customers_data.csv"
    )
    print(f"[WARN] Local path {file_path} not found. Fetching from fallback URL: {fallback_url}")
    df = pd.read_csv(fallback_url)
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    df.to_csv(file_path, index=False)
    return df


def preprocess_data(df: pd.DataFrame, feature_cols=None):
    """Drop demographic metadata and standardize continuous spending features."""
    if feature_cols is None:
        feature_cols = ["Fresh", "Milk", "Grocery", "Frozen", "Detergents_Paper", "Delicassen"]
    
    X = df[feature_cols].copy()
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X, X_scaled, scaler, feature_cols


def evaluate_clusters(X_scaled: np.ndarray, max_k: int = 10):
    """Evaluate inertia and silhouette scores across a range of k values."""
    print("\n--- Cluster Evaluation (K-Means) ---")
    results = []
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=42, n_init=10)
        labels = km.fit_predict(X_scaled)
        sil = silhouette_score(X_scaled, labels)
        results.append({"k": k, "inertia": km.inertia_, "silhouette": sil})
        print(f"k = {k:2d} | Inertia: {km.inertia_:10.2f} | Silhouette Score: {sil:.4f}")
    return pd.DataFrame(results)


def train_kmeans(X_scaled: np.ndarray, n_clusters: int = 3, random_state: int = 42) -> KMeans:
    """Fit K-Means clustering model."""
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    kmeans.fit(X_scaled)
    return kmeans


def apply_pca(X_scaled: np.ndarray, n_components: int = 2):
    """Apply PCA for 2D visualization and dimensionality reduction."""
    pca = PCA(n_components=n_components, random_state=42)
    X_pca = pca.fit_transform(X_scaled)
    print(f"\n[INFO] PCA Explained Variance: {pca.explained_variance_ratio_}")
    print(f"[INFO] Cumulative Explained Variance: {np.sum(pca.explained_variance_ratio_):.2%}")
    return X_pca, pca


def generate_cluster_profiles(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Calculate mean and median spending per cluster with customer counts."""
    summary = df.groupby("Cluster")[feature_cols].mean().round(2)
    summary["Customer_Count"] = df["Cluster"].value_counts()
    summary["Pct_Total"] = (df["Cluster"].value_counts(normalize=True) * 100).round(2)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Wholesale Customer Segmentation with K-Means & PCA")
    parser.add_argument("--data", default="data/Wholesale_customers_data.csv", help="Path to input CSV data")
    parser.add_argument("--clusters", type=int, default=3, help="Number of clusters (default: 3)")
    parser.add_argument("--output", default="data/segmented_customers.csv", help="Path to export segmented CSV")
    args = parser.parse_args()

    # 1. Load Data
    df = load_data(args.data)
    print(f"[INFO] Successfully loaded dataset with {df.shape[0]} rows and {df.shape[1]} columns.")

    # 2. Preprocess & Scale
    X, X_scaled, scaler, feature_cols = preprocess_data(df)

    # 3. Cluster Evaluation
    evaluate_clusters(X_scaled, max_k=8)

    # 4. Train Final Model
    print(f"\n[INFO] Fitting K-Means with {args.clusters} clusters...")
    kmeans = train_kmeans(X_scaled, n_clusters=args.clusters)
    df["Cluster"] = kmeans.labels_

    # 5. Dimensionality Reduction (PCA)
    X_pca, pca = apply_pca(X_scaled, n_components=2)
    df["PCA1"] = X_pca[:, 0]
    df["PCA2"] = X_pca[:, 1]

    # 6. Profile Clusters
    profiles = generate_cluster_profiles(df, feature_cols)
    print("\n--- Cluster Spending Profiles (Mean Annual Spending in m.u.) ---")
    print(profiles)

    # 7. Channel Distribution
    if "Channel" in df.columns:
        print("\n--- Sales Channel Breakdown per Cluster (1 = Horeca, 2 = Retail) ---")
        channel_cross = pd.crosstab(df["Cluster"], df["Channel"])
        channel_cross.columns = ["Horeca", "Retail"]
        print(channel_cross)

    # 8. Save Output
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    df.to_csv(args.output, index=False)
    print(f"\n[SUCCESS] Segmented data exported to: {args.output}")


if __name__ == "__main__":
    main()
