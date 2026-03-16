import numpy as np
import pandas as pd
from src.data_loader import zscore_matrix


# divide and conquer
# split on the feature with the biggest variance
def split_cluster(X_cluster, original_indices):
    if len(X_cluster) <= 1:
        return (
            X_cluster,
            original_indices
        ), (
            np.empty((0, X_cluster.shape[1])),
            np.array([], dtype=int)
        )

    variances = np.var(X_cluster, axis=0)
    split_dim = np.argmax(variances)

    sort_order = np.argsort(X_cluster[:, split_dim])
    X_sorted = X_cluster[sort_order]
    idx_sorted = original_indices[sort_order]

    # split at the median so the clusters stay balanced
    mid = len(X_sorted) // 2

    left_cluster = (X_sorted[:mid], idx_sorted[:mid])
    right_cluster = (X_sorted[mid:], idx_sorted[mid:])

    return left_cluster, right_cluster


# divide and conquer
# keep splitting until we get k clusters
def recursive_topdown_clustering(X, k):
    clusters = [(X, np.arange(len(X)))]

    while len(clusters) < k:
        split_scores = []

        for X_cluster, idx_cluster in clusters:
            if len(X_cluster) <= 1:
                split_scores.append(-1)
            else:
                score = np.sum(np.var(X_cluster, axis=0)) * len(X_cluster)
                split_scores.append(score)

        cluster_to_split_idx = int(np.argmax(split_scores))
        X_cluster, idx_cluster = clusters.pop(cluster_to_split_idx)

        if len(X_cluster) <= 1:
            clusters.append((X_cluster, idx_cluster))
            break

        left_cluster, right_cluster = split_cluster(X_cluster, idx_cluster)
        clusters.append(left_cluster)
        clusters.append(right_cluster)

    return clusters


def analyze_task2(df, sensor_cols, num_clusters):
    X = df[sensor_cols].to_numpy(dtype=float)
    X = zscore_matrix(X)

    clusters = recursive_topdown_clustering(X, num_clusters)
    cluster_assignments = np.full(len(df), -1, dtype=int)

    summary_rows = []

    for cluster_id, (cluster_points, original_indices) in enumerate(clusters):
        cluster_assignments[original_indices] = cluster_id

        cluster_labels = df.iloc[original_indices]["rul_category"]
        counts = cluster_labels.value_counts()

        majority_class = counts.idxmax()
        majority_count = counts.max()

        summary_row = {
            "cluster_id": cluster_id,
            "cluster_size": len(original_indices),
            "majority_class": majority_class,
            "majority_class_count": majority_count,
            "majority_class_ratio": majority_count / len(original_indices)
        }

        for class_name in [
            "Extremely Low RUL",
            "Moderately Low RUL",
            "Moderately High RUL",
            "Extremely High RUL"
        ]:
            summary_row[class_name] = counts.get(class_name, 0)

        summary_rows.append(summary_row)

    df_clusters = df.copy()
    df_clusters["cluster_id"] = cluster_assignments

    summary_df = pd.DataFrame(summary_rows).sort_values(by="cluster_id")

    return summary_df, df_clusters