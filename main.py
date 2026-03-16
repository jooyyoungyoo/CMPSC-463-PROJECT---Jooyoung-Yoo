import os
import numpy as np

from src.data_loader import load_and_prepare_data
from src.segmentation import perform_segmentation, analyze_task1
from src.clustering import recursive_topdown_clustering, analyze_task2
from src.closest_pair_analysis import closest_pair, analyze_closest_pair
from src.kadane_analysis import kadane, analyze_task3


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(BASE_DIR, "data", "rul_hrs.csv")

N_ROWS = 10000
MIN_SEGMENT_LENGTH = 64
SEGMENT_THRESHOLD_FACTOR = 0.75
NUM_CLUSTERS = 4
RANDOM_SEED = 42


# toy examples to check each algorithm on small inputs
def run_toy_examples():
    print("\n" + "=" * 60)
    print("TOY EXAMPLE VERIFICATION")
    print("=" * 60)

    toy_signal = np.array([1, 1, 1, 1, 10, 10, 10, 10], dtype=float)
    segments, threshold = perform_segmentation(
        toy_signal,
        threshold_factor=0.1,
        min_len=2
    )

    print("\nToy Example 1 - Segmentation")
    print("Signal:", toy_signal)
    print("Variance Threshold:", threshold)
    print("Segments:", segments)
    print("Expected behavior: signal should split into stable regions")

    toy_points_cluster = np.array([
        [1, 1],
        [1, 2],
        [2, 1],
        [8, 8],
        [8, 9],
        [9, 8]
    ], dtype=float)

    toy_clusters = recursive_topdown_clustering(toy_points_cluster, 2)

    print("\nToy Example 2 - Clustering")
    print("Points:")
    print(toy_points_cluster)
    for i, (cluster_points, original_indices) in enumerate(toy_clusters):
        print(f"Cluster {i}:")
        print(cluster_points)
    print("Expected behavior: points should separate into two natural groups")

    toy_points_cp = [
        (1.0, 1.0, 0),
        (2.0, 2.0, 1),
        (8.0, 8.0, 2),
        (8.0, 9.0, 3),
        (1.2, 1.1, 4)
    ]
    best_pair, best_dist = closest_pair(toy_points_cp)

    print("\nToy Example 3 - Closest Pair")
    print("Points:", toy_points_cp)
    print("Closest Pair:", best_pair)
    print("Distance:", best_dist)

    toy_array = np.array([-2, 3, 5, -1, 4, -10, 6], dtype=float)
    max_sum, start_idx, end_idx = kadane(toy_array)

    print("\nToy Example 4 - Kadane")
    print("Array:", toy_array)
    print("Max Sum:", max_sum)
    print("Start Index:", start_idx)
    print("End Index:", end_idx)
    print("Best Subarray:", toy_array[start_idx:end_idx + 1])
    print("Expected max subarray: [3, 5, -1, 4], sum = 11")


def main():
    run_toy_examples()

    print("Loading data...")
    df, sensor_cols, quantiles = load_and_prepare_data(DATA_FILE, N_ROWS)

    q10, q50, q90 = quantiles

    print("\nDataset loaded successfully.")
    print(f"Shape used: {df.shape}")
    print(f"Number of available sensor columns: {len(sensor_cols)}")
    print(f"Sensor columns: {sensor_cols}")
    print(f"Q10 = {q10:.4f}, Q50 = {q50:.4f}, Q90 = {q90:.4f}")

    print("\nRunning Task 1: Divide-and-Conquer Segmentation...")
    task1_results, selected_sensors = analyze_task1(
        df=df,
        sensor_cols=sensor_cols,
        random_seed=RANDOM_SEED,
        threshold_factor=SEGMENT_THRESHOLD_FACTOR,
        min_segment_length=MIN_SEGMENT_LENGTH
    )
    print("Task 1 complete.")
    print(task1_results)

    print("\nRunning Task 2: Divide-and-Conquer Clustering...")
    task2_summary, df_clusters = analyze_task2(
        df=df,
        sensor_cols=sensor_cols,
        num_clusters=NUM_CLUSTERS
    )
    print("Task 2 complete.")
    print(task2_summary)

    print("\nRunning Closest Pair Analysis...")
    closest_pair_results = analyze_closest_pair(
        df=df,
        sensor_cols=sensor_cols
    )
    print("Closest Pair complete.")
    print(closest_pair_results)

    print("\nRunning Task 3: Maximum Subarray (Kadane)...")
    task3_results = analyze_task3(
        df=df,
        sensor_cols=sensor_cols
    )
    print("Task 3 complete.")
    print(task3_results.head(10))

    print("\nAll tasks completed successfully.")
    print(f"Selected sensors for Task 1: {selected_sensors}")


if __name__ == "__main__":
    main()