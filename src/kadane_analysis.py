import pandas as pd
import numpy as np


# maximum subarray
def kadane(arr):
    max_sum = -float("inf")
    current_sum = 0

    start = 0
    best_start = 0
    best_end = 0

    for i, value in enumerate(arr):
        if current_sum <= 0:
            current_sum = value
            start = i
        else:
            current_sum += value

        if current_sum > max_sum:
            max_sum = current_sum
            best_start = start
            best_end = i

    return max_sum, best_start, best_end


def analyze_task3(df, sensor_cols):
    rows = []

    for sensor in sensor_cols:
        signal = df[sensor].to_numpy()

        # build transformed signal from absolute first differences
        d = np.abs(np.diff(signal))
        d_mean = np.mean(d)
        x = d - d_mean

        # run kadane to get the strongest deviation interval
        max_sum, start_idx, end_idx = kadane(x)

        interval_start = start_idx + 1
        interval_end = end_idx + 1

        interval_labels = df.iloc[interval_start:interval_end + 1]["rul_category"]

        if len(interval_labels) == 0:
            dominant_class = "N/A"
            low_rul_ratio = 0.0
        else:
            dominant_class = interval_labels.value_counts().idxmax()
            low_count = (
                (interval_labels == "Extremely Low RUL").sum() +
                (interval_labels == "Moderately Low RUL").sum()
            )
            low_rul_ratio = low_count / len(interval_labels)

        rows.append({
            "sensor": sensor,
            "max_subarray_sum": max_sum,
            "start_index": interval_start,
            "end_index": interval_end,
            "interval_length": interval_end - interval_start + 1,
            "dominant_rul_category_in_interval": dominant_class,
            "low_rul_ratio_in_interval": low_rul_ratio
        })

    results_df = pd.DataFrame(rows).sort_values(by="max_subarray_sum", ascending=False)

    return results_df