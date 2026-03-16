import numpy as np
import pandas as pd


# divide and conquer
# keep splitting until the segment is stable
def segment_signal_recursive(signal, left, right, variance_threshold, min_len, segments):
    segment = signal[left:right]
    seg_len = right - left

    if seg_len <= min_len:
        segments.append((left, right))
        return

    variance = np.var(segment)

    if variance > variance_threshold:
        mid = (left + right) // 2
        if mid == left or mid == right:
            segments.append((left, right))
            return

        segment_signal_recursive(signal, left, mid, variance_threshold, min_len, segments)
        segment_signal_recursive(signal, mid, right, variance_threshold, min_len, segments)
    else:
        segments.append((left, right))


# use global variance to set the split threshold
def perform_segmentation(signal, threshold_factor=0.75, min_len=64):
    global_variance = np.var(signal)
    variance_threshold = threshold_factor * global_variance

    segments = []
    segment_signal_recursive(
        signal=signal,
        left=0,
        right=len(signal),
        variance_threshold=variance_threshold,
        min_len=min_len,
        segments=segments
    )
    return segments, variance_threshold


def analyze_task1(df, sensor_cols, random_seed, threshold_factor, min_segment_length):
    np.random.seed(random_seed)

    # pick 10 sensors from the full list
    if len(sensor_cols) >= 10:
        step = max(1, len(sensor_cols) // 10)
        selected_sensors = sensor_cols[::step][:10]
    else:
        selected_sensors = sensor_cols

    task1_rows = []

    for sensor in selected_sensors:
        signal = df[sensor].to_numpy()

        segments, threshold = perform_segmentation(
            signal,
            threshold_factor=threshold_factor,
            min_len=min_segment_length
        )

        # segmentation complexity = number of final segments
        complexity_score = len(segments)

        dominant_categories = []
        for start, end in segments:
            segment_labels = df.iloc[start:end]["rul_category"]
            dominant_label = segment_labels.value_counts().idxmax()
            dominant_categories.append(dominant_label)

        overall_dominant_label = pd.Series(dominant_categories).value_counts().idxmax()

        task1_rows.append({
            "sensor": sensor,
            "variance_threshold": threshold,
            "num_segments": complexity_score,
            "avg_segment_length": len(signal) / complexity_score,
            "dominant_segment_rul_category": overall_dominant_label
        })

    task1_results = pd.DataFrame(task1_rows)
    task1_results = task1_results.sort_values(by="num_segments", ascending=False)

    return task1_results, selected_sensors