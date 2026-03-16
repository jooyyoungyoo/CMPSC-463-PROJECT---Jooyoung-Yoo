import numpy as np
import pandas as pd


def point_distance(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


# closest pair
def brute_force_closest_pair(points):
    best_dist = float("inf")
    best_pair = None

    n = len(points)
    for i in range(n):
        for j in range(i + 1, n):
            dist = point_distance(points[i], points[j])
            if dist < best_dist:
                best_dist = dist
                best_pair = (points[i], points[j])

    return best_pair, best_dist


# closest pair
# check points near the middle line
def closest_split_pair(px, py, delta, best_pair):
    mid_x = px[len(px) // 2][0]
    sy = [p for p in py if mid_x - delta <= p[0] <= mid_x + delta]

    best = delta
    split_pair = best_pair

    for i in range(len(sy)):
        for j in range(i + 1, min(i + 8, len(sy))):
            p, q = sy[i], sy[j]
            dist = point_distance(p, q)
            if dist < best:
                best = dist
                split_pair = (p, q)

    return split_pair, best


# closest pair
# base case: check all pairs directly
def closest_pair_recursive(px, py):
    n = len(px)

    if n <= 3:
        return brute_force_closest_pair(px)

    mid = n // 2
    qx = px[:mid]
    rx = px[mid:]

    left_ids = {p[2] for p in qx}
    qy = [p for p in py if p[2] in left_ids]
    ry = [p for p in py if p[2] not in left_ids]

    left_pair, dist_left = closest_pair_recursive(qx, qy)
    right_pair, dist_right = closest_pair_recursive(rx, ry)

    if dist_left <= dist_right:
        delta = dist_left
        best_pair = left_pair
    else:
        delta = dist_right
        best_pair = right_pair

    split_pair, split_dist = closest_split_pair(px, py, delta, best_pair)

    if split_dist < delta:
        return split_pair, split_dist
    return best_pair, delta


# closest pair
def closest_pair(points):
    px = sorted(points, key=lambda p: (p[0], p[1], p[2]))
    py = sorted(points, key=lambda p: (p[1], p[0], p[2]))
    return closest_pair_recursive(px, py)


# turn each time step into one 2D point
# x = first half of sensors, y = second half
def build_2d_points(df, sensor_cols):
    half = len(sensor_cols) // 2
    first_half = sensor_cols[:half]
    second_half = sensor_cols[half:]

    points = []
    for i in range(len(df)):
        x = df.loc[df.index[i], first_half].mean()
        y = df.loc[df.index[i], second_half].mean()
        points.append((x, y, i))

    return points


def analyze_closest_pair(df, sensor_cols):
    points = build_2d_points(df, sensor_cols)
    best_pair, best_dist = closest_pair(points)

    p1, p2 = best_pair
    idx1 = p1[2]
    idx2 = p2[2]

    result_df = pd.DataFrame([{
        "index_1": idx1,
        "index_2": idx2,
        "point_1_x": p1[0],
        "point_1_y": p1[1],
        "point_2_x": p2[0],
        "point_2_y": p2[1],
        "distance": best_dist,
        "rul_1": df.iloc[idx1]["rul"],
        "rul_2": df.iloc[idx2]["rul"],
        "rul_category_1": df.iloc[idx1]["rul_category"],
        "rul_category_2": df.iloc[idx2]["rul_category"]
    }])

    return result_df