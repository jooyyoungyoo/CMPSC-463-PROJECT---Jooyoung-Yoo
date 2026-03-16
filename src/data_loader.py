import pandas as pd


def get_sensor_columns(df):
    return sorted([col for col in df.columns if col.startswith("sensor_")])


def zscore_matrix(X):
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0
    return (X - means) / stds


def load_and_prepare_data(filepath, n_rows=10000):
    df = pd.read_csv(filepath)

    if "Unnamed: 0" in df.columns:
        df = df.drop(columns=["Unnamed: 0"])

    df = df.iloc[:n_rows].copy()

    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])

    sensor_cols = get_sensor_columns(df)

    df[sensor_cols] = df[sensor_cols].ffill().bfill()
    for col in sensor_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mean())

    q10 = df["rul"].quantile(0.10)
    q50 = df["rul"].quantile(0.50)
    q90 = df["rul"].quantile(0.90)

    def assign_rul_category(rul_value):
        if rul_value < q10:
            return "Extremely Low RUL"
        elif rul_value < q50:
            return "Moderately Low RUL"
        elif rul_value < q90:
            return "Moderately High RUL"
        else:
            return "Extremely High RUL"

    df["rul_category"] = df["rul"].apply(assign_rul_category)

    return df, sensor_cols, (q10, q50, q90)