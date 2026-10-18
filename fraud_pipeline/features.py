"""Shared custom preprocessing for training and online inference."""
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, RobustScaler

REQUIRED = ["transaction_time", "amount", "lat", "lon", "merchant_lat", "merchant_lon",
            "population_city", "cat_id", "gender", "us_state"]
NUMERIC = ["log_amount", "distance_km", "log_population", "hour_sin", "hour_cos",
           "weekday_sin", "weekday_cos", "weekend", "night_amount"]
CATEGORICAL = ["cat_id", "gender", "us_state"]


def engineer(frame):
    """Read raw transactions; return fixed features without IDs, labels or names.

    Deterministic per row, without fitting on a test month or future transactions.
    """
    missing = set(REQUIRED) - set(frame.columns)
    if missing:
        raise ValueError(f"Missing input fields: {sorted(missing)}")
    t = pd.to_datetime(frame.transaction_time, errors="raise", utc=True)
    out = pd.DataFrame(index=frame.index)
    amount = pd.to_numeric(frame.amount, errors="coerce")
    out["log_amount"] = np.log1p(amount.clip(lower=0))
    lat = np.deg2rad(pd.to_numeric(frame.lat, errors="coerce"))
    lat2 = np.deg2rad(pd.to_numeric(frame.merchant_lat, errors="coerce"))
    lon = np.deg2rad(pd.to_numeric(frame.lon, errors="coerce"))
    lon2 = np.deg2rad(pd.to_numeric(frame.merchant_lon, errors="coerce"))
    # Local equirectangular distance; separate from the seminar's haversine features.
    dlon = (lon2 - lon + np.pi) % (2 * np.pi) - np.pi
    out["distance_km"] = 6371 * np.sqrt((lat2-lat)**2 + (dlon*np.cos((lat+lat2)/2))**2)
    out["log_population"] = np.log1p(pd.to_numeric(frame.population_city, errors="coerce").clip(lower=0))
    hour = t.dt.hour + t.dt.minute / 60
    out["hour_sin"], out["hour_cos"] = np.sin(2*np.pi*hour/24), np.cos(2*np.pi*hour/24)
    out["weekday_sin"] = np.sin(2*np.pi*t.dt.dayofweek/7)
    out["weekday_cos"] = np.cos(2*np.pi*t.dt.dayofweek/7)
    out["weekend"] = (t.dt.dayofweek >= 5).astype(int)
    out["night_amount"] = out.log_amount * (t.dt.hour < 6)
    for column in CATEGORICAL:
        out[column] = frame[column].fillna("unknown").astype(str)
    return out.replace([np.inf, -np.inf], np.nan)


def build_model():
    numeric = Pipeline([("impute", SimpleImputer(strategy="median", add_indicator=True)),
                        ("scale", RobustScaler())])
    transforms = ColumnTransformer([
        ("numeric", numeric, NUMERIC),
        ("category", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL)])
    return Pipeline([("features", FunctionTransformer(engineer)), ("transform", transforms),
                     ("classifier", LogisticRegression(C=0.5, class_weight="balanced",
                                                       max_iter=500, random_state=42))])
