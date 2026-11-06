"""
PSP301 prototype: anomaly screening using real, publicly released Amazon purchase histories.

Data source:
Berke et al., "Open e-commerce 1.0, five years of crowdsourced U.S. Amazon purchase
histories with user demographics," Scientific Data, 2024.
https://doi.org/10.1038/s41597-024-03329-6
Harvard Dataverse DOI: https://doi.org/10.7910/DVN/YGLYDY

IMPORTANT:
- Download amazon-purchases.csv from the public Harvard Dataverse dataset and place it
  beside this script before running.
- The dataset consists of purchase histories voluntarily shared by 5,027 U.S. Amazon
  users for research, covering 2018 to October 2022. It is not Amazon's complete internal
  transaction database.
- It contains no confirmed fraud labels, login attempts, payout changes, or refund-abuse
  outcomes. This script performs unsupervised anomaly screening only; it does NOT identify
  fraud or calculate precision/recall/ROC-AUC.
- It reads only the columns needed for this demo and does not export product titles,
  ASINs, or shipping states. Output IDs are re-pseudonymised with a random per-run key.
- An anomaly is a prompt for human review, not proof of wrongdoing.
Requirements: Python 3.10+, pandas, numpy, scikit-learn.
Run: python amazon_real_purchase_anomaly_detection.py
"""

from pathlib import Path
import hashlib
import hmac
import secrets
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

DATA_FILE = Path("amazon-purchases.csv")
OUTPUT_FILE = Path("amazon_monthly_anomaly_flags.csv")
SUMMARY_FILE = Path("amazon_anomaly_run_summary.txt")
CONTAMINATION = 0.01  # chosen review-volume setting, NOT an estimated fraud rate
RANDOM_STATE = 42


def find_columns(path: Path):
    """Find required columns case-insensitively, tolerating minor header whitespace."""
    header = pd.read_csv(path, nrows=0).columns
    normalised = {str(col).strip().casefold(): col for col in header}
    aliases = {
        "order date": ("order date", "order_date", "purchase date"),
        "purchase price per unit": ("purchase price per unit", "purchase_price_per_unit", "unit price"),
        "quantity": ("quantity", "qty"),
        "category": ("category",),
        "survey responseid": ("survey responseid", "survey_responseid", "participant id"),
    }
    resolved = {}
    for target, names in aliases.items():
        found = next((normalised[name] for name in names if name in normalised), None)
        if found is None:
            raise ValueError(
                f"Missing required column for '{target}'. Found headers: {list(header)}"
            )
        resolved[target] = found
    return resolved


def load_and_prepare(path: Path):
    cols = find_columns(path)
    # Only load the minimum fields needed for this demonstration.
    df = pd.read_csv(
        path,
        usecols=list(cols.values()),
        low_memory=False,
    )
    reverse = {original: standard for standard, original in cols.items()}
    df = df.rename(columns=reverse)

    df["order date"] = pd.to_datetime(df["order date"], errors="coerce")
    df["purchase price per unit"] = pd.to_numeric(
        df["purchase price per unit"], errors="coerce"
    )
    df["quantity"] = pd.to_numeric(df["quantity"], errors="coerce")
    df["category"] = df["category"].astype("string").str.strip()
    df.loc[df["category"].isin(["", "nan", "None"]), "category"] = pd.NA
    df["survey responseid"] = df["survey responseid"].astype("string").str.strip()

    before = len(df)
    df = df.dropna(
        subset=["order date", "purchase price per unit", "quantity", "survey responseid"]
    )
    df = df[
        (df["purchase price per unit"] >= 0)
        & (df["quantity"] > 0)
        & (df["survey responseid"] != "")
    ].copy()

    # Keep the published dataset's documented period; malformed/out-of-period rows are excluded.
    df = df[
        (df["order date"] >= pd.Timestamp("2018-01-01"))
        & (df["order date"] <= pd.Timestamp("2022-10-31"))
    ].copy()

    if df.empty:
        raise ValueError("No usable purchase records remain after cleaning.")

    df["month"] = df["order date"].dt.to_period("M").astype(str)
    df["line_spend"] = df["purchase price per unit"] * df["quantity"]

    # Re-pseudonymise IDs with a random secret that is not saved or exported.
    # This prevents the output token from being a stable public identifier across runs.
    salt = secrets.token_bytes(32)
    unique_ids = df["survey responseid"].dropna().unique()
    token_map = {
        str(uid): hmac.new(salt, str(uid).encode("utf-8"), hashlib.sha256).hexdigest()[:16]
        for uid in unique_ids
    }
    df["user_token"] = df["survey responseid"].map(token_map)

    # Drop the source participant identifier as soon as it is no longer needed.
    df = df.drop(columns=["survey responseid", "order date"])

    raw_rows = before
    clean_rows = len(df)
    return df, raw_rows, clean_rows, len(unique_ids)


def build_monthly_features(df: pd.DataFrame):
    monthly = (
        df.groupby(["user_token", "month"], as_index=False)
        .agg(
            monthly_spend=("line_spend", "sum"),
            item_units=("quantity", "sum"),
            purchase_lines=("quantity", "size"),
            mean_unit_price=("purchase price per unit", "mean"),
            distinct_categories=("category", "nunique"),
        )
    )
    month_num = pd.PeriodIndex(monthly["month"], freq="M").month.astype(int)
    monthly["month_sin"] = np.sin(2 * np.pi * month_num / 12)
    monthly["month_cos"] = np.cos(2 * np.pi * month_num / 12)
    return monthly


def add_explanation_flags(monthly: pd.DataFrame):
    # These are descriptive comparison points, not causal explanations.
    metrics = [
        "monthly_spend",
        "item_units",
        "purchase_lines",
        "mean_unit_price",
        "distinct_categories",
    ]
    quantiles = {col: monthly[col].quantile(0.99) for col in metrics}

    def explain(row):
        reasons = []
        for col, threshold in quantiles.items():
            if pd.notna(row[col]) and row[col] > threshold:
                label = {
                    "monthly_spend": "monthly spend above sample 99th percentile",
                    "item_units": "item quantity above sample 99th percentile",
                    "purchase_lines": "purchase-line count above sample 99th percentile",
                    "mean_unit_price": "mean unit price above sample 99th percentile",
                    "distinct_categories": "category diversity above sample 99th percentile",
                }[col]
                reasons.append(label)
        return "; ".join(reasons) if reasons else "unusual combination of monthly features"

    monthly["review_reason"] = monthly.apply(explain, axis=1)
    return monthly


def run():
    if not DATA_FILE.exists():
        print(
            "Dataset file not found.\n\n"
            "1. Open the official Harvard Dataverse dataset:\n"
            "   https://doi.org/10.7910/DVN/YGLYDY\n"
            "2. Download the public file named 'amazon-purchases.csv' (about 299 MB).\n"
            "3. Put it in the same folder as this script, then run the script again.\n\n"
            "Do not replace the real dataset with fabricated records. The GitHub demo sample "
            "is only a small subset and should not be treated as a representative full dataset."
        )
        sys.exit(2)

    data, raw_rows, clean_rows, user_count = load_and_prepare(DATA_FILE)
    monthly = build_monthly_features(data)
    if len(monthly) < 30:
        raise ValueError(
            f"Only {len(monthly)} user-month observations found; too few for this demo."
        )

    feature_cols = [
        "monthly_spend",
        "item_units",
        "purchase_lines",
        "mean_unit_price",
        "distinct_categories",
        "month_sin",
        "month_cos",
    ]
    X = monthly[feature_cols].copy()
    for col in ["monthly_spend", "item_units", "purchase_lines", "mean_unit_price"]:
        X[col] = np.log1p(X[col].clip(lower=0))
    X = X.replace([np.inf, -np.inf], np.nan).fillna(0)

    model = IsolationForest(
        n_estimators=200,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )
    labels = model.fit_predict(X)
    monthly["anomaly_flag"] = labels == -1
    monthly["anomaly_score"] = -model.score_samples(X)
    monthly = add_explanation_flags(monthly)

    flagged = monthly[monthly["anomaly_flag"]].copy()
    flagged = flagged.sort_values("anomaly_score", ascending=False)

    # Only aggregated monthly features and per-run pseudonymous tokens are exported.
    output_cols = [
        "user_token",
        "month",
        "monthly_spend",
        "item_units",
        "purchase_lines",
        "mean_unit_price",
        "distinct_categories",
        "anomaly_score",
        "review_reason",
    ]
    flagged[output_cols].to_csv(OUTPUT_FILE, index=False)

    summary = [
        "Open e-commerce 1.0 — Amazon purchase-history anomaly-screening run",
        f"Input file: {DATA_FILE.name}",
        f"Rows read: {raw_rows:,}",
        f"Usable purchase records after cleaning: {clean_rows:,}",
        f"Distinct source participants in usable records: {user_count:,}",
        f"User-month observations: {len(monthly):,}",
        f"Anomaly candidates flagged: {len(flagged):,}",
        f"Configured contamination/review-volume setting: {CONTAMINATION:.1%}",
        "",
        "Interpretation and limits:",
        "- This is unsupervised anomaly screening, not a fraud classifier.",
        "- The dataset has no confirmed fraud labels or login/payout/refund-abuse signals.",
        "- Precision, recall, F1 and ROC-AUC cannot be calculated from these data.",
        "- A flagged user-month is not proof of fraud; use human review and verification.",
        "- Output omits product titles, ASINs, categories per purchase, and shipping states.",
        "- Participant tokens are re-pseudonymised with a random per-run key.",
    ]
    SUMMARY_FILE.write_text("\n".join(summary) + "\n", encoding="utf-8")

    print("\n".join(summary))
    print(f"\nSaved flagged monthly summaries to: {OUTPUT_FILE.resolve()}")
    print(f"Saved run notes to: {SUMMARY_FILE.resolve()}")


if __name__ == "__main__":
    run()
