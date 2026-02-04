# ==============================
# FED MACRO PROBABILITY MODEL
# ==============================

# ---------- CORE ----------
import pandas as pd
import numpy as np

# ---------- DATA ----------
from pandas_datareader import data as pdr
from datetime import datetime

# ---------- VISUAL ----------
import matplotlib.pyplot as plt

# ---------- MODEL ----------
from sklearn.model_selection import TimeSeriesSplit
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report


plt.style.use("seaborn-v0_8-darkgrid")


# ==============================
# 1. DATA EXTRACTION (FRED)
# ==============================

start = datetime(2000, 1, 1)
end = datetime.today()

series = {
    "CPI": "CPIAUCSL",
    "CORE_CPI": "CPILFESL",
    "UNEMP": "UNRATE",
    "FED_FUNDS": "FEDFUNDS",
    "DGS2": "DGS2",
    "DGS10": "DGS10"
}

df = pd.DataFrame()

for name, code in series.items():
    df[name] = pdr.DataReader(code, "fred", start, end)

# Monthly frequency
df = df.resample("M").last()


# ==============================
# 2. FEATURE ENGINEERING
# ==============================

# Dates
df["DATE"] = df.index
df["YEAR"] = df.index.year

# Inflation
df["CPI_YOY"] = df["CPI"].pct_change(12) * 100
df["CORE_CPI_YOY"] = df["CORE_CPI"].pct_change(12) * 100

# Trends
df["CPI_3M_TREND"] = df["CPI_YOY"] - df["CPI_YOY"].shift(3)

# Labor
df["UNEMP_CHANGE"] = df["UNEMP"] - df["UNEMP"].shift(3)

# Yield curve
df["YC_SLOPE"] = df["DGS10"] - df["DGS2"]

# Real rate
df["REAL_RATE"] = df["FED_FUNDS"] - df["CPI_YOY"]


# ==============================
# 3. REGIME DETECTION
# ==============================

def detect_regime(row):

    if row["CPI_YOY"] > 3:
        return "INFLATION"

    elif row["UNEMP_CHANGE"] > 0.3 and row["YC_SLOPE"] < 0:
        return "RECESSION"

    else:
        return "NORMAL"


df["REGIME"] = df.apply(detect_regime, axis=1)

regime_map = {
    "NORMAL": 0,
    "INFLATION": 1,
    "RECESSION": 2
}

df["REGIME_ENC"] = df["REGIME"].map(regime_map)


# ==============================
# 4. LABELING FED DECISIONS
# ==============================

df["RATE_CHANGE"] = df["FED_FUNDS"].diff()


def label_decision(x):

    if x > 0:
        return "HIKE"
    elif x < 0:
        return "CUT"
    else:
        return "PAUSE"


df["RATE_DECISION"] = df["RATE_CHANGE"].apply(label_decision)


# ==============================
# 5. CLEAN DATA
# ==============================

df = df.dropna(subset=[
    "CPI_YOY",
    "CORE_CPI_YOY",
    "CPI_3M_TREND",
    "UNEMP",
    "UNEMP_CHANGE",
    "REAL_RATE",
    "YC_SLOPE",
    "REGIME_ENC",
    "RATE_DECISION"
])

decision_map = {
    "CUT": 0,
    "PAUSE": 1,
    "HIKE": 2
}

df["DECISION_ENC"] = df["RATE_DECISION"].map(decision_map)


# ==============================
# 6. MODELING
# ==============================

features = [
    "CPI_YOY",
    "CPI_3M_TREND",
    "CORE_CPI_YOY",
    "UNEMP",
    "UNEMP_CHANGE",
    "REAL_RATE",
    "YC_SLOPE",
    "REGIME_ENC"
]

X = df[features]
y = df["DECISION_ENC"]

tscv = TimeSeriesSplit(n_splits=5)

# Note: some scikit-learn installs may not accept the `multi_class` keyword.
# Removed to ensure compatibility; solver='lbfgs' handles multiclass training.
model = LogisticRegression(
    class_weight="balanced",   # FIX BIAS
    max_iter=1000,
    solver="lbfgs"
)


# ==============================
# 7. TRAIN & VALIDATE
# ==============================

for train_idx, test_idx in tscv.split(X):

    X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
    y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)


print("\nMODEL PERFORMANCE\n")
print(classification_report(y_test, y_pred))


# ==============================
# 8. CURRENT PROBABILITIES
# ==============================

latest = X.iloc[-1:].copy()

probs = model.predict_proba(latest)[0]

prob_table = pd.DataFrame({
    "Decision": ["CUT", "PAUSE", "HIKE"],
    "Probability": probs
}).sort_values("Probability", ascending=False)

print("\nCURRENT FED PROBABILITIES\n")
print(prob_table)


# ==============================
# 9. VISUALIZATION (MATPLOTLIB)
# ==============================

# CPI vs FED RATE
plt.figure(figsize=(14,6))

plt.plot(df["DATE"], df["CPI_YOY"], label="CPI YoY")
plt.plot(df["DATE"], df["FED_FUNDS"], label="Fed Funds")

plt.axhline(2, linestyle="--", linewidth=1, label="Inflation Target")

plt.title("Inflation vs Fed Policy Rate")
plt.xlabel("Year")
plt.ylabel("Percent")
plt.legend()
plt.show()


# Unemployment + Recession
plt.figure(figsize=(14,6))

plt.plot(df["DATE"], df["UNEMP"], label="Unemployment")

plt.plot(
    df[df["REGIME"]=="RECESSION"]["DATE"],
    df[df["REGIME"]=="RECESSION"]["UNEMP"],
    "ro",
    label="Recession Signal"
)

plt.title("Unemployment & Recession Regime")
plt.xlabel("Year")
plt.ylabel("Rate")
plt.legend()
plt.show()


# Yield Curve
plt.figure(figsize=(14,6))

plt.plot(df["DATE"], df["YC_SLOPE"], label="10Y - 2Y Spread")

plt.axhline(0, linestyle="--", linewidth=1)

plt.title("Yield Curve Slope")
plt.xlabel("Year")
plt.ylabel("Spread")
plt.legend()
plt.show()


# Regimes over time
colors = {
    "NORMAL": "green",
    "INFLATION": "orange",
    "RECESSION": "red"
}

plt.figure(figsize=(14,6))

for r in df["REGIME"].unique():

    subset = df[df["REGIME"] == r]

    plt.scatter(
        subset["DATE"],
        subset["CPI_YOY"],
        color=colors[r],
        label=r,
        s=15
    )

plt.title("Economic Regimes (CPI Based)")
plt.xlabel("Year")
plt.ylabel("CPI YoY")
plt.legend()
plt.show()


# ==============================
# 10. EXPORT
# ==============================

df_export = df[features + ["RATE_DECISION", "REGIME", "DATE"]]

df_export.to_excel("fed_macro_model_v2.xlsx")
prob_table.to_excel("fed_decision_probabilities_v2.xlsx")


print("\nFILES EXPORTED SUCCESSFULLY")
