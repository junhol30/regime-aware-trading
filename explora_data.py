# Junho Lee (utg2ue)
import numpy as np
import pandas as pd
import yfinance as yf


# --------------------------------------------------
# SETTINGS
# --------------------------------------------------

TICKER = "SPY"
START_DATE = "2010-01-01"
END_DATE = "2026-08-18"


# --------------------------------------------------
# DOWNLOAD DATA
# --------------------------------------------------

print("=" * 70)
print("REGIME-AWARE TRADING PROJECT")
print("=" * 70)

print(f"\nDownloading {TICKER} data...")

data = yf.download(
    TICKER,
    start=START_DATE,
    end=END_DATE,
    auto_adjust=True,
    progress=False
)

if data.empty:
    raise ValueError("No market data was downloaded.")

# yfinance can sometimes return MultiIndex columns.
if isinstance(data.columns, pd.MultiIndex):
    data.columns = data.columns.get_level_values(0)

print(f"Downloaded {len(data):,} trading days.")

print(f"First date: {data.index.min().date()}")
print(f"Last date:  {data.index.max().date()}")


# --------------------------------------------------
# CALCULATE RETURNS
# --------------------------------------------------

data["return"] = data["Close"].pct_change()

data["log_return"] = np.log(
    data["Close"] / data["Close"].shift(1)
)


# --------------------------------------------------
# CALCULATE VOLATILITY
# --------------------------------------------------

# 20-day rolling standard deviation of daily returns.
data["volatility_20d"] = (
    data["return"]
    .rolling(window=20)
    .std()
)

# Annualized volatility.
data["annualized_volatility_20d"] = (
    data["volatility_20d"] * np.sqrt(252)
)


# --------------------------------------------------
# CALCULATE MOMENTUM
# --------------------------------------------------

# Percentage change over approximately one trading month.
data["momentum_20d"] = data["Close"].pct_change(20)

# Longer-term momentum.
data["momentum_60d"] = data["Close"].pct_change(60)


# --------------------------------------------------
# MOVING AVERAGES
# --------------------------------------------------

data["sma_20"] = data["Close"].rolling(20).mean()
data["sma_50"] = data["Close"].rolling(50).mean()
data["sma_200"] = data["Close"].rolling(200).mean()


# --------------------------------------------------
# DISTANCE FROM MOVING AVERAGE
# --------------------------------------------------

data["distance_from_sma20"] = (
    data["Close"] / data["sma_20"] - 1
)


# --------------------------------------------------
# DRAWDOWN
# --------------------------------------------------

data["running_max"] = data["Close"].cummax()

data["drawdown"] = (
    data["Close"] / data["running_max"] - 1
)


# --------------------------------------------------
# CLEAN DATA
# --------------------------------------------------

data = data.dropna().copy()


# --------------------------------------------------
# SUMMARY
# --------------------------------------------------

print("\n" + "=" * 70)
print("DATA SUMMARY")
print("=" * 70)

print(f"\nUsable observations: {len(data):,}")

print("\nFirst five rows:")
print(
    data[
        [
            "Close",
            "return",
            "volatility_20d",
            "momentum_20d",
            "momentum_60d",
            "distance_from_sma20",
            "drawdown"
        ]
    ].head()
)


print("\n" + "=" * 70)
print("RETURN STATISTICS")
print("=" * 70)

print(f"""
Average daily return:     {data['return'].mean():.6f}
Daily volatility:         {data['return'].std():.6f}
Annualized return:        {data['return'].mean() * 252:.4f}
Annualized volatility:    {data['return'].std() * np.sqrt(252):.4f}
Minimum daily return:     {data['return'].min():.4f}
Maximum daily return:     {data['return'].max():.4f}
""")


print("=" * 70)
print("FEATURE STATISTICS")
print("=" * 70)

features = [
    "return",
    "volatility_20d",
    "momentum_20d",
    "momentum_60d",
    "distance_from_sma20",
    "drawdown"
]

print(data[features].describe())


# --------------------------------------------------
# SAVE DATA
# --------------------------------------------------

output_file = "spy_features.csv"

data.to_csv(output_file)

print(f"\nProcessed data saved to {output_file}")
print("\nStep 1 complete.")