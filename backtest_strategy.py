# Junho Lee (utg2ue)
import numpy as np
import pandas as pd


# ============================================================
# SETTINGS
# ============================================================

TRADING_DAYS = 252

# Momentum thresholds
STRONG_MOMENTUM_THRESHOLD = 0.00
WEAK_MOMENTUM_THRESHOLD = 0.01

# Mean-reversion threshold for the high-volatility regime
REVERSAL_THRESHOLD = 0.02


# ============================================================
# PERFORMANCE METRICS
# ============================================================

def calculate_performance(returns, name):
    """
    Calculate common trading-strategy performance statistics.
    """

    returns = returns.dropna()

    if len(returns) == 0:
        raise ValueError(f"No returns available for {name}.")

    cumulative_growth = (1 + returns).cumprod()

    total_return = cumulative_growth.iloc[-1] - 1

    years = len(returns) / TRADING_DAYS

    if years > 0:
        cagr = cumulative_growth.iloc[-1] ** (1 / years) - 1
    else:
        cagr = np.nan

    annualized_volatility = returns.std() * np.sqrt(TRADING_DAYS)

    if returns.std() != 0:
        sharpe = (
            returns.mean()
            / returns.std()
            * np.sqrt(TRADING_DAYS)
        )
    else:
        sharpe = np.nan

    running_peak = cumulative_growth.cummax()

    drawdown = (
        cumulative_growth / running_peak
    ) - 1

    max_drawdown = drawdown.min()

    active_days = returns[returns != 0]

    if len(active_days) > 0:
        hit_rate = (active_days > 0).mean()
    else:
        hit_rate = np.nan

    return {
        "Strategy": name,
        "Total Return": total_return,
        "CAGR": cagr,
        "Annualized Volatility": annualized_volatility,
        "Sharpe": sharpe,
        "Max Drawdown": max_drawdown,
        "Hit Rate": hit_rate,
    }


# ============================================================
# LOAD DATA
# ============================================================

print("Loading regime data...")

df = pd.read_csv(
    "spy_regimes.csv",
    index_col=0,
    parse_dates=True
)

df = df.sort_index()

print(f"Loaded {len(df):,} observations.")
print(f"First date: {df.index.min().date()}")
print(f"Last date:  {df.index.max().date()}")


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "return",
    "momentum_20d",
    "regime"
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: {missing_columns}"
    )


# ============================================================
# CREATE TRADING SIGNALS
# ============================================================

print()
print("=" * 70)
print("BUILDING REGIME-AWARE SIGNALS")
print("=" * 70)

df["signal"] = 0.0


# ------------------------------------------------------------
# REGIME 1
# Low-volatility bull market
#
# Strong momentum strategy:
#
# Positive 20-day momentum -> long SPY
# Negative 20-day momentum -> short SPY
# ------------------------------------------------------------

regime_1 = df["regime"] == 1

df.loc[
    regime_1
    & (
        df["momentum_20d"]
        > STRONG_MOMENTUM_THRESHOLD
    ),
    "signal"
] = 1.0

df.loc[
    regime_1
    & (
        df["momentum_20d"]
        < -STRONG_MOMENTUM_THRESHOLD
    ),
    "signal"
] = -1.0


# ------------------------------------------------------------
# REGIME 0
# Intermediate / bull market
#
# Require stronger momentum before entering.
# Otherwise remain in cash.
# ------------------------------------------------------------

regime_0 = df["regime"] == 0

df.loc[
    regime_0
    & (
        df["momentum_20d"]
        > WEAK_MOMENTUM_THRESHOLD
    ),
    "signal"
] = 1.0

df.loc[
    regime_0
    & (
        df["momentum_20d"]
        < -WEAK_MOMENTUM_THRESHOLD
    ),
    "signal"
] = -1.0


# ------------------------------------------------------------
# REGIME 2
# High-volatility / bear market
#
# Mean-reversion strategy:
#
# Large negative daily move -> buy expected rebound
# Large positive daily move -> short expected reversal
# ------------------------------------------------------------

regime_2 = df["regime"] == 2

df.loc[
    regime_2
    & (
        df["return"]
        < -REVERSAL_THRESHOLD
    ),
    "signal"
] = 1.0

df.loc[
    regime_2
    & (
        df["return"]
        > REVERSAL_THRESHOLD
    ),
    "signal"
] = -1.0


# ============================================================
# PREVENT LOOK-AHEAD BIAS
# ============================================================

# The signal generated using today's information cannot be
# traded retroactively today.
#
# Shift the position forward one day:
#
# information at time t
#        ↓
# position at time t + 1

df["position"] = df["signal"].shift(1)

df["position"] = df["position"].fillna(0)


# ============================================================
# STRATEGY RETURNS
# ============================================================

df["strategy_return"] = (
    df["position"]
    * df["return"]
)

df["buy_hold_return"] = df["return"]


# ============================================================
# SIGNAL STATISTICS
# ============================================================

print()
print("Position distribution:")

position_counts = df["position"].value_counts().sort_index()

for position, count in position_counts.items():

    percentage = count / len(df) * 100

    if position == 1:
        label = "Long"

    elif position == -1:
        label = "Short"

    else:
        label = "Cash"

    print(
        f"{label:>6}: "
        f"{count:>5,} days "
        f"({percentage:6.2f}%)"
    )


# ============================================================
# PERFORMANCE
# ============================================================

strategy_metrics = calculate_performance(
    df["strategy_return"],
    "Regime-Aware Strategy"
)

buy_hold_metrics = calculate_performance(
    df["buy_hold_return"],
    "Buy & Hold SPY"
)

performance = pd.DataFrame(
    [
        strategy_metrics,
        buy_hold_metrics
    ]
)

performance = performance.set_index("Strategy")


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("=" * 70)
print("BACKTEST RESULTS")
print("=" * 70)

for strategy_name, row in performance.iterrows():

    print()
    print(strategy_name)
    print("-" * 45)

    print(
        f"Total Return:           "
        f"{row['Total Return']:>10.2%}"
    )

    print(
        f"CAGR:                   "
        f"{row['CAGR']:>10.2%}"
    )

    print(
        f"Annualized Volatility:  "
        f"{row['Annualized Volatility']:>10.2%}"
    )

    print(
        f"Sharpe Ratio:           "
        f"{row['Sharpe']:>10.2f}"
    )

    print(
        f"Maximum Drawdown:       "
        f"{row['Max Drawdown']:>10.2%}"
    )

    print(
        f"Hit Rate:               "
        f"{row['Hit Rate']:>10.2%}"
    )


# ============================================================
# PERFORMANCE BY REGIME
# ============================================================

print()
print("=" * 70)
print("STRATEGY PERFORMANCE BY REGIME")
print("=" * 70)

regime_results = []

for regime in sorted(df["regime"].dropna().unique()):

    subset = df[
        df["regime"] == regime
    ]

    active = subset[
        subset["position"] != 0
    ]

    if len(active) == 0:
        continue

    avg_return = active[
        "strategy_return"
    ].mean()

    volatility = active[
        "strategy_return"
    ].std()

    if volatility != 0:

        sharpe = (
            avg_return
            / volatility
            * np.sqrt(TRADING_DAYS)
        )

    else:
        sharpe = np.nan

    hit_rate = (
        active["strategy_return"] > 0
    ).mean()

    regime_results.append(
        {
            "Regime": int(regime),
            "Active Days": len(active),
            "Average Daily Return": avg_return,
            "Annualized Sharpe": sharpe,
            "Hit Rate": hit_rate,
        }
    )

regime_performance = pd.DataFrame(
    regime_results
)

print()

if len(regime_performance) > 0:
    print(
        regime_performance.to_string(
            index=False
        )
    )


# ============================================================
# EQUITY CURVES
# ============================================================

df["strategy_equity"] = (
    1 + df["strategy_return"]
).cumprod()

df["buy_hold_equity"] = (
    1 + df["buy_hold_return"]
).cumprod()


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    "strategy_backtest.csv"
)

performance.to_csv(
    "strategy_performance.csv"
)

regime_performance.to_csv(
    "strategy_regime_performance.csv",
    index=False
)


print()
print("=" * 70)
print("FILES SAVED")
print("=" * 70)

print()
print("strategy_backtest.csv")
print("strategy_performance.csv")
print("strategy_regime_performance.csv")

print()
print("Step 3 complete.")