# Junho Lee (utg2ue)
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "spy_walk_forward_regimes.csv"

TRADING_DAYS = 252

# Keep the SAME threshold we already used.
# We are deliberately not optimizing this after seeing results.
REVERSAL_THRESHOLD = 0.02

STRESS_REGIME = 2


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_metrics(returns, position):
    """
    Calculate performance statistics for a strategy.
    """

    returns = returns.dropna()

    position = (
        position
        .reindex(returns.index)
        .fillna(0)
    )

    active_mask = position != 0

    active_returns = returns[
        active_mask
    ]

    equity = (
        1 + returns
    ).cumprod()

    total_return = (
        equity.iloc[-1] - 1
    )

    years = (
        len(returns)
        / TRADING_DAYS
    )

    if (
        years > 0
        and equity.iloc[-1] > 0
    ):
        cagr = (
            equity.iloc[-1]
            ** (1 / years)
            - 1
        )
    else:
        cagr = np.nan

    daily_volatility = (
        returns.std()
    )

    annualized_volatility = (
        daily_volatility
        * np.sqrt(TRADING_DAYS)
    )

    if daily_volatility > 0:
        sharpe = (
            returns.mean()
            / daily_volatility
            * np.sqrt(TRADING_DAYS)
        )
    else:
        sharpe = np.nan

    running_peak = (
        equity.cummax()
    )

    drawdown = (
        equity
        / running_peak
        - 1
    )

    max_drawdown = (
        drawdown.min()
    )

    if len(active_returns) > 0:

        hit_rate = (
            active_returns > 0
        ).mean()

        avg_active_return = (
            active_returns.mean()
        )

        median_active_return = (
            active_returns.median()
        )

    else:

        hit_rate = np.nan
        avg_active_return = np.nan
        median_active_return = np.nan

    return {
        "Active Days":
            int(active_mask.sum()),

        "Total Return":
            total_return,

        "CAGR":
            cagr,

        "Annualized Volatility":
            annualized_volatility,

        "Sharpe":
            sharpe,

        "Max Drawdown":
            max_drawdown,

        "Hit Rate":
            hit_rate,

        "Average Active Return":
            avg_active_return,

        "Median Active Return":
            median_active_return,
    }


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("HMM REGIME FILTER TEST")
print("=" * 80)

print("\nLoading walk-forward data...")

df = pd.read_csv(
    DATA_FILE,
    index_col=0,
    parse_dates=True
)

df = df.sort_index()


# ============================================================
# VALIDATE DATA
# ============================================================

required_columns = [
    "return",
    "walk_forward_regime",
]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns: "
        f"{missing_columns}"
    )


df = df.dropna(
    subset=required_columns
).copy()

df["walk_forward_regime"] = (
    df["walk_forward_regime"]
    .astype(int)
)


print(
    f"Loaded {len(df):,} "
    f"out-of-sample observations."
)

print(
    f"First date: "
    f"{df.index.min().date()}"
)

print(
    f"Last date:  "
    f"{df.index.max().date()}"
)


# ============================================================
# CREATE THE SAME MEAN-REVERSION SIGNAL
# ============================================================

# Rule:
#
# SPY falls more than 2% today
#       ↓
# LONG SPY tomorrow
#
# SPY rises more than 2% today
#       ↓
# SHORT SPY tomorrow

df["reversal_signal"] = 0.0

df.loc[
    df["return"]
    < -REVERSAL_THRESHOLD,
    "reversal_signal"
] = 1.0

df.loc[
    df["return"]
    > REVERSAL_THRESHOLD,
    "reversal_signal"
] = -1.0


# ============================================================
# STRATEGY 1: UNFILTERED MEAN REVERSION
# ============================================================

# This strategy completely ignores the HMM.
#
# Any ±2% move generates a reversal trade the following day.

df["unfiltered_position"] = (
    df["reversal_signal"]
    .shift(1)
    .fillna(0)
)


# ============================================================
# STRATEGY 2: HMM-FILTERED MEAN REVERSION
# ============================================================

# The HMM regime observed today can only be used to determine
# tomorrow's position.

df["tradable_regime"] = (
    df["walk_forward_regime"]
    .shift(1)
)

# A reversal signal is allowed ONLY if the market was in the
# high-volatility/stress regime when the signal occurred.

df["filtered_position"] = 0.0

stress_signal_mask = (
    df["tradable_regime"]
    == STRESS_REGIME
)

df.loc[
    stress_signal_mask,
    "filtered_position"
] = df.loc[
    stress_signal_mask,
    "unfiltered_position"
]


# ============================================================
# CALCULATE RETURNS
# ============================================================

df["unfiltered_return"] = (
    df["unfiltered_position"]
    * df["return"]
)

df["filtered_return"] = (
    df["filtered_position"]
    * df["return"]
)


# ============================================================
# CALCULATE PERFORMANCE
# ============================================================

unfiltered_metrics = calculate_metrics(
    df["unfiltered_return"],
    df["unfiltered_position"]
)

filtered_metrics = calculate_metrics(
    df["filtered_return"],
    df["filtered_position"]
)


# ============================================================
# RESULTS TABLE
# ============================================================

results = pd.DataFrame(
    [
        {
            "Strategy":
                "Unfiltered Mean Reversion",
            **unfiltered_metrics,
        },
        {
            "Strategy":
                "HMM Stress-Regime Filter",
            **filtered_metrics,
        },
    ]
)

results = results.set_index(
    "Strategy"
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n" + "=" * 80)
print("MEAN-REVERSION BENCHMARK")
print("=" * 80)


for strategy, row in results.iterrows():

    print()
    print(strategy)

    print("-" * 60)

    print(
        f"Active Days:            "
        f"{int(row['Active Days']):,}"
    )

    print(
        f"Total Return:           "
        f"{row['Total Return']:.2%}"
    )

    print(
        f"CAGR:                   "
        f"{row['CAGR']:.2%}"
    )

    print(
        f"Annualized Volatility:  "
        f"{row['Annualized Volatility']:.2%}"
    )

    print(
        f"Sharpe Ratio:           "
        f"{row['Sharpe']:.2f}"
    )

    print(
        f"Maximum Drawdown:       "
        f"{row['Max Drawdown']:.2%}"
    )

    print(
        f"Hit Rate:               "
        f"{row['Hit Rate']:.2%}"
    )

    print(
        f"Average Active Return:  "
        f"{row['Average Active Return']:.4%}"
    )

    print(
        f"Median Active Return:   "
        f"{row['Median Active Return']:.4%}"
    )


# ============================================================
# DIRECT COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("DOES THE HMM FILTER ADD VALUE?")
print("=" * 80)


unfiltered_sharpe = (
    unfiltered_metrics["Sharpe"]
)

filtered_sharpe = (
    filtered_metrics["Sharpe"]
)


sharpe_difference = (
    filtered_sharpe
    - unfiltered_sharpe
)


unfiltered_hit_rate = (
    unfiltered_metrics["Hit Rate"]
)

filtered_hit_rate = (
    filtered_metrics["Hit Rate"]
)


hit_rate_difference = (
    filtered_hit_rate
    - unfiltered_hit_rate
)


print(
    f"\nUnfiltered Sharpe:       "
    f"{unfiltered_sharpe:.2f}"
)

print(
    f"HMM-Filtered Sharpe:     "
    f"{filtered_sharpe:.2f}"
)

print(
    f"Sharpe Improvement:      "
    f"{sharpe_difference:+.2f}"
)


print(
    f"\nUnfiltered Hit Rate:      "
    f"{unfiltered_hit_rate:.2%}"
)

print(
    f"HMM-Filtered Hit Rate:    "
    f"{filtered_hit_rate:.2%}"
)

print(
    f"Hit-Rate Improvement:     "
    f"{hit_rate_difference:+.2%}"
)


# ============================================================
# TRADE REDUCTION
# ============================================================

unfiltered_days = (
    unfiltered_metrics[
        "Active Days"
    ]
)

filtered_days = (
    filtered_metrics[
        "Active Days"
    ]
)


if unfiltered_days > 0:

    trade_reduction = (
        1
        - filtered_days
        / unfiltered_days
    )

else:

    trade_reduction = np.nan


print(
    f"\nUnfiltered Active Days:   "
    f"{unfiltered_days:,}"
)

print(
    f"HMM-Filtered Active Days: "
    f"{filtered_days:,}"
)

print(
    f"Signal Reduction:         "
    f"{trade_reduction:.2%}"
)


# ============================================================
# INTERPRETATION
# ============================================================

print("\n" + "=" * 80)
print("INTERPRETATION")
print("=" * 80)


if (
    filtered_sharpe
    > unfiltered_sharpe
):

    print(
        "\nThe HMM stress-regime filter "
        "improved the Sharpe ratio."
    )

else:

    print(
        "\nThe HMM stress-regime filter "
        "did NOT improve the Sharpe ratio."
    )


if (
    filtered_hit_rate
    > unfiltered_hit_rate
):

    print(
        "The HMM filter also improved "
        "the signal hit rate."
    )

else:

    print(
        "The HMM filter did NOT improve "
        "the signal hit rate."
    )


print(
    "\nThis test uses the same ±2% "
    "mean-reversion threshold for both "
    "strategies."
)

print(
    "The only difference is whether the "
    "walk-forward HMM stress regime is "
    "used as a filter."
)


# ============================================================
# SAVE RESULTS
# ============================================================

df.to_csv(
    "regime_filter_daily.csv"
)

results.to_csv(
    "regime_filter_results.csv"
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    "\nregime_filter_daily.csv"
)

print(
    "regime_filter_results.csv"
)

print(
    "\nRegime-filter benchmark complete."
)