# Junho Lee (utg2ue)
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "spy_walk_forward_regimes.csv"

TRADING_DAYS = 252

# Keep the same thresholds used in our earlier analysis.
# We do NOT change them based on the new out-of-sample results.
MOMENTUM_THRESHOLD = 0.00
REVERSAL_THRESHOLD = 0.02


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_metrics(returns, active_mask):
    """
    Calculate performance statistics.

    'returns' contains the strategy's daily returns during
    the selected regime.

    'active_mask' tells us which days the strategy actually
    held a non-zero position.
    """

    returns = returns.dropna()

    active_mask = active_mask.reindex(
        returns.index
    ).fillna(False)

    active_returns = returns[
        active_mask
    ]

    if len(returns) == 0:
        return {
            "active_days": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
            "average_active_return": np.nan,
        }

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
        annualized_return = (
            equity.iloc[-1]
            ** (1 / years)
            - 1
        )
    else:
        annualized_return = np.nan

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

        average_active_return = (
            active_returns.mean()
        )

    else:

        hit_rate = np.nan
        average_active_return = np.nan

    return {
        "active_days":
            int(active_mask.sum()),

        "total_return":
            total_return,

        "annualized_return":
            annualized_return,

        "annualized_volatility":
            annualized_volatility,

        "sharpe":
            sharpe,

        "max_drawdown":
            max_drawdown,

        "hit_rate":
            hit_rate,

        "average_active_return":
            average_active_return,
    }


# ============================================================
# LOAD WALK-FORWARD DATA
# ============================================================

print("=" * 80)
print("WALK-FORWARD STRATEGY ANALYSIS")
print("=" * 80)

print("\nLoading walk-forward regime data...")

df = pd.read_csv(
    DATA_FILE,
    index_col=0,
    parse_dates=True
)

df = df.sort_index()


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "Close",
    "return",
    "momentum_20d",
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
# IMPORTANT TIMING CONVENTION
# ============================================================

print("\n" + "=" * 80)
print("SIGNAL TIMING")
print("=" * 80)

print(
    "\nInformation observed on day t "
    "creates a position for day t+1."
)

print(
    "This prevents the strategy from "
    "trading on information before it "
    "was available."
)


# ============================================================
# MOMENTUM SIGNAL
# ============================================================

# Positive 20-day momentum:
#     LONG
#
# Negative 20-day momentum:
#     SHORT

df["momentum_signal"] = 0.0

df.loc[
    df["momentum_20d"]
    > MOMENTUM_THRESHOLD,
    "momentum_signal"
] = 1.0

df.loc[
    df["momentum_20d"]
    < -MOMENTUM_THRESHOLD,
    "momentum_signal"
] = -1.0


# ============================================================
# MEAN-REVERSION SIGNAL
# ============================================================

# If SPY falls more than 2% today:
#     LONG tomorrow
#
# If SPY rises more than 2% today:
#     SHORT tomorrow

df["mean_reversion_signal"] = 0.0

df.loc[
    df["return"]
    < -REVERSAL_THRESHOLD,
    "mean_reversion_signal"
] = 1.0

df.loc[
    df["return"]
    > REVERSAL_THRESHOLD,
    "mean_reversion_signal"
] = -1.0


# ============================================================
# SHIFT SIGNALS FORWARD ONE DAY
# ============================================================

df["momentum_position"] = (
    df["momentum_signal"]
    .shift(1)
    .fillna(0)
)

df["mean_reversion_position"] = (
    df["mean_reversion_signal"]
    .shift(1)
    .fillna(0)
)


# ============================================================
# IMPORTANT: SHIFT REGIME TOO
# ============================================================

# The regime identified at the end of day t cannot be used
# to trade the return that already occurred during day t.
#
# Therefore:
#
# regime observed at t
#        ↓
# regime used for position at t+1

df["tradable_regime"] = (
    df["walk_forward_regime"]
    .shift(1)
)

df = df.dropna(
    subset=["tradable_regime"]
).copy()

df["tradable_regime"] = (
    df["tradable_regime"]
    .astype(int)
)


# ============================================================
# RAW STRATEGY RETURNS
# ============================================================

df["momentum_return"] = (
    df["momentum_position"]
    * df["return"]
)

df["mean_reversion_return"] = (
    df["mean_reversion_position"]
    * df["return"]
)


# ============================================================
# REGIME NAMES
# ============================================================

REGIME_NAMES = {
    0: "Intermediate",
    1: "Low-Volatility / Bull",
    2: "High-Volatility / Stress",
}


# ============================================================
# TEST BOTH STRATEGIES INSIDE EVERY REGIME
# ============================================================

print("\n" + "=" * 80)
print("OUT-OF-SAMPLE MOMENTUM VS MEAN REVERSION")
print("=" * 80)

comparison_rows = []


for regime in range(3):

    regime_mask = (
        df["tradable_regime"]
        == regime
    )

    regime_df = (
        df.loc[regime_mask]
        .copy()
    )

    print()
    print(
        f"REGIME {regime}: "
        f"{REGIME_NAMES[regime]}"
    )

    print("-" * 80)

    print(
        f"Total regime days: "
        f"{len(regime_df):,}"
    )

    strategies = [
        (
            "Momentum",
            "momentum_return",
            "momentum_position"
        ),
        (
            "Mean Reversion",
            "mean_reversion_return",
            "mean_reversion_position"
        ),
    ]

    for (
        strategy_name,
        return_column,
        position_column
    ) in strategies:

        active_mask = (
            regime_df[
                position_column
            ]
            != 0
        )

        metrics = (
            calculate_metrics(
                regime_df[
                    return_column
                ],
                active_mask
            )
        )

        comparison_rows.append(
            {
                "regime":
                    regime,

                "regime_label":
                    REGIME_NAMES[
                        regime
                    ],

                "strategy":
                    strategy_name,

                "regime_days":
                    len(regime_df),

                **metrics,
            }
        )

        print()
        print(strategy_name)

        print(
            f"  Active Days:             "
            f"{metrics['active_days']:,}"
        )

        print(
            f"  Total Return:            "
            f"{metrics['total_return']:.2%}"
        )

        print(
            f"  Annualized Return:       "
            f"{metrics['annualized_return']:.2%}"
        )

        print(
            f"  Annualized Volatility:   "
            f"{metrics['annualized_volatility']:.2%}"
        )

        print(
            f"  Sharpe Ratio:            "
            f"{metrics['sharpe']:.2f}"
        )

        print(
            f"  Maximum Drawdown:        "
            f"{metrics['max_drawdown']:.2%}"
        )

        print(
            f"  Hit Rate:                "
            f"{metrics['hit_rate']:.2%}"
        )

        print(
            f"  Avg Active-Day Return:   "
            f"{metrics['average_active_return']:.4%}"
        )


comparison_df = pd.DataFrame(
    comparison_rows
)


# ============================================================
# SIDE-BY-SIDE SUMMARY
# ============================================================

print("\n" + "=" * 80)
print("OUT-OF-SAMPLE SUMMARY")
print("=" * 80)


for regime in range(3):

    subset = comparison_df[
        comparison_df["regime"]
        == regime
    ]

    print()
    print(
        f"Regime {regime}: "
        f"{REGIME_NAMES[regime]}"
    )

    print("-" * 65)

    for _, row in subset.iterrows():

        print(
            f"{row['strategy']:<16}"
            f"Sharpe = "
            f"{row['sharpe']:>6.2f}   "
            f"Hit Rate = "
            f"{row['hit_rate']:>7.2%}   "
            f"Active Days = "
            f"{int(row['active_days']):>4}"
        )


# ============================================================
# IDENTIFY THE BETTER STRATEGY
# ============================================================

print("\n" + "=" * 80)
print("BETTER STRATEGY BY REGIME")
print("=" * 80)

best_rows = []


for regime in range(3):

    subset = comparison_df[
        comparison_df["regime"]
        == regime
    ].copy()

    valid = subset.dropna(
        subset=["sharpe"]
    )

    if len(valid) == 0:
        continue

    best_index = (
        valid["sharpe"]
        .idxmax()
    )

    best = (
        valid.loc[
            best_index
        ]
    )

    best_rows.append(
        best.to_dict()
    )

    print()
    print(
        f"Regime {regime} "
        f"({REGIME_NAMES[regime]})"
    )

    print(
        f"  Higher Sharpe: "
        f"{best['strategy']}"
    )

    print(
        f"  Sharpe:        "
        f"{best['sharpe']:.2f}"
    )

    print(
        f"  Hit Rate:      "
        f"{best['hit_rate']:.2%}"
    )

    print(
        f"  Active Days:   "
        f"{int(best['active_days']):,}"
    )


best_df = pd.DataFrame(
    best_rows
)


# ============================================================
# ORIGINAL HYPOTHESIS CHECK
# ============================================================

print("\n" + "=" * 80)
print("ORIGINAL HYPOTHESIS CHECK")
print("=" * 80)

print(
    "\nOur earlier in-sample analysis suggested:"
)

print(
    "  Regime 0 -> neither strategy compelling"
)

print(
    "  Regime 1 -> momentum"
)

print(
    "  Regime 2 -> mean reversion"
)

print(
    "\nThe numbers above tell us whether those "
    "relationships survive when regimes are "
    "identified without future information."
)


# ============================================================
# SAVE DAILY DATA
# ============================================================

df.to_csv(
    "walk_forward_strategy_daily.csv"
)


# ============================================================
# SAVE COMPARISON TABLE
# ============================================================

comparison_df.to_csv(
    "walk_forward_strategy_comparison.csv",
    index=False
)


# ============================================================
# SAVE BEST STRATEGIES
# ============================================================

best_df.to_csv(
    "walk_forward_best_strategy_by_regime.csv",
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    "\nwalk_forward_strategy_daily.csv"
)

print(
    "walk_forward_strategy_comparison.csv"
)

print(
    "walk_forward_best_strategy_by_regime.csv"
)

print(
    "\nStep 5B complete."
)