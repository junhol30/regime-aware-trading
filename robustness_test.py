# Junho Lee (utg2ue)
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "spy_walk_forward_regimes.csv"
OUTPUT_DIR = "figures"

TRADING_DAYS = 252
STRESS_REGIME = 2

# We are testing several reasonable definitions of an
# "extreme" daily SPY move.
THRESHOLDS = [
    0.010,
    0.015,
    0.020,
    0.025,
    0.030,
]


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_metrics(returns, positions):

    returns = returns.fillna(0)

    positions = (
        positions
        .reindex(returns.index)
        .fillna(0)
    )

    active_mask = positions != 0
    active_returns = returns[active_mask]

    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1

    years = len(returns) / TRADING_DAYS

    if years > 0 and equity.iloc[-1] > 0:

        cagr = (
            equity.iloc[-1]
            ** (1 / years)
            - 1
        )

    else:

        cagr = np.nan

    daily_volatility = returns.std()

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

    running_peak = equity.cummax()

    drawdown = (
        equity
        / running_peak
        - 1
    )

    max_drawdown = drawdown.min()

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

        "cagr":
            cagr,

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
# LOAD DATA
# ============================================================

print("=" * 80)
print("MEAN-REVERSION THRESHOLD ROBUSTNESS TEST")
print("=" * 80)

print("\nLoading walk-forward regime data...")

df = pd.read_csv(
    DATA_FILE,
    index_col=0,
    parse_dates=True
)

df = df.sort_index()

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
# IMPORTANT TIMING
# ============================================================

# The regime identified at the end of today can only be used
# for tomorrow's trade.

df["tradable_regime"] = (
    df["walk_forward_regime"]
    .shift(1)
)


# ============================================================
# RUN THRESHOLD TESTS
# ============================================================

print("\n" + "=" * 80)
print("RUNNING THRESHOLD TESTS")
print("=" * 80)

all_results = []


for threshold in THRESHOLDS:

    print(
        f"\nTesting ±"
        f"{threshold:.1%} threshold..."
    )

    # --------------------------------------------------------
    # CREATE SIGNAL
    # --------------------------------------------------------

    signal = pd.Series(
        0.0,
        index=df.index
    )

    # Large negative move today:
    # bet on rebound tomorrow.

    signal.loc[
        df["return"] < -threshold
    ] = 1.0

    # Large positive move today:
    # bet on reversal tomorrow.

    signal.loc[
        df["return"] > threshold
    ] = -1.0


    # --------------------------------------------------------
    # SHIFT SIGNAL TO NEXT DAY
    # --------------------------------------------------------

    unfiltered_position = (
        signal
        .shift(1)
        .fillna(0)
    )


    # --------------------------------------------------------
    # HMM FILTER
    # --------------------------------------------------------

    filtered_position = pd.Series(
        0.0,
        index=df.index
    )

    stress_mask = (
        df["tradable_regime"]
        == STRESS_REGIME
    )

    filtered_position.loc[
        stress_mask
    ] = unfiltered_position.loc[
        stress_mask
    ]


    # --------------------------------------------------------
    # RETURNS
    # --------------------------------------------------------

    unfiltered_returns = (
        unfiltered_position
        * df["return"]
    )

    filtered_returns = (
        filtered_position
        * df["return"]
    )


    # --------------------------------------------------------
    # METRICS
    # --------------------------------------------------------

    unfiltered_metrics = (
        calculate_metrics(
            unfiltered_returns,
            unfiltered_position
        )
    )

    filtered_metrics = (
        calculate_metrics(
            filtered_returns,
            filtered_position
        )
    )


    # --------------------------------------------------------
    # STORE RESULTS
    # --------------------------------------------------------

    all_results.append(
        {
            "threshold":
                threshold,

            "threshold_percent":
                threshold * 100,

            "strategy":
                "Unfiltered",

            **unfiltered_metrics,
        }
    )

    all_results.append(
        {
            "threshold":
                threshold,

            "threshold_percent":
                threshold * 100,

            "strategy":
                "HMM Filtered",

            **filtered_metrics,
        }
    )


# ============================================================
# RESULTS DATAFRAME
# ============================================================

results = pd.DataFrame(
    all_results
)


# ============================================================
# PRINT FULL COMPARISON
# ============================================================

print("\n" + "=" * 80)
print("ROBUSTNESS RESULTS")
print("=" * 80)


for threshold in THRESHOLDS:

    threshold_results = results[
        results["threshold"]
        == threshold
    ]

    unfiltered = threshold_results[
        threshold_results["strategy"]
        == "Unfiltered"
    ].iloc[0]

    filtered = threshold_results[
        threshold_results["strategy"]
        == "HMM Filtered"
    ].iloc[0]

    print()
    print(
        f"±{threshold:.1%} DAILY MOVE"
    )

    print("-" * 80)

    print(
        f"{'Metric':<25}"
        f"{'Unfiltered':>15}"
        f"{'HMM Filtered':>18}"
        f"{'Difference':>15}"
    )

    print(
        f"{'Active Days':<25}"
        f"{int(unfiltered['active_days']):>15}"
        f"{int(filtered['active_days']):>18}"
        f"{int(filtered['active_days'] - unfiltered['active_days']):>15}"
    )

    print(
        f"{'CAGR':<25}"
        f"{unfiltered['cagr']:>14.2%}"
        f"{filtered['cagr']:>17.2%}"
        f"{filtered['cagr'] - unfiltered['cagr']:>14.2%}"
    )

    print(
        f"{'Sharpe':<25}"
        f"{unfiltered['sharpe']:>15.2f}"
        f"{filtered['sharpe']:>18.2f}"
        f"{filtered['sharpe'] - unfiltered['sharpe']:>+15.2f}"
    )

    print(
        f"{'Hit Rate':<25}"
        f"{unfiltered['hit_rate']:>14.2%}"
        f"{filtered['hit_rate']:>17.2%}"
        f"{filtered['hit_rate'] - unfiltered['hit_rate']:>+14.2%}"
    )

    print(
        f"{'Max Drawdown':<25}"
        f"{unfiltered['max_drawdown']:>14.2%}"
        f"{filtered['max_drawdown']:>17.2%}"
        f"{filtered['max_drawdown'] - unfiltered['max_drawdown']:>+14.2%}"
    )

    print(
        f"{'Avg Active Return':<25}"
        f"{unfiltered['average_active_return']:>14.4%}"
        f"{filtered['average_active_return']:>17.4%}"
        f"{filtered['average_active_return'] - unfiltered['average_active_return']:>+14.4%}"
    )


# ============================================================
# CREATE SUMMARY TABLE
# ============================================================

summary_rows = []


for threshold in THRESHOLDS:

    threshold_results = results[
        results["threshold"]
        == threshold
    ]

    unfiltered = threshold_results[
        threshold_results["strategy"]
        == "Unfiltered"
    ].iloc[0]

    filtered = threshold_results[
        threshold_results["strategy"]
        == "HMM Filtered"
    ].iloc[0]

    summary_rows.append(
        {
            "threshold_percent":
                threshold * 100,

            "unfiltered_sharpe":
                unfiltered["sharpe"],

            "filtered_sharpe":
                filtered["sharpe"],

            "sharpe_improvement":
                (
                    filtered["sharpe"]
                    - unfiltered["sharpe"]
                ),

            "unfiltered_hit_rate":
                unfiltered["hit_rate"],

            "filtered_hit_rate":
                filtered["hit_rate"],

            "hit_rate_improvement":
                (
                    filtered["hit_rate"]
                    - unfiltered["hit_rate"]
                ),

            "unfiltered_max_drawdown":
                unfiltered["max_drawdown"],

            "filtered_max_drawdown":
                filtered["max_drawdown"],

            "unfiltered_active_days":
                int(
                    unfiltered[
                        "active_days"
                    ]
                ),

            "filtered_active_days":
                int(
                    filtered[
                        "active_days"
                    ]
                ),
        }
    )


summary = pd.DataFrame(
    summary_rows
)


# ============================================================
# ROBUSTNESS SCORE
# ============================================================

sharpe_wins = (
    summary[
        "sharpe_improvement"
    ] > 0
).sum()

hit_rate_wins = (
    summary[
        "hit_rate_improvement"
    ] > 0
).sum()

drawdown_wins = (
    summary[
        "filtered_max_drawdown"
    ]
    >
    summary[
        "unfiltered_max_drawdown"
    ]
).sum()


print("\n" + "=" * 80)
print("ROBUSTNESS SUMMARY")
print("=" * 80)

print(
    f"\nThresholds tested: "
    f"{len(THRESHOLDS)}"
)

print(
    f"HMM Sharpe improvements: "
    f"{sharpe_wins}/"
    f"{len(THRESHOLDS)}"
)

print(
    f"HMM hit-rate improvements: "
    f"{hit_rate_wins}/"
    f"{len(THRESHOLDS)}"
)

print(
    f"HMM drawdown improvements: "
    f"{drawdown_wins}/"
    f"{len(THRESHOLDS)}"
)


average_sharpe_improvement = (
    summary[
        "sharpe_improvement"
    ].mean()
)

average_hit_rate_improvement = (
    summary[
        "hit_rate_improvement"
    ].mean()
)


print(
    f"\nAverage Sharpe improvement: "
    f"{average_sharpe_improvement:+.2f}"
)

print(
    f"Average hit-rate improvement: "
    f"{average_hit_rate_improvement:+.2%}"
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)


# ============================================================
# FIGURE 1: SHARPE ROBUSTNESS
# ============================================================

print(
    "\nGenerating Sharpe robustness chart..."
)

unfiltered_data = results[
    results["strategy"]
    == "Unfiltered"
]

filtered_data = results[
    results["strategy"]
    == "HMM Filtered"
]


fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    unfiltered_data[
        "threshold_percent"
    ],
    unfiltered_data[
        "sharpe"
    ],
    marker="o",
    linewidth=2,
    label="Unfiltered Mean Reversion",
)

ax.plot(
    filtered_data[
        "threshold_percent"
    ],
    filtered_data[
        "sharpe"
    ],
    marker="o",
    linewidth=2,
    label="HMM Stress-Regime Filter",
)

ax.axhline(
    0,
    linewidth=1,
    color="black",
)

ax.set_title(
    "Mean-Reversion Sharpe Across Signal Thresholds",
    fontsize=14,
    fontweight="bold",
)

ax.set_xlabel(
    "Absolute Daily SPY Move Threshold (%)"
)

ax.set_ylabel(
    "Sharpe Ratio"
)

ax.legend()

ax.grid(
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "threshold_sharpe_robustness.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2: HIT-RATE ROBUSTNESS
# ============================================================

print(
    "Generating hit-rate robustness chart..."
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

ax.plot(
    unfiltered_data[
        "threshold_percent"
    ],
    unfiltered_data[
        "hit_rate"
    ] * 100,
    marker="o",
    linewidth=2,
    label="Unfiltered Mean Reversion",
)

ax.plot(
    filtered_data[
        "threshold_percent"
    ],
    filtered_data[
        "hit_rate"
    ] * 100,
    marker="o",
    linewidth=2,
    label="HMM Stress-Regime Filter",
)

ax.axhline(
    50,
    linewidth=1,
    linestyle="--",
    color="black",
    label="50% Reference",
)

ax.set_title(
    "Mean-Reversion Hit Rate Across Signal Thresholds",
    fontsize=14,
    fontweight="bold",
)

ax.set_xlabel(
    "Absolute Daily SPY Move Threshold (%)"
)

ax.set_ylabel(
    "Hit Rate (%)"
)

ax.legend()

ax.grid(
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "threshold_hit_rate_robustness.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# SAVE CSV FILES
# ============================================================

results.to_csv(
    "robustness_threshold_results.csv",
    index=False
)

summary.to_csv(
    "robustness_threshold_summary.csv",
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    "\nrobustness_threshold_results.csv"
)

print(
    "robustness_threshold_summary.csv"
)

print(
    "figures/threshold_sharpe_robustness.png"
)

print(
    "figures/threshold_hit_rate_robustness.png"
)

print(
    "\nStep 7 threshold robustness test complete."
)