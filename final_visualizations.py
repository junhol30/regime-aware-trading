# Junho Lee (utg2ue)
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

WALK_FORWARD_FILE = "spy_walk_forward_regimes.csv"
FILTER_FILE = "regime_filter_daily.csv"
FILTER_RESULTS_FILE = "regime_filter_results.csv"

OUTPUT_DIR = "figures"

REGIME_NAMES = {
    0: "Intermediate",
    1: "Low-Volatility / Bull",
    2: "High-Volatility / Stress",
}

REGIME_COLORS = {
    0: "tab:orange",
    1: "tab:green",
    2: "tab:red",
}


# ============================================================
# SETUP
# ============================================================

print("=" * 80)
print("FINAL PROJECT VISUALIZATIONS")
print("=" * 80)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print(
    f"\nFigures will be saved to: "
    f"{OUTPUT_DIR}/"
)


# ============================================================
# LOAD WALK-FORWARD REGIME DATA
# ============================================================

print("\nLoading walk-forward regime data...")

regimes = pd.read_csv(
    WALK_FORWARD_FILE,
    index_col=0,
    parse_dates=True
)

regimes = regimes.sort_index()

required_regime_columns = [
    "Close",
    "walk_forward_regime",
]

missing_regime_columns = [
    column
    for column in required_regime_columns
    if column not in regimes.columns
]

if missing_regime_columns:
    raise ValueError(
        "Missing required regime columns: "
        f"{missing_regime_columns}"
    )

regimes = regimes.dropna(
    subset=required_regime_columns
).copy()

regimes["walk_forward_regime"] = (
    regimes["walk_forward_regime"]
    .astype(int)
)

print(
    f"Loaded {len(regimes):,} "
    f"walk-forward observations."
)


# ============================================================
# LOAD FILTER TEST DATA
# ============================================================

print("Loading regime-filter backtest data...")

backtest = pd.read_csv(
    FILTER_FILE,
    index_col=0,
    parse_dates=True
)

backtest = backtest.sort_index()

required_backtest_columns = [
    "unfiltered_return",
    "filtered_return",
]

missing_backtest_columns = [
    column
    for column in required_backtest_columns
    if column not in backtest.columns
]

if missing_backtest_columns:
    raise ValueError(
        "Missing required backtest columns: "
        f"{missing_backtest_columns}"
    )

print(
    f"Loaded {len(backtest):,} "
    f"backtest observations."
)


# ============================================================
# LOAD SUMMARY RESULTS
# ============================================================

results = pd.read_csv(
    FILTER_RESULTS_FILE,
    index_col=0
)

required_result_columns = [
    "Sharpe",
    "Max Drawdown",
    "Hit Rate",
    "Active Days",
]

missing_result_columns = [
    column
    for column in required_result_columns
    if column not in results.columns
]

if missing_result_columns:
    raise ValueError(
        "Missing required result columns: "
        f"{missing_result_columns}"
    )


# ============================================================
# CREATE EQUITY CURVES
# ============================================================

backtest["unfiltered_equity"] = (
    1 + backtest["unfiltered_return"]
).cumprod()

backtest["filtered_equity"] = (
    1 + backtest["filtered_return"]
).cumprod()


# ============================================================
# CREATE DRAWDOWN SERIES
# ============================================================

backtest["unfiltered_drawdown"] = (
    backtest["unfiltered_equity"]
    / backtest["unfiltered_equity"].cummax()
    - 1
)

backtest["filtered_drawdown"] = (
    backtest["filtered_equity"]
    / backtest["filtered_equity"].cummax()
    - 1
)


# ============================================================
# FIGURE 1
# WALK-FORWARD HMM REGIMES
# ============================================================

print("\nGenerating walk-forward regime chart...")

fig, ax = plt.subplots(
    figsize=(15, 7)
)

ax.plot(
    regimes.index,
    regimes["Close"],
    linewidth=1.2,
    color="black",
    alpha=0.75,
)

for regime in range(3):

    mask = (
        regimes["walk_forward_regime"]
        == regime
    )

    ax.scatter(
        regimes.index[mask],
        regimes.loc[mask, "Close"],
        s=9,
        alpha=0.75,
        color=REGIME_COLORS[regime],
        label=(
            f"Regime {regime}: "
            f"{REGIME_NAMES[regime]}"
        ),
    )

ax.set_title(
    "Walk-Forward Hidden Markov Model Market Regimes",
    fontsize=15,
    fontweight="bold",
)

ax.set_xlabel("Date")
ax.set_ylabel("SPY Price")

ax.legend(
    loc="upper left"
)

ax.grid(
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "walk_forward_hmm_regimes.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 2
# EQUITY CURVE COMPARISON
# ============================================================

print("Generating equity-curve comparison...")

fig, ax = plt.subplots(
    figsize=(15, 7)
)

ax.plot(
    backtest.index,
    backtest["unfiltered_equity"],
    linewidth=2,
    label="Unfiltered ±2% Mean Reversion",
)

ax.plot(
    backtest.index,
    backtest["filtered_equity"],
    linewidth=2,
    label="HMM Stress-Regime Filter",
)

ax.set_title(
    "Mean-Reversion Strategy: Effect of HMM Regime Filtering",
    fontsize=15,
    fontweight="bold",
)

ax.set_xlabel("Date")
ax.set_ylabel("Growth of $1")

ax.legend()

ax.grid(
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "hmm_filter_equity_curve.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 3
# DRAWDOWN COMPARISON
# ============================================================

print("Generating drawdown comparison...")

fig, ax = plt.subplots(
    figsize=(15, 7)
)

ax.plot(
    backtest.index,
    backtest["unfiltered_drawdown"] * 100,
    linewidth=1.7,
    label="Unfiltered ±2% Mean Reversion",
)

ax.plot(
    backtest.index,
    backtest["filtered_drawdown"] * 100,
    linewidth=1.7,
    label="HMM Stress-Regime Filter",
)

ax.axhline(
    0,
    linewidth=1,
    color="black",
)

ax.set_title(
    "Strategy Drawdown: Unfiltered vs HMM-Filtered",
    fontsize=15,
    fontweight="bold",
)

ax.set_xlabel("Date")
ax.set_ylabel("Drawdown (%)")

ax.legend()

ax.grid(
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "hmm_filter_drawdown.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 4
# SHARPE RATIO COMPARISON
# ============================================================

print("Generating Sharpe-ratio comparison...")

strategy_labels = [
    "Unfiltered\nMean Reversion",
    "HMM-Filtered\nMean Reversion",
]

unfiltered_sharpe = results.loc[
    "Unfiltered Mean Reversion",
    "Sharpe"
]

filtered_sharpe = results.loc[
    "HMM Stress-Regime Filter",
    "Sharpe"
]

sharpe_values = [
    unfiltered_sharpe,
    filtered_sharpe,
]

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    strategy_labels,
    sharpe_values,
)

ax.set_title(
    "HMM Filtering Improved Risk-Adjusted Performance",
    fontsize=15,
    fontweight="bold",
)

ax.set_ylabel("Sharpe Ratio")

ax.grid(
    axis="y",
    alpha=0.2
)

for bar, value in zip(
    bars,
    sharpe_values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        f"{value:.2f}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "hmm_filter_sharpe.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 5
# HIT RATE COMPARISON
# ============================================================

print("Generating hit-rate comparison...")

unfiltered_hit_rate = (
    results.loc[
        "Unfiltered Mean Reversion",
        "Hit Rate"
    ]
    * 100
)

filtered_hit_rate = (
    results.loc[
        "HMM Stress-Regime Filter",
        "Hit Rate"
    ]
    * 100
)

hit_rate_values = [
    unfiltered_hit_rate,
    filtered_hit_rate,
]

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    strategy_labels,
    hit_rate_values,
)

ax.axhline(
    50,
    linewidth=1.2,
    linestyle="--",
    color="black",
    label="50% Reference",
)

ax.set_title(
    "HMM Filtering Improved Mean-Reversion Hit Rate",
    fontsize=15,
    fontweight="bold",
)

ax.set_ylabel("Hit Rate (%)")

ax.legend()

ax.grid(
    axis="y",
    alpha=0.2
)

for bar, value in zip(
    bars,
    hit_rate_values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        f"{value:.1f}%",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "hmm_filter_hit_rate.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 6
# MAXIMUM DRAWDOWN COMPARISON
# ============================================================

print("Generating maximum-drawdown comparison...")

unfiltered_max_dd = (
    results.loc[
        "Unfiltered Mean Reversion",
        "Max Drawdown"
    ]
    * 100
)

filtered_max_dd = (
    results.loc[
        "HMM Stress-Regime Filter",
        "Max Drawdown"
    ]
    * 100
)

max_dd_values = [
    unfiltered_max_dd,
    filtered_max_dd,
]

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    strategy_labels,
    max_dd_values,
)

ax.axhline(
    0,
    linewidth=1,
    color="black",
)

ax.set_title(
    "HMM Filtering Reduced Maximum Drawdown",
    fontsize=15,
    fontweight="bold",
)

ax.set_ylabel("Maximum Drawdown (%)")

ax.grid(
    axis="y",
    alpha=0.2
)

for bar, value in zip(
    bars,
    max_dd_values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        f"{value:.1f}%",
        ha="center",
        va="top",
        fontweight="bold",
    )

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "hmm_filter_max_drawdown.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# FIGURE 7
# ACTIVE SIGNAL COUNT
# ============================================================

print("Generating signal-count comparison...")

unfiltered_days = int(
    results.loc[
        "Unfiltered Mean Reversion",
        "Active Days"
    ]
)

filtered_days = int(
    results.loc[
        "HMM Stress-Regime Filter",
        "Active Days"
    ]
)

active_day_values = [
    unfiltered_days,
    filtered_days,
]

fig, ax = plt.subplots(
    figsize=(9, 6)
)

bars = ax.bar(
    strategy_labels,
    active_day_values,
)

ax.set_title(
    "HMM Filter Removed Lower-Quality Mean-Reversion Signals",
    fontsize=15,
    fontweight="bold",
)

ax.set_ylabel("Active Trading Days")

ax.grid(
    axis="y",
    alpha=0.2
)

for bar, value in zip(
    bars,
    active_day_values
):

    ax.text(
        bar.get_x()
        + bar.get_width() / 2,
        value,
        f"{value}",
        ha="center",
        va="bottom",
        fontweight="bold",
    )

fig.tight_layout()

fig.savefig(
    os.path.join(
        OUTPUT_DIR,
        "hmm_filter_signal_count.png"
    ),
    dpi=300,
    bbox_inches="tight",
)

plt.close(fig)


# ============================================================
# PRINT FINAL RESULTS
# ============================================================

sharpe_improvement = (
    filtered_sharpe
    - unfiltered_sharpe
)

hit_rate_improvement = (
    filtered_hit_rate
    - unfiltered_hit_rate
)

signal_reduction = (
    1
    - filtered_days
    / unfiltered_days
) * 100


print("\n" + "=" * 80)
print("FINAL RESEARCH SUMMARY")
print("=" * 80)

print(
    f"\nUnfiltered Sharpe:        "
    f"{unfiltered_sharpe:.2f}"
)

print(
    f"HMM-Filtered Sharpe:      "
    f"{filtered_sharpe:.2f}"
)

print(
    f"Sharpe Improvement:       "
    f"{sharpe_improvement:+.2f}"
)

print(
    f"\nUnfiltered Hit Rate:       "
    f"{unfiltered_hit_rate:.2f}%"
)

print(
    f"HMM-Filtered Hit Rate:     "
    f"{filtered_hit_rate:.2f}%"
)

print(
    f"Hit-Rate Improvement:      "
    f"{hit_rate_improvement:+.2f} percentage points"
)

print(
    f"\nUnfiltered Max Drawdown:   "
    f"{unfiltered_max_dd:.2f}%"
)

print(
    f"HMM-Filtered Max Drawdown: "
    f"{filtered_max_dd:.2f}%"
)

print(
    f"\nUnfiltered Active Days:    "
    f"{unfiltered_days:,}"
)

print(
    f"HMM-Filtered Active Days:  "
    f"{filtered_days:,}"
)

print(
    f"Signal Reduction:          "
    f"{signal_reduction:.2f}%"
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)
print("FIGURES SAVED")
print("=" * 80)

figure_names = [
    "walk_forward_hmm_regimes.png",
    "hmm_filter_equity_curve.png",
    "hmm_filter_drawdown.png",
    "hmm_filter_sharpe.png",
    "hmm_filter_hit_rate.png",
    "hmm_filter_max_drawdown.png",
    "hmm_filter_signal_count.png",
]

for figure_name in figure_names:

    print(
        f"{OUTPUT_DIR}/"
        f"{figure_name}"
    )

print(
    "\nStep 6A complete."
)