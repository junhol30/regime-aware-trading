# Junho Lee (utg2ue)
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# ============================================================
# SETTINGS
# ============================================================

TRADING_DAYS = 252

MOMENTUM_THRESHOLD = 0.00
REVERSAL_THRESHOLD = 0.02


# ============================================================
# PERFORMANCE FUNCTION
# ============================================================

def calculate_metrics(returns):
    """
    Calculate performance metrics for a return series.
    """

    returns = returns.dropna()

    if len(returns) == 0:
        return {
            "active_days": 0,
            "total_return": np.nan,
            "annualized_return": np.nan,
            "annualized_volatility": np.nan,
            "sharpe": np.nan,
            "max_drawdown": np.nan,
            "hit_rate": np.nan,
        }

    active_returns = returns[returns != 0]

    if len(active_returns) == 0:
        return {
            "active_days": 0,
            "total_return": 0.0,
            "annualized_return": 0.0,
            "annualized_volatility": 0.0,
            "sharpe": np.nan,
            "max_drawdown": 0.0,
            "hit_rate": np.nan,
        }

    equity = (1 + returns).cumprod()

    total_return = equity.iloc[-1] - 1

    years = len(returns) / TRADING_DAYS

    if years > 0 and equity.iloc[-1] > 0:
        annualized_return = (
            equity.iloc[-1] ** (1 / years)
        ) - 1
    else:
        annualized_return = np.nan

    volatility = returns.std()

    annualized_volatility = (
        volatility * np.sqrt(TRADING_DAYS)
    )

    if volatility > 0:
        sharpe = (
            returns.mean()
            / volatility
            * np.sqrt(TRADING_DAYS)
        )
    else:
        sharpe = np.nan

    running_peak = equity.cummax()

    drawdown = (
        equity / running_peak
    ) - 1

    max_drawdown = drawdown.min()

    hit_rate = (
        active_returns > 0
    ).mean()

    return {
        "active_days": len(active_returns),
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_volatility,
        "sharpe": sharpe,
        "max_drawdown": max_drawdown,
        "hit_rate": hit_rate,
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
# CHECK COLUMNS
# ============================================================

required_columns = [
    "Close",
    "return",
    "momentum_20d",
    "regime",
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
# CREATE MOMENTUM SIGNAL
# ============================================================

# Positive 20-day momentum:
#     go long
#
# Negative 20-day momentum:
#     go short

df["momentum_signal"] = 0.0

df.loc[
    df["momentum_20d"] > MOMENTUM_THRESHOLD,
    "momentum_signal"
] = 1.0

df.loc[
    df["momentum_20d"] < -MOMENTUM_THRESHOLD,
    "momentum_signal"
] = -1.0


# ============================================================
# CREATE MEAN-REVERSION SIGNAL
# ============================================================

# Large negative return:
#     buy the next day
#
# Large positive return:
#     short the next day

df["mean_reversion_signal"] = 0.0

df.loc[
    df["return"] < -REVERSAL_THRESHOLD,
    "mean_reversion_signal"
] = 1.0

df.loc[
    df["return"] > REVERSAL_THRESHOLD,
    "mean_reversion_signal"
] = -1.0


# ============================================================
# SHIFT SIGNALS
# ============================================================

# Today's information can only create tomorrow's position.

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
# STRATEGY RETURNS
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
# COMPARE STRATEGIES WITHIN EACH REGIME
# ============================================================

print()
print("=" * 80)
print("MOMENTUM VS MEAN REVERSION BY REGIME")
print("=" * 80)

results = []

for regime in sorted(
    df["regime"].dropna().unique()
):

    regime_df = df[
        df["regime"] == regime
    ].copy()

    for strategy_name, return_column in [
        ("Momentum", "momentum_return"),
        (
            "Mean Reversion",
            "mean_reversion_return"
        ),
    ]:

        metrics = calculate_metrics(
            regime_df[return_column]
        )

        results.append(
            {
                "regime": int(regime),
                "strategy": strategy_name,
                **metrics,
            }
        )


results_df = pd.DataFrame(results)


# ============================================================
# PRINT RESULTS
# ============================================================

for regime in sorted(
    results_df["regime"].unique()
):

    print()
    print(f"REGIME {regime}")
    print("-" * 80)

    regime_results = results_df[
        results_df["regime"] == regime
    ]

    for _, row in regime_results.iterrows():

        print()
        print(row["strategy"])

        print(
            f"  Active Days:          "
            f"{int(row['active_days']):,}"
        )

        print(
            f"  Total Return:         "
            f"{row['total_return']:.2%}"
        )

        print(
            f"  Annualized Return:    "
            f"{row['annualized_return']:.2%}"
        )

        print(
            f"  Annualized Volatility:"
            f" {row['annualized_volatility']:.2%}"
        )

        print(
            f"  Sharpe Ratio:         "
            f"{row['sharpe']:.2f}"
        )

        print(
            f"  Maximum Drawdown:     "
            f"{row['max_drawdown']:.2%}"
        )

        print(
            f"  Hit Rate:             "
            f"{row['hit_rate']:.2%}"
        )


# ============================================================
# IDENTIFY BEST STRATEGY PER REGIME
# ============================================================

print()
print("=" * 80)
print("BEST STRATEGY BY REGIME")
print("=" * 80)

best_rows = []

for regime in sorted(
    results_df["regime"].unique()
):

    subset = results_df[
        results_df["regime"] == regime
    ].copy()

    valid_subset = subset.dropna(
        subset=["sharpe"]
    )

    if len(valid_subset) == 0:
        continue

    best_index = valid_subset[
        "sharpe"
    ].idxmax()

    best = valid_subset.loc[
        best_index
    ]

    best_rows.append(best)

    print()
    print(
        f"Regime {regime}: "
        f"{best['strategy']}"
    )

    print(
        f"  Sharpe:   "
        f"{best['sharpe']:.2f}"
    )

    print(
        f"  Hit Rate: "
        f"{best['hit_rate']:.2%}"
    )


best_df = pd.DataFrame(best_rows)


# ============================================================
# REGIME LABELS
# ============================================================

regime_labels = {
    0: "Intermediate / Bull",
    1: "Low-Volatility / Bull",
    2: "High-Volatility / Bear",
}

df["regime_label"] = (
    df["regime"]
    .map(regime_labels)
)


# ============================================================
# FIGURE 1
# SPY PRICE COLORED BY HMM REGIME
# ============================================================

print()
print("Generating regime chart...")

fig, ax = plt.subplots(
    figsize=(14, 7)
)

regime_colors = {
    0: "tab:orange",
    1: "tab:green",
    2: "tab:red",
}

for regime in sorted(
    df["regime"].dropna().unique()
):

    mask = (
        df["regime"] == regime
    )

    ax.scatter(
        df.index[mask],
        df.loc[mask, "Close"],
        s=7,
        alpha=0.75,
        color=regime_colors.get(
            regime,
            "gray"
        ),
        label=(
            f"Regime {regime}: "
            f"{regime_labels.get(regime, '')}"
        ),
    )

ax.set_title(
    "SPY Market Regimes Identified by Hidden Markov Model"
)

ax.set_xlabel("Date")
ax.set_ylabel("SPY Price")

ax.legend()

ax.grid(
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    "spy_hmm_regimes.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# FIGURE 2
# SHARPE RATIO BY REGIME AND STRATEGY
# ============================================================

print("Generating Sharpe comparison...")

sharpe_pivot = (
    results_df
    .pivot(
        index="regime",
        columns="strategy",
        values="sharpe"
    )
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

sharpe_pivot.plot(
    kind="bar",
    ax=ax
)

ax.axhline(
    0,
    linewidth=1
)

ax.set_title(
    "Momentum vs Mean Reversion by Market Regime"
)

ax.set_xlabel("HMM Regime")
ax.set_ylabel("Annualized Sharpe Ratio")

ax.tick_params(
    axis="x",
    rotation=0
)

ax.grid(
    axis="y",
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    "strategy_sharpe_by_regime.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# FIGURE 3
# HIT RATE BY REGIME AND STRATEGY
# ============================================================

print("Generating hit-rate comparison...")

hit_rate_pivot = (
    results_df
    .pivot(
        index="regime",
        columns="strategy",
        values="hit_rate"
    )
    * 100
)

fig, ax = plt.subplots(
    figsize=(10, 6)
)

hit_rate_pivot.plot(
    kind="bar",
    ax=ax
)

ax.axhline(
    50,
    linewidth=1,
    linestyle="--"
)

ax.set_title(
    "Trading Signal Hit Rate by Market Regime"
)

ax.set_xlabel("HMM Regime")
ax.set_ylabel("Hit Rate (%)")

ax.tick_params(
    axis="x",
    rotation=0
)

ax.grid(
    axis="y",
    alpha=0.2
)

fig.tight_layout()

fig.savefig(
    "strategy_hit_rate_by_regime.png",
    dpi=300
)

plt.close(fig)


# ============================================================
# FIGURE 4
# BUY-AND-HOLD VS ORIGINAL REGIME-AWARE STRATEGY
# ============================================================

print("Generating equity curve comparison...")

try:

    backtest = pd.read_csv(
        "strategy_backtest.csv",
        index_col=0,
        parse_dates=True
    )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        backtest.index,
        backtest["strategy_equity"],
        label="Regime-Aware Strategy",
        linewidth=2
    )

    ax.plot(
        backtest.index,
        backtest["buy_hold_equity"],
        label="Buy & Hold SPY",
        linewidth=2
    )

    ax.set_title(
        "Regime-Aware Strategy vs Buy & Hold SPY"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel(
        "Growth of $1"
    )

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        "strategy_equity_curve.png",
        dpi=300
    )

    plt.close(fig)

except FileNotFoundError:

    print(
        "strategy_backtest.csv not found. "
        "Skipping equity curve."
    )


# ============================================================
# FIGURE 5
# DRAWDOWN COMPARISON
# ============================================================

print("Generating drawdown comparison...")

try:

    backtest = pd.read_csv(
        "strategy_backtest.csv",
        index_col=0,
        parse_dates=True
    )

    strategy_equity = (
        backtest["strategy_equity"]
    )

    buy_hold_equity = (
        backtest["buy_hold_equity"]
    )

    strategy_drawdown = (
        strategy_equity
        / strategy_equity.cummax()
        - 1
    )

    buy_hold_drawdown = (
        buy_hold_equity
        / buy_hold_equity.cummax()
        - 1
    )

    fig, ax = plt.subplots(
        figsize=(14, 7)
    )

    ax.plot(
        strategy_drawdown.index,
        strategy_drawdown * 100,
        label="Regime-Aware Strategy",
        linewidth=1.5
    )

    ax.plot(
        buy_hold_drawdown.index,
        buy_hold_drawdown * 100,
        label="Buy & Hold SPY",
        linewidth=1.5
    )

    ax.axhline(
        0,
        linewidth=1
    )

    ax.set_title(
        "Drawdown Comparison"
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown (%)")

    ax.legend()

    ax.grid(
        alpha=0.2
    )

    fig.tight_layout()

    fig.savefig(
        "strategy_drawdowns.png",
        dpi=300
    )

    plt.close(fig)

except FileNotFoundError:

    print(
        "strategy_backtest.csv not found. "
        "Skipping drawdown chart."
    )


# ============================================================
# SAVE TABLES
# ============================================================

results_df.to_csv(
    "strategy_comparison.csv",
    index=False
)

best_df.to_csv(
    "best_strategy_by_regime.csv",
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print()
print("=" * 80)
print("FILES SAVED")
print("=" * 80)

print()
print("strategy_comparison.csv")
print("best_strategy_by_regime.csv")
print("spy_hmm_regimes.png")
print("strategy_sharpe_by_regime.png")
print("strategy_hit_rate_by_regime.png")
print("strategy_equity_curve.png")
print("strategy_drawdowns.png")

print()
print("Step 4 complete.")