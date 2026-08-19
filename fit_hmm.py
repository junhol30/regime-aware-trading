# Junho Lee (utg2ue)
import warnings

import numpy as np
import pandas as pd
from hmmlearn.hmm import GaussianHMM
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = "spy_features.csv"
OUTPUT_FILE = "spy_regimes.csv"

N_REGIMES = 3
RANDOM_STATE = 42

# These are the variables the HMM will use to identify regimes.
FEATURES = [
    "return",
    "volatility_20d",
    "momentum_20d",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 75)
print("HIDDEN MARKOV MODEL REGIME DETECTION")
print("=" * 75)

print("\nLoading SPY feature data...")

df = pd.read_csv(DATA_FILE, index_col=0, parse_dates=True)

print(f"Loaded {len(df):,} observations.")
print(f"First date: {df.index.min().date()}")
print(f"Last date:  {df.index.max().date()}")


# ============================================================
# CHECK FEATURES
# ============================================================

print("\n" + "=" * 75)
print("FEATURES USED BY HMM")
print("=" * 75)

for feature in FEATURES:
    if feature not in df.columns:
        raise ValueError(
            f"Required feature '{feature}' was not found in {DATA_FILE}.\n"
            f"Available columns: {list(df.columns)}"
        )

print("\nThe HMM will use:")

for feature in FEATURES:
    print(f"  - {feature}")


# ============================================================
# PREPARE DATA
# ============================================================

model_df = df.dropna(subset=FEATURES).copy()

X = model_df[FEATURES].values

print(f"\nObservations available for HMM: {len(model_df):,}")


# ============================================================
# STANDARDIZE FEATURES
# ============================================================

# The features have very different numerical scales.
#
# Example:
# return          ~ 0.01
# volatility      ~ 0.02
# momentum        ~ 0.05
#
# Standardization puts them onto comparable scales so that
# one feature does not dominate simply because its numbers
# happen to be larger.

scaler = StandardScaler()

X_scaled = scaler.fit_transform(X)

print("\nFeatures standardized successfully.")


# ============================================================
# FIT GAUSSIAN HIDDEN MARKOV MODEL
# ============================================================

print("\n" + "=" * 75)
print("TRAINING HIDDEN MARKOV MODEL")
print("=" * 75)

print(f"\nNumber of hidden regimes: {N_REGIMES}")

model = GaussianHMM(
    n_components=N_REGIMES,
    covariance_type="full",
    n_iter=1000,
    random_state=RANDOM_STATE,
)

model.fit(X_scaled)

print("HMM training complete.")


# ============================================================
# INFER MOST LIKELY REGIME
# ============================================================

regimes = model.predict(X_scaled)

model_df["regime"] = regimes

print("\nRegime assignments complete.")


# ============================================================
# REGIME COUNTS
# ============================================================

print("\n" + "=" * 75)
print("REGIME FREQUENCY")
print("=" * 75)

regime_counts = model_df["regime"].value_counts().sort_index()

for regime in range(N_REGIMES):

    count = regime_counts.get(regime, 0)
    percentage = count / len(model_df) * 100

    print(
        f"Regime {regime}: "
        f"{count:,} days "
        f"({percentage:.2f}%)"
    )


# ============================================================
# REGIME CHARACTERISTICS
# ============================================================

print("\n" + "=" * 75)
print("REGIME CHARACTERISTICS")
print("=" * 75)

regime_stats = (
    model_df
    .groupby("regime")
    .agg(
        observations=("return", "count"),
        avg_daily_return=("return", "mean"),
        daily_volatility=("return", "std"),
        avg_20d_volatility=("volatility_20d", "mean"),
        avg_momentum=("momentum_20d", "mean"),
        avg_drawdown=("drawdown", "mean"),
    )
)

regime_stats["annualized_return"] = (
    regime_stats["avg_daily_return"] * 252
)

regime_stats["annualized_volatility"] = (
    regime_stats["daily_volatility"] * np.sqrt(252)
)

regime_stats["sharpe_approx"] = (
    regime_stats["annualized_return"]
    / regime_stats["annualized_volatility"]
)

display_columns = [
    "observations",
    "avg_daily_return",
    "annualized_return",
    "annualized_volatility",
    "avg_20d_volatility",
    "avg_momentum",
    "avg_drawdown",
    "sharpe_approx",
]

print()
print(regime_stats[display_columns].round(4))


# ============================================================
# TRANSITION MATRIX
# ============================================================

print("\n" + "=" * 75)
print("REGIME TRANSITION MATRIX")
print("=" * 75)

transition_matrix = pd.DataFrame(
    model.transmat_,
    index=[
        f"Current Regime {i}"
        for i in range(N_REGIMES)
    ],
    columns=[
        f"Next Regime {i}"
        for i in range(N_REGIMES)
    ],
)

print()
print(transition_matrix.round(4))


# ============================================================
# REGIME PERSISTENCE
# ============================================================

print("\n" + "=" * 75)
print("REGIME PERSISTENCE")
print("=" * 75)

print(
    "\nProbability that the market remains "
    "in the same regime the next day:\n"
)

for regime in range(N_REGIMES):

    persistence = model.transmat_[regime, regime]

    print(
        f"Regime {regime}: "
        f"{persistence:.2%}"
    )


# ============================================================
# AVERAGE REGIME DURATION
# ============================================================

# If p is the probability of remaining in a regime,
# the approximate expected duration is:
#
#       1
#  -----------
#    1 - p

print("\nApproximate expected regime duration:\n")

for regime in range(N_REGIMES):

    persistence = model.transmat_[regime, regime]

    if persistence < 1:
        duration = 1 / (1 - persistence)

        print(
            f"Regime {regime}: "
            f"{duration:.1f} trading days"
        )

    else:
        print(
            f"Regime {regime}: "
            "effectively persistent"
        )


# ============================================================
# ASSIGN ECONOMIC LABELS
# ============================================================

# HMM regime numbers are arbitrary.
#
# Regime 0 does NOT automatically mean bull.
# Regime 1 does NOT automatically mean bear.
#
# We interpret the states after examining their statistical
# characteristics.
#
# Here we create descriptive labels using average return and
# volatility.

stats_for_labels = regime_stats.copy()

lowest_vol_regime = (
    stats_for_labels["annualized_volatility"].idxmin()
)

highest_vol_regime = (
    stats_for_labels["annualized_volatility"].idxmax()
)

remaining_regimes = [
    r
    for r in range(N_REGIMES)
    if r not in [lowest_vol_regime, highest_vol_regime]
]

regime_labels = {}

# High-volatility regime
high_vol_return = stats_for_labels.loc[
    highest_vol_regime,
    "annualized_return"
]

if high_vol_return < 0:
    regime_labels[highest_vol_regime] = "High-Volatility / Bear"
else:
    regime_labels[highest_vol_regime] = "High-Volatility"


# Low-volatility regime
low_vol_return = stats_for_labels.loc[
    lowest_vol_regime,
    "annualized_return"
]

if low_vol_return > 0:
    regime_labels[lowest_vol_regime] = "Low-Volatility / Bull"
else:
    regime_labels[lowest_vol_regime] = "Low-Volatility"


# Remaining regime
for regime in remaining_regimes:

    regime_return = stats_for_labels.loc[
        regime,
        "annualized_return"
    ]

    if regime_return > 0:
        regime_labels[regime] = "Intermediate / Bull"
    else:
        regime_labels[regime] = "Intermediate / Bear"


model_df["regime_label"] = (
    model_df["regime"].map(regime_labels)
)


# ============================================================
# PRINT INTERPRETATION
# ============================================================

print("\n" + "=" * 75)
print("ECONOMIC INTERPRETATION")
print("=" * 75)

for regime in range(N_REGIMES):

    stats = regime_stats.loc[regime]

    print(f"\nRegime {regime}")
    print("-" * 40)

    print(
        f"Label:                 "
        f"{regime_labels[regime]}"
    )

    print(
        f"Annualized return:     "
        f"{stats['annualized_return']:.2%}"
    )

    print(
        f"Annualized volatility: "
        f"{stats['annualized_volatility']:.2%}"
    )

    print(
        f"Average momentum:      "
        f"{stats['avg_momentum']:.2%}"
    )

    print(
        f"Average drawdown:      "
        f"{stats['avg_drawdown']:.2%}"
    )

    print(
        f"Approx. Sharpe:        "
        f"{stats['sharpe_approx']:.2f}"
    )


# ============================================================
# SAVE RESULTS
# ============================================================

columns_to_save = list(df.columns)

result = df.copy()

result["regime"] = np.nan
result["regime_label"] = None

result.loc[model_df.index, "regime"] = (
    model_df["regime"]
)

result.loc[model_df.index, "regime_label"] = (
    model_df["regime_label"]
)

result.to_csv(OUTPUT_FILE)

regime_stats.to_csv("regime_statistics.csv")
transition_matrix.to_csv("transition_matrix.csv")


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 75)
print("FILES SAVED")
print("=" * 75)

print(f"\n{OUTPUT_FILE}")
print("regime_statistics.csv")
print("transition_matrix.csv")

print("\nStep 2 complete.")