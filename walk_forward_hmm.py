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

# Roughly four years of trading history before we begin
# generating out-of-sample regime classifications.
INITIAL_TRAINING_DAYS = 1000

# Retrain the HMM approximately once per month.
RETRAIN_EVERY = 21

N_REGIMES = 3
RANDOM_STATE = 42

FEATURES = [
    "return",
    "volatility_20d",
    "momentum_20d",
]


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 80)
print("WALK-FORWARD HMM REGIME DETECTION")
print("=" * 80)

print("\nLoading SPY feature data...")

df = pd.read_csv(
    DATA_FILE,
    index_col=0,
    parse_dates=True
)

df = df.sort_index()

required_columns = FEATURES + ["Close"]

missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing required columns in {DATA_FILE}: "
        f"{missing_columns}"
    )

df = df.dropna(
    subset=required_columns
).copy()

print(f"Loaded {len(df):,} observations.")
print(f"First date: {df.index.min().date()}")
print(f"Last date:  {df.index.max().date()}")


# ============================================================
# WALK-FORWARD SETTINGS
# ============================================================

print("\n" + "=" * 80)
print("WALK-FORWARD SETTINGS")
print("=" * 80)

print("\nFeatures used:")

for feature in FEATURES:
    print(f"  - {feature}")

print(f"\nHidden regimes:        {N_REGIMES}")
print(f"Initial training days: {INITIAL_TRAINING_DAYS}")
print(
    f"Retrain frequency:     "
    f"every {RETRAIN_EVERY} trading days"
)

if len(df) <= INITIAL_TRAINING_DAYS:
    raise ValueError(
        "Not enough observations for the initial "
        "training period."
    )


# ============================================================
# REGIME MAPPING
# ============================================================

def build_regime_mapping(train_df, raw_states):
    """
    HMM state numbers are arbitrary.

    State 0 in one fitted model may represent a completely
    different market environment from State 0 in another
    fitted model.

    We therefore examine the states after every retraining
    and map them to consistent economic regime numbers:

        Regime 0 = Intermediate
        Regime 1 = Low-Volatility / Bull
        Regime 2 = High-Volatility / Stress
    """

    temp = train_df.copy()

    temp["raw_state"] = raw_states

    state_stats = (
        temp
        .groupby("raw_state")
        .agg(
            observations=("return", "count"),
            avg_return=("return", "mean"),
            avg_volatility=(
                "volatility_20d",
                "mean"
            ),
            avg_momentum=(
                "momentum_20d",
                "mean"
            ),
        )
    )

    # --------------------------------------------------------
    # HIGH-VOLATILITY STATE
    # --------------------------------------------------------

    high_vol_state = (
        state_stats["avg_volatility"]
        .idxmax()
    )

    # --------------------------------------------------------
    # LOW-VOLATILITY STATE
    # --------------------------------------------------------

    low_vol_state = (
        state_stats["avg_volatility"]
        .idxmin()
    )

    # --------------------------------------------------------
    # INTERMEDIATE STATE
    # --------------------------------------------------------

    intermediate_states = [
        state
        for state in state_stats.index
        if state not in [
            high_vol_state,
            low_vol_state
        ]
    ]

    if len(intermediate_states) != 1:
        raise ValueError(
            "Could not identify the intermediate regime."
        )

    intermediate_state = (
        intermediate_states[0]
    )

    mapping = {
        intermediate_state: 0,
        low_vol_state: 1,
        high_vol_state: 2,
    }

    return mapping, state_stats


# ============================================================
# STORAGE
# ============================================================

walk_forward_regimes = pd.Series(
    np.nan,
    index=df.index,
    name="walk_forward_regime"
)

regime_confidence = pd.Series(
    np.nan,
    index=df.index,
    name="regime_confidence"
)

retraining_records = []

model = None
scaler = None
regime_mapping = None

last_training_end = None


# ============================================================
# WALK-FORWARD LOOP
# ============================================================

print("\n" + "=" * 80)
print("RUNNING WALK-FORWARD ESTIMATION")
print("=" * 80)

total_predictions = (
    len(df) - INITIAL_TRAINING_DAYS
)

print(
    f"\nOut-of-sample predictions to generate: "
    f"{total_predictions:,}"
)

print(
    "\nEach prediction uses only information "
    "available on or before that date."
)

print()


for i in range(
    INITIAL_TRAINING_DAYS,
    len(df)
):

    prediction_number = (
        i - INITIAL_TRAINING_DAYS + 1
    )

    # ========================================================
    # RETRAIN HMM
    # ========================================================

    should_retrain = (
        model is None
        or (
            prediction_number - 1
        ) % RETRAIN_EVERY == 0
    )

    if should_retrain:

        # Training data stops BEFORE the day we are about
        # to classify.
        #
        # Therefore future observations are never used
        # to estimate the model.

        train_df = df.iloc[:i].copy()

        X_train = (
            train_df[FEATURES]
            .values
        )

        scaler = StandardScaler()

        X_train_scaled = (
            scaler.fit_transform(
                X_train
            )
        )

        model = GaussianHMM(
            n_components=N_REGIMES,
            covariance_type="full",
            n_iter=500,
            random_state=RANDOM_STATE,
        )

        model.fit(
            X_train_scaled
        )

        raw_training_states = (
            model.predict(
                X_train_scaled
            )
        )

        (
            regime_mapping,
            state_stats
        ) = build_regime_mapping(
            train_df,
            raw_training_states
        )

        last_training_end = (
            train_df.index.max()
        )

        retraining_records.append(
            {
                "prediction_start_date":
                    df.index[i],

                "training_start":
                    train_df.index.min(),

                "training_end":
                    train_df.index.max(),

                "training_observations":
                    len(train_df),

                "raw_intermediate_state":
                    [
                        raw
                        for raw, mapped
                        in regime_mapping.items()
                        if mapped == 0
                    ][0],

                "raw_low_vol_state":
                    [
                        raw
                        for raw, mapped
                        in regime_mapping.items()
                        if mapped == 1
                    ][0],

                "raw_high_vol_state":
                    [
                        raw
                        for raw, mapped
                        in regime_mapping.items()
                        if mapped == 2
                    ][0],
            }
        )


    # ========================================================
    # SEQUENTIAL REGIME INFERENCE
    # ========================================================

    # THIS IS THE IMPORTANT FIX.
    #
    # Previously we did something similar to:
    #
    #     model.predict(today)
    #
    # That gave the HMM only one isolated observation.
    #
    # Instead, we now provide the entire sequence of
    # observations from the beginning of the current model's
    # training history THROUGH TODAY.
    #
    # The HMM can therefore use:
    #
    #     yesterday's likely regime
    #              +
    #     transition probabilities
    #              +
    #     today's market features
    #
    # when determining today's state.

    sequence_df = (
        df.iloc[:i + 1]
        .copy()
    )

    X_sequence = (
        sequence_df[FEATURES]
        .values
    )

    X_sequence_scaled = (
        scaler.transform(
            X_sequence
        )
    )

    # --------------------------------------------------------
    # POSTERIOR STATE PROBABILITIES
    # --------------------------------------------------------

    # predict_proba gives the probability of each hidden
    # state for every observation in the sequence.

    raw_probabilities = (
        model.predict_proba(
            X_sequence_scaled
        )
    )

    # We only care about the LAST observation:
    # today's regime probabilities.

    today_raw_probabilities = (
        raw_probabilities[-1]
    )

    # Most likely raw HMM state.

    today_raw_state = int(
        np.argmax(
            today_raw_probabilities
        )
    )

    # Convert arbitrary HMM state number into our consistent
    # economic regime number.

    today_regime = (
        regime_mapping[
            today_raw_state
        ]
    )

    walk_forward_regimes.iloc[i] = (
        today_regime
    )

    regime_confidence.iloc[i] = (
        today_raw_probabilities[
            today_raw_state
        ]
    )


    # ========================================================
    # PROGRESS
    # ========================================================

    if (
        prediction_number % 250 == 0
        or prediction_number
        == total_predictions
    ):

        print(
            f"  Completed "
            f"{prediction_number:,}/"
            f"{total_predictions:,} "
            f"predictions..."
        )


# ============================================================
# COMBINE RESULTS
# ============================================================

results = df.copy()

results["walk_forward_regime"] = (
    walk_forward_regimes
)

results["regime_confidence"] = (
    regime_confidence
)

results = results.dropna(
    subset=[
        "walk_forward_regime"
    ]
).copy()

results["walk_forward_regime"] = (
    results[
        "walk_forward_regime"
    ]
    .astype(int)
)


# ============================================================
# ECONOMIC LABELS
# ============================================================

REGIME_NAMES = {
    0: "Intermediate",
    1: "Low-Volatility / Bull",
    2: "High-Volatility / Stress",
}

results["regime_label"] = (
    results[
        "walk_forward_regime"
    ]
    .map(
        REGIME_NAMES
    )
)


# ============================================================
# BASIC RESULTS
# ============================================================

print("\n" + "=" * 80)
print("OUT-OF-SAMPLE REGIME RESULTS")
print("=" * 80)

print(
    f"\nFirst out-of-sample date: "
    f"{results.index.min().date()}"
)

print(
    f"Last out-of-sample date:  "
    f"{results.index.max().date()}"
)

print(
    f"Out-of-sample observations: "
    f"{len(results):,}"
)


# ============================================================
# REGIME FREQUENCY
# ============================================================

print("\nRegime frequency:")

frequency_rows = []

for regime in range(
    N_REGIMES
):

    mask = (
        results[
            "walk_forward_regime"
        ]
        == regime
    )

    count = mask.sum()

    percentage = (
        count
        / len(results)
        * 100
    )

    frequency_rows.append(
        {
            "regime": regime,
            "label":
                REGIME_NAMES[regime],
            "days": count,
            "percentage":
                percentage,
        }
    )

    print(
        f"  Regime {regime} "
        f"({REGIME_NAMES[regime]}): "
        f"{count:,} days "
        f"({percentage:.2f}%)"
    )


frequency_df = pd.DataFrame(
    frequency_rows
)


# ============================================================
# REGIME CHARACTERISTICS
# ============================================================

print("\n" + "=" * 80)
print("OUT-OF-SAMPLE REGIME CHARACTERISTICS")
print("=" * 80)

summary_rows = []

for regime in range(
    N_REGIMES
):

    subset = results[
        results[
            "walk_forward_regime"
        ]
        == regime
    ]

    if len(subset) == 0:
        continue

    average_daily_return = (
        subset["return"]
        .mean()
    )

    annualized_return = (
        average_daily_return
        * 252
    )

    annualized_volatility = (
        subset["return"]
        .std()
        * np.sqrt(252)
    )

    if (
        annualized_volatility
        > 0
    ):

        sharpe = (
            annualized_return
            / annualized_volatility
        )

    else:

        sharpe = np.nan

    average_momentum = (
        subset[
            "momentum_20d"
        ]
        .mean()
    )

    average_confidence = (
        subset[
            "regime_confidence"
        ]
        .mean()
    )

    summary_rows.append(
        {
            "regime":
                regime,

            "label":
                REGIME_NAMES[
                    regime
                ],

            "observations":
                len(subset),

            "annualized_return":
                annualized_return,

            "annualized_volatility":
                annualized_volatility,

            "average_momentum":
                average_momentum,

            "average_confidence":
                average_confidence,

            "sharpe_approx":
                sharpe,
        }
    )

    print()
    print(
        f"Regime {regime}: "
        f"{REGIME_NAMES[regime]}"
    )

    print(
        f"  Observations:          "
        f"{len(subset):,}"
    )

    print(
        f"  Annualized Return:     "
        f"{annualized_return:.2%}"
    )

    print(
        f"  Annualized Volatility: "
        f"{annualized_volatility:.2%}"
    )

    print(
        f"  Average Momentum:      "
        f"{average_momentum:.2%}"
    )

    print(
        f"  Average Confidence:    "
        f"{average_confidence:.2%}"
    )

    print(
        f"  Approx. Sharpe:        "
        f"{sharpe:.2f}"
    )


summary_df = pd.DataFrame(
    summary_rows
)


# ============================================================
# TRANSITION MATRIX
# ============================================================

print("\n" + "=" * 80)
print("OUT-OF-SAMPLE TRANSITION MATRIX")
print("=" * 80)

previous_regime = (
    results[
        "walk_forward_regime"
    ]
    .shift(1)
)

current_regime = (
    results[
        "walk_forward_regime"
    ]
)

transition_counts = pd.crosstab(
    previous_regime,
    current_regime
)

transition_matrix = (
    transition_counts.div(
        transition_counts.sum(
            axis=1
        ),
        axis=0
    )
)

transition_matrix = (
    transition_matrix.reindex(
        index=range(
            N_REGIMES
        ),
        columns=range(
            N_REGIMES
        ),
        fill_value=0,
    )
)

transition_matrix.index = [
    f"Current Regime {i}"
    for i in range(
        N_REGIMES
    )
]

transition_matrix.columns = [
    f"Next Regime {i}"
    for i in range(
        N_REGIMES
    )
]

print()
print(
    transition_matrix.round(4)
)


# ============================================================
# STRESS-PERIOD SANITY CHECK
# ============================================================

print("\n" + "=" * 80)
print("STRESS-PERIOD SANITY CHECK")
print("=" * 80)

stress_periods = {
    "2015-2016 Selloff":
        (
            "2015-08-01",
            "2016-03-31"
        ),

    "2018 Q4 Selloff":
        (
            "2018-10-01",
            "2018-12-31"
        ),

    "COVID Crash":
        (
            "2020-02-15",
            "2020-04-30"
        ),

    "2022 Bear Market":
        (
            "2022-01-01",
            "2022-12-31"
        ),
}


stress_rows = []

for period_name, (
    start_date,
    end_date
) in stress_periods.items():

    period = results.loc[
        start_date:end_date
    ]

    if len(period) == 0:
        continue

    counts = (
        period[
            "walk_forward_regime"
        ]
        .value_counts()
    )

    high_vol_days = (
        counts.get(
            2,
            0
        )
    )

    high_vol_percentage = (
        high_vol_days
        / len(period)
        * 100
    )

    stress_rows.append(
        {
            "period":
                period_name,

            "trading_days":
                len(period),

            "high_vol_days":
                high_vol_days,

            "high_vol_percentage":
                high_vol_percentage,
        }
    )

    print()
    print(period_name)

    print(
        f"  Trading Days:          "
        f"{len(period):,}"
    )

    print(
        f"  High-Vol Regime Days:  "
        f"{high_vol_days:,}"
    )

    print(
        f"  High-Vol Percentage:   "
        f"{high_vol_percentage:.2f}%"
    )


stress_df = pd.DataFrame(
    stress_rows
)


# ============================================================
# RETRAINING LOG
# ============================================================

retraining_df = pd.DataFrame(
    retraining_records
)

print("\n" + "=" * 80)
print("WALK-FORWARD VALIDATION")
print("=" * 80)

print(
    f"\nNumber of HMM retrainings: "
    f"{len(retraining_df):,}"
)

print(
    "\nEach model was trained using only "
    "historical observations available "
    "before the corresponding prediction."
)

print(
    "\nCurrent-day regime inference uses the "
    "historical sequence ending on that day "
    "rather than classifying the observation "
    "in isolation."
)


# ============================================================
# SAVE RESULTS
# ============================================================

results.to_csv(
    "spy_walk_forward_regimes.csv"
)

summary_df.to_csv(
    "walk_forward_regime_statistics.csv",
    index=False
)

frequency_df.to_csv(
    "walk_forward_regime_frequency.csv",
    index=False
)

transition_matrix.to_csv(
    "walk_forward_transition_matrix.csv"
)

retraining_df.to_csv(
    "walk_forward_retraining_log.csv",
    index=False
)

stress_df.to_csv(
    "walk_forward_stress_test.csv",
    index=False
)


# ============================================================
# FINISHED
# ============================================================

print("\n" + "=" * 80)
print("FILES SAVED")
print("=" * 80)

print(
    "\nspy_walk_forward_regimes.csv"
)

print(
    "walk_forward_regime_statistics.csv"
)

print(
    "walk_forward_regime_frequency.csv"
)

print(
    "walk_forward_transition_matrix.csv"
)

print(
    "walk_forward_retraining_log.csv"
)

print(
    "walk_forward_stress_test.csv"
)

print(
    "\nCorrected Step 5A complete."
)