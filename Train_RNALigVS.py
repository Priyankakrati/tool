# =========================================================
# FILE 2: train_rnaligvs.py
# =========================================================

import pandas as pd
import numpy as np
import json

from itertools import product

print("🔹 Loading training dataset...")

# =========================================================
# LOAD DATA
# =========================================================

df = pd.read_csv("training_data.csv")

df = df.dropna()

print("Dataset shape:", df.shape)

# =========================================================
# TARGET
# =========================================================

y = -df["Affinity"]

# =========================================================
# FEATURES
# =========================================================

features = [

    "Contact_density",

    "Electrostatic_score",

    "Hbond_strength",

    "Pi_stacking",

    "Curvature"
]

# =========================================================
# NORMALIZATION
# =========================================================

for f in features:

    min_val = df[f].min()

    max_val = df[f].max()

    if max_val - min_val != 0:

        df[f] = (
            df[f] - min_val
        ) / (
            max_val - min_val
        )

print("✅ Features normalized")

# =========================================================
# WEIGHT SEARCH
# =========================================================

print("🔹 Optimizing weights...")

weight_ranges = {

    "Contact_density":
        [0.25, 0.3, 0.35, 0.4],

    "Electrostatic_score":
        [0.25, 0.3, 0.35],

    "Hbond_strength":
        [0.1, 0.15],

    "Pi_stacking":
        [0.05, 0.1],

    "Curvature":
        [0.05, 0.1]
}

keys = list(weight_ranges.keys())

combinations = list(
    product(*weight_ranges.values())
)

best_r = -1

best_weights = None

for combo in combinations:

    weights = dict(
        zip(keys, combo)
    )

    # Normalize weights
    total = sum(weights.values())

    weights = {
        k: v / total
        for k, v in weights.items()
    }

    # Score
    score = sum(
        weights[f] * df[f]
        for f in features
    )

    # PCC
    r = np.corrcoef(
        y,
        score
    )[0,1]

    if np.isnan(r):
        continue

    if abs(r) > best_r:

        best_r = abs(r)

        best_weights = weights

print("\n🔥 BEST PCC:",
      round(best_r, 4))

print("\n🔥 BEST WEIGHTS:")
print(best_weights)

# =========================================================
# FINAL SCORE
# =========================================================

df["Score"] = sum(
    best_weights[f] * df[f]
    for f in features
)

# =========================================================
# MODEL PARAMETERS
# =========================================================

mean = df["Score"].mean()

std = df["Score"].std()

model = {

    "mean": float(mean),

    "std": float(std),

    "weights": best_weights
}

with open(
    "model_params.json",
    "w"
) as f:

    json.dump(
        model,
        f,
        indent=4
    )

print("\n✅ model_params.json saved!")
