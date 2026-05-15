# =========================================================
# FILE 3: vs.py
# =========================================================

import json
import numpy as np
import pandas as pd
from scipy.special import expit

# =========================================================
# LOAD MODEL
# =========================================================

with open("model_params.json") as f:

    model = json.load(f)

weights = model["weights"]

mean = model["mean"]

std = model["std"]

# =========================================================
# LOAD FEATURES
# =========================================================

df = pd.read_csv(
    "RNALigVS_final_features.csv"
)

features = [

    "Contact_density",

    "Electrostatic_score",

    "Hbond_strength",

    "Pi_stacking",

    "Curvature"
]

# =========================================================
# NORMALIZE FEATURES
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

# =========================================================
# SCORE
# =========================================================

df["Raw_Score"] = sum(
    weights[f] * df[f]
    for f in features
)

# =========================================================
# Z-SCORE
# =========================================================

df["Zscore"] = (
    df["Raw_Score"] - mean
) / std

# =========================================================
# BINDING PROBABILITY
# =========================================================

df["Binding_Probability"] = expit(
    df["Zscore"]
)

# =========================================================
# SORT
# =========================================================

df = df.sort_values(
    "Binding_Probability",
    ascending=False
)

df["Rank"] = range(
    1,
    len(df) + 1
)

# =========================================================
# SAVE RESULTS
# =========================================================

df.to_csv(
    "RNALigVS_predictions.csv",
    index=False
)

print("\n🔥 RNALigVS Screening Complete!")

print(
    df[
        [
            "PDB_ID",
            "Binding_Probability",
            "Rank"
        ]
    ].head()
)
