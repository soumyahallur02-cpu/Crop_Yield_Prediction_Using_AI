"""
optimize.py
------------
Given FIXED conditions a farmer cannot easily change this season
(crop, region, soil type, weather forecast, existing soil nutrients),
this module searches over the CONTROLLABLE inputs:

    - fertilizer_kg_ha
    - irrigation_mm
    - pesticide_l_ha
    - planting_density_k_per_ha

...to find the combination that maximizes predicted yield, optionally
subject to a cost/resource budget. This uses the trained ML model as a
"simulator" and performs a grid search (a simple, explainable optimizer;
swappable for Bayesian optimization / scipy.optimize for finer control).

Usage:
    python3 optimize.py
"""

import itertools
import joblib
import numpy as np
import pandas as pd

MODEL_PATH = "./models/crop_yield_model.pkl"

# Approximate cost per unit (illustrative, adjust to local prices)
COST_PER_UNIT = {
    "fertilizer_kg_ha": 0.9,   # $ per kg
    "irrigation_mm": 0.15,     # $ per mm applied (pumping/water cost)
    "pesticide_l_ha": 12.0,    # $ per liter
}

SEARCH_SPACE = {
    "fertilizer_kg_ha": np.arange(0, 341, 20),
    "irrigation_mm": np.arange(0, 701, 50),
    "pesticide_l_ha": np.arange(0, 6.1, 1.0),
    "planting_density_k_per_ha": np.arange(40, 161, 10),
}


def optimize_inputs(fixed_conditions: dict, model=None, budget_usd=None, top_k=5):
    """
    fixed_conditions must include:
      crop, region, soil_type, rainfall_mm, avg_temp_c, humidity_pct,
      sunlight_hours, soil_ph, soil_nitrogen_kg_ha, soil_phosphorus_kg_ha,
      soil_potassium_kg_ha, soil_organic_matter_pct, ndvi, days_to_harvest

    Returns a DataFrame of the top_k best (fertilizer, irrigation, pesticide,
    density) combinations ranked by predicted yield, with cost annotated,
    optionally filtered to combos within budget_usd.
    """
    if model is None:
        model = joblib.load(MODEL_PATH)

    combos = list(itertools.product(
        SEARCH_SPACE["fertilizer_kg_ha"],
        SEARCH_SPACE["irrigation_mm"],
        SEARCH_SPACE["pesticide_l_ha"],
        SEARCH_SPACE["planting_density_k_per_ha"],
    ))

    rows = []
    for fert, irrig, pest, dens in combos:
        row = dict(fixed_conditions)
        row.update({
            "fertilizer_kg_ha": fert,
            "irrigation_mm": irrig,
            "pesticide_l_ha": pest,
            "planting_density_k_per_ha": dens,
        })
        rows.append(row)

    candidates = pd.DataFrame(rows)
    candidates["predicted_yield_tons_ha"] = model.predict(candidates)

    candidates["input_cost_usd_ha"] = (
        candidates["fertilizer_kg_ha"] * COST_PER_UNIT["fertilizer_kg_ha"] +
        candidates["irrigation_mm"] * COST_PER_UNIT["irrigation_mm"] +
        candidates["pesticide_l_ha"] * COST_PER_UNIT["pesticide_l_ha"]
    )

    if budget_usd is not None:
        candidates = candidates[candidates["input_cost_usd_ha"] <= budget_usd]

    candidates = candidates.sort_values(
        "predicted_yield_tons_ha", ascending=False
    ).reset_index(drop=True)

    cols = [
        "fertilizer_kg_ha", "irrigation_mm", "pesticide_l_ha",
        "planting_density_k_per_ha", "predicted_yield_tons_ha", "input_cost_usd_ha",
    ]
    return candidates[cols].head(top_k)


def marginal_analysis(fixed_conditions: dict, model=None):
    """
    Holds all controllable inputs at a mid-range baseline except one,
    which is swept across its range — shows the yield-response curve
    for each input individually. Useful for farmer-facing 'what if' charts.
    """
    if model is None:
        model = joblib.load(MODEL_PATH)

    baseline = dict(fixed_conditions)
    baseline.update({
        "fertilizer_kg_ha": 150, "irrigation_mm": 300,
        "pesticide_l_ha": 2.0, "planting_density_k_per_ha": 90,
    })

    curves = {}
    for var, values in SEARCH_SPACE.items():
        rows = []
        for v in values:
            row = dict(baseline)
            row[var] = v
            rows.append(row)
        df = pd.DataFrame(rows)
        df["predicted_yield_tons_ha"] = model.predict(df)
        curves[var] = df[[var, "predicted_yield_tons_ha"]]

    return curves


if __name__ == "__main__":
    example_field = {
        "crop": "Maize",
        "region": "Central",
        "soil_type": "Loamy",
        "rainfall_mm": 550,
        "avg_temp_c": 25,
        "humidity_pct": 60,
        "sunlight_hours": 7.5,
        "soil_ph": 6.4,
        "soil_nitrogen_kg_ha": 55,
        "soil_phosphorus_kg_ha": 38,
        "soil_potassium_kg_ha": 42,
        "soil_organic_matter_pct": 3.2,
        "ndvi": 0.65,
        "days_to_harvest": 120,
    }

    print("=== Best input combinations (no budget limit) ===")
    print(optimize_inputs(example_field).to_string(index=False))

    print("\n=== Best input combinations (budget <= $250/ha) ===")
    print(optimize_inputs(example_field, budget_usd=250).to_string(index=False))
