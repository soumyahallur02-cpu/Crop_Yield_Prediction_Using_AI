"""
predict.py
----------
Command-line tool: predict crop yield for one field, and print an
optimization recommendation alongside it.

Example:
    python3 predict.py --crop Wheat --region North --soil_type Loamy \
        --rainfall_mm 420 --avg_temp_c 19 --humidity_pct 55 \
        --sunlight_hours 7 --soil_ph 6.8 --soil_nitrogen_kg_ha 60 \
        --soil_phosphorus_kg_ha 40 --soil_potassium_kg_ha 45 \
        --soil_organic_matter_pct 3.0 --fertilizer_kg_ha 140 \
        --irrigation_mm 250 --pesticide_l_ha 2 --ndvi 0.62 \
        --planting_density_k_per_ha 85 --days_to_harvest 130
"""

import argparse
import joblib
import pandas as pd

from optimize import optimize_inputs

MODEL_PATH = "./models/crop_yield_model.pkl"

FIELDS = [
    "crop", "region", "soil_type", "rainfall_mm", "avg_temp_c", "humidity_pct",
    "sunlight_hours", "soil_ph", "soil_nitrogen_kg_ha", "soil_phosphorus_kg_ha",
    "soil_potassium_kg_ha", "soil_organic_matter_pct", "fertilizer_kg_ha",
    "irrigation_mm", "pesticide_l_ha", "ndvi", "planting_density_k_per_ha",
    "days_to_harvest",
]


def parse_args():
    p = argparse.ArgumentParser(description="Predict crop yield for a field")
    p.add_argument("--crop", required=True, choices=["Wheat", "Rice", "Maize", "Soybean", "Cotton"])
    p.add_argument("--region", required=True, choices=["North", "South", "East", "West", "Central"])
    p.add_argument("--soil_type", required=True, choices=["Loamy", "Sandy", "Clayey", "Silty"])
    p.add_argument("--rainfall_mm", type=float, required=True)
    p.add_argument("--avg_temp_c", type=float, required=True)
    p.add_argument("--humidity_pct", type=float, required=True)
    p.add_argument("--sunlight_hours", type=float, required=True)
    p.add_argument("--soil_ph", type=float, required=True)
    p.add_argument("--soil_nitrogen_kg_ha", type=float, required=True)
    p.add_argument("--soil_phosphorus_kg_ha", type=float, required=True)
    p.add_argument("--soil_potassium_kg_ha", type=float, required=True)
    p.add_argument("--soil_organic_matter_pct", type=float, required=True)
    p.add_argument("--fertilizer_kg_ha", type=float, required=True)
    p.add_argument("--irrigation_mm", type=float, required=True)
    p.add_argument("--pesticide_l_ha", type=float, required=True)
    p.add_argument("--ndvi", type=float, required=True)
    p.add_argument("--planting_density_k_per_ha", type=float, required=True)
    p.add_argument("--days_to_harvest", type=float, required=True)
    p.add_argument("--budget_usd", type=float, default=None,
                    help="Optional per-hectare input budget for optimization search")
    return p.parse_args()


def main():
    args = parse_args()
    row = {f: getattr(args, f) for f in FIELDS}

    model = joblib.load(MODEL_PATH)
    df = pd.DataFrame([row])
    pred = model.predict(df)[0]

    print("=" * 60)
    print(f"PREDICTED YIELD: {pred:.2f} tons/hectare  ({row['crop']}, {row['region']} region)")
    print("=" * 60)

    fixed = {k: v for k, v in row.items()
             if k not in ("fertilizer_kg_ha", "irrigation_mm", "pesticide_l_ha", "planting_density_k_per_ha")}

    print("\nTop 3 optimized input combinations for this field:")
    best = optimize_inputs(fixed, model=model, budget_usd=args.budget_usd, top_k=3)
    print(best.to_string(index=False))

    if not best.empty:
        top = best.iloc[0]
        uplift = top["predicted_yield_tons_ha"] - pred
        pct = (uplift / pred) * 100 if pred > 0 else 0
        print(f"\nRecommendation: adjusting to fertilizer={top['fertilizer_kg_ha']:.0f} kg/ha, "
              f"irrigation={top['irrigation_mm']:.0f} mm, pesticide={top['pesticide_l_ha']:.1f} L/ha, "
              f"density={top['planting_density_k_per_ha']:.0f}k/ha")
        print(f"could raise yield by ~{uplift:.2f} t/ha ({pct:+.1f}%) at an input cost of "
              f"${top['input_cost_usd_ha']:.0f}/ha.")


if __name__ == "__main__":
    main()
