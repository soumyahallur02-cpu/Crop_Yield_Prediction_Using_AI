"""
generate_data.py
-----------------
Generates a realistic SYNTHETIC crop yield dataset for the project.

In a production version, you would replace this with real data from:
  - Soil sensors (moisture, pH, NPK)
  - Weather APIs (NOAA, OpenWeather, NASA POWER)
  - Satellite imagery indices (NDVI from Sentinel-2 / Landsat)
  - Farm records (planting date, fertilizer applied, irrigation logs)

The synthetic generator below encodes realistic agronomic relationships
(e.g. yield responds positively to rainfall up to a point, then drops from
waterlogging; fertilizer has diminishing returns; extreme heat hurts yield)
so that the ML models trained on it learn sensible, explainable patterns.
"""

import numpy as np
import pandas as pd

RANDOM_SEED = 42
N_SAMPLES = 6000

CROPS = ["Wheat", "Rice", "Maize", "Soybean", "Cotton"]
REGIONS = ["North", "South", "East", "West", "Central"]
SOIL_TYPES = ["Loamy", "Sandy", "Clayey", "Silty"]

# Base yield potential (tons/hectare) per crop under ideal conditions
CROP_BASE_YIELD = {
    "Wheat": 4.5,
    "Rice": 5.5,
    "Maize": 6.0,
    "Soybean": 3.0,
    "Cotton": 2.2,
}

# Ideal rainfall (mm) and temperature (C) per crop growing season
CROP_IDEAL_RAINFALL = {
    "Wheat": 450, "Rice": 1200, "Maize": 600, "Soybean": 500, "Cotton": 700,
}
CROP_IDEAL_TEMP = {
    "Wheat": 18, "Rice": 25, "Maize": 24, "Soybean": 23, "Cotton": 28,
}


def gaussian_response(x, ideal, spread):
    """Yield multiplier that peaks at 1.0 when x == ideal, falls off on both sides."""
    return np.exp(-0.5 * ((x - ideal) / spread) ** 2)


def generate_dataset(n=N_SAMPLES, seed=RANDOM_SEED):
    rng = np.random.default_rng(seed)

    crop = rng.choice(CROPS, size=n)
    region = rng.choice(REGIONS, size=n)
    soil_type = rng.choice(SOIL_TYPES, size=n)

    rainfall_mm = rng.normal(700, 250, n).clip(50, 2000)
    avg_temp_c = rng.normal(24, 6, n).clip(5, 45)
    humidity_pct = rng.normal(60, 15, n).clip(10, 100)
    sunlight_hours = rng.normal(7, 1.5, n).clip(2, 12)

    soil_ph = rng.normal(6.5, 0.7, n).clip(4.0, 9.0)
    soil_nitrogen = rng.normal(50, 15, n).clip(5, 120)      # kg/ha
    soil_phosphorus = rng.normal(40, 12, n).clip(5, 100)    # kg/ha
    soil_potassium = rng.normal(45, 14, n).clip(5, 110)     # kg/ha
    soil_organic_matter = rng.normal(3.0, 1.0, n).clip(0.2, 8.0)  # %

    fertilizer_kg_ha = rng.normal(150, 60, n).clip(0, 400)
    irrigation_mm = rng.normal(300, 120, n).clip(0, 900)
    pesticide_l_ha = rng.normal(2.0, 1.0, n).clip(0, 8)

    ndvi = rng.normal(0.6, 0.15, n).clip(0.1, 0.95)  # satellite vegetation index
    planting_density_k_ha = rng.normal(80, 20, n).clip(20, 200)  # thousand plants/ha

    days_to_harvest = rng.normal(120, 25, n).clip(60, 220)

    rows = []
    for i in range(n):
        c = crop[i]
        base = CROP_BASE_YIELD[c]

        ideal_rain = CROP_IDEAL_RAINFALL[c]
        ideal_temp = CROP_IDEAL_TEMP[c]

        # Each factor is bounded to [~0.5, 1.0] so the compound product stays realistic
        # (weighted geometric mean instead of a raw product of many terms).
        rain_factor = 0.5 + 0.5 * gaussian_response(rainfall_mm[i] + irrigation_mm[i] * 0.8, ideal_rain, ideal_rain * 0.6)
        temp_factor = 0.5 + 0.5 * gaussian_response(avg_temp_c[i], ideal_temp, 8)
        ph_factor = 0.6 + 0.4 * gaussian_response(soil_ph[i], 6.5, 1.5)

        # Fertilizer: diminishing returns (log curve), capped benefit
        fert_factor = 0.75 + 0.25 * (1 - np.exp(-fertilizer_kg_ha[i] / 120))

        # NPK balance factor
        npk_factor = 0.8 + 0.2 * gaussian_response(
            soil_nitrogen[i] + soil_phosphorus[i] + soil_potassium[i], 140, 80
        )

        organic_factor = 0.9 + 0.1 * (soil_organic_matter[i] / 8.0)
        ndvi_factor = 0.7 + 0.3 * ndvi[i]
        sunlight_factor = 0.85 + 0.15 * gaussian_response(sunlight_hours[i], 8, 3)
        pest_pressure_penalty = 1 - 0.02 * max(0, 3 - pesticide_l_ha[i])  # too little pesticide -> pest loss
        density_factor = 0.85 + 0.15 * gaussian_response(planting_density_k_ha[i], 90, 50)

        # Weighted geometric mean of all factors (keeps overall scale sane)
        factors = np.array([
            rain_factor, temp_factor, ph_factor, fert_factor,
            npk_factor, organic_factor, ndvi_factor, sunlight_factor,
            pest_pressure_penalty, density_factor,
        ])
        yield_multiplier = np.prod(factors ** 0.35)  # dampen compounding effect

        noise = rng.normal(1.0, 0.08)
        yield_t_ha = max(0.3, base * yield_multiplier * noise)

        rows.append({
            "crop": c,
            "region": region[i],
            "soil_type": soil_type[i],
            "rainfall_mm": round(rainfall_mm[i], 1),
            "avg_temp_c": round(avg_temp_c[i], 1),
            "humidity_pct": round(humidity_pct[i], 1),
            "sunlight_hours": round(sunlight_hours[i], 2),
            "soil_ph": round(soil_ph[i], 2),
            "soil_nitrogen_kg_ha": round(soil_nitrogen[i], 1),
            "soil_phosphorus_kg_ha": round(soil_phosphorus[i], 1),
            "soil_potassium_kg_ha": round(soil_potassium[i], 1),
            "soil_organic_matter_pct": round(soil_organic_matter[i], 2),
            "fertilizer_kg_ha": round(fertilizer_kg_ha[i], 1),
            "irrigation_mm": round(irrigation_mm[i], 1),
            "pesticide_l_ha": round(pesticide_l_ha[i], 2),
            "ndvi": round(ndvi[i], 3),
            "planting_density_k_per_ha": round(planting_density_k_ha[i], 1),
            "days_to_harvest": round(days_to_harvest[i]),
            "yield_tons_per_ha": round(yield_t_ha, 3),
        })

    return pd.DataFrame(rows)


if __name__ == "__main__":
    df = generate_dataset()
    out_path = "/home/claude/crop_yield_project/data/crop_yield_data.csv"
    df.to_csv(out_path, index=False)
    print(f"Generated {len(df)} rows -> {out_path}")
    print(df.describe(include="all").T[["count", "mean", "min", "max"]] if False else df.head())
