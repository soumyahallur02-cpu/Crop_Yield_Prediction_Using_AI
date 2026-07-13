"""
train_model.py
---------------
Trains and evaluates ML models to predict crop yield (tons/hectare) from
soil, weather, satellite (NDVI), and farm-management features.

Models compared:
  - Linear Regression (baseline)
  - Random Forest Regressor
  - Gradient Boosting Regressor  (best performer, saved as final model)

Outputs:
  - models/crop_yield_model.pkl   (trained pipeline: preprocessing + model)
  - models/feature_importance.png
  - models/model_comparison.png
  - models/metrics.json
"""

import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

DATA_PATH = "./data/crop_yield_data.csv"
MODEL_DIR = "./models"

TARGET = "yield_tons_per_ha"
CATEGORICAL_FEATURES = ["crop", "region", "soil_type"]
NUMERIC_FEATURES = [
    "rainfall_mm", "avg_temp_c", "humidity_pct", "sunlight_hours",
    "soil_ph", "soil_nitrogen_kg_ha", "soil_phosphorus_kg_ha",
    "soil_potassium_kg_ha", "soil_organic_matter_pct",
    "fertilizer_kg_ha", "irrigation_mm", "pesticide_l_ha",
    "ndvi", "planting_density_k_per_ha", "days_to_harvest",
]


def build_preprocessor():
    return ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(model, X_test, y_test, name):
    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)
    print(f"{name:22s}  MAE={mae:.3f}  RMSE={rmse:.3f}  R2={r2:.3f}")
    return {"model": name, "mae": mae, "rmse": rmse, "r2": r2}


def main():
    df = pd.read_csv(DATA_PATH)
    X = df[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = build_preprocessor()

    candidates = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            random_state=42, n_jobs=-1
        ),
        "Gradient Boosting": GradientBoostingRegressor(
            n_estimators=300, max_depth=3, learning_rate=0.05, random_state=42
        ),
    }

    results = []
    fitted_pipelines = {}

    for name, estimator in candidates.items():
        pipe = Pipeline([("prep", preprocessor), ("model", estimator)])
        pipe.fit(X_train, y_train)
        fitted_pipelines[name] = pipe
        results.append(evaluate(pipe, X_test, y_test, name))

    # Pick best model by R2
    best_name = max(results, key=lambda r: r["r2"])["model"]
    best_pipeline = fitted_pipelines[best_name]
    print(f"\nBest model: {best_name}")

    # 5-fold cross validation on the best model for robustness check
    cv_scores = cross_val_score(best_pipeline, X, y, cv=5, scoring="r2")
    print(f"5-fold CV R2: mean={cv_scores.mean():.3f}  std={cv_scores.std():.3f}")

    # Save model
    import os
    os.makedirs(MODEL_DIR, exist_ok=True)
    joblib.dump(best_pipeline, f"{MODEL_DIR}/crop_yield_model.pkl")

    with open(f"{MODEL_DIR}/metrics.json", "w") as f:
        json.dump({
            "results": results,
            "best_model": best_name,
            "cv_r2_mean": cv_scores.mean(),
            "cv_r2_std": cv_scores.std(),
            "numeric_features": NUMERIC_FEATURES,
            "categorical_features": CATEGORICAL_FEATURES,
        }, f, indent=2)

    # --- Plot 1: model comparison ---
    plt.figure(figsize=(7, 4.5))
    names = [r["model"] for r in results]
    r2s = [r["r2"] for r in results]
    colors = ["#8B9DC3" if n != best_name else "#2E7D32" for n in names]
    plt.bar(names, r2s, color=colors)
    plt.ylabel("R² score (test set)")
    plt.title("Model Comparison — Crop Yield Prediction")
    plt.ylim(0, 1)
    for i, v in enumerate(r2s):
        plt.text(i, v + 0.02, f"{v:.3f}", ha="center", fontweight="bold")
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/model_comparison.png", dpi=150)
    plt.close()

    # --- Plot 2: feature importance (best tree-based model) ---
    model_step = best_pipeline.named_steps["model"]
    if hasattr(model_step, "feature_importances_"):
        cat_names = list(
            best_pipeline.named_steps["prep"]
            .named_transformers_["cat"]
            .get_feature_names_out(CATEGORICAL_FEATURES)
        )
        all_feature_names = NUMERIC_FEATURES + cat_names
        importances = model_step.feature_importances_
        imp_df = pd.DataFrame({
            "feature": all_feature_names, "importance": importances
        }).sort_values("importance", ascending=True).tail(15)

        plt.figure(figsize=(8, 6))
        plt.barh(imp_df["feature"], imp_df["importance"], color="#2E7D32")
        plt.xlabel("Importance")
        plt.title(f"Top 15 Feature Importances — {best_name}")
        plt.tight_layout()
        plt.savefig(f"{MODEL_DIR}/feature_importance.png", dpi=150)
        plt.close()

    # --- Plot 3: predicted vs actual ---
    preds = best_pipeline.predict(X_test)
    plt.figure(figsize=(6, 6))
    plt.scatter(y_test, preds, alpha=0.3, s=15, color="#2E7D32")
    lims = [0, max(y_test.max(), preds.max()) + 0.5]
    plt.plot(lims, lims, "r--", linewidth=1.5, label="Perfect prediction")
    plt.xlabel("Actual yield (tons/ha)")
    plt.ylabel("Predicted yield (tons/ha)")
    plt.title(f"Predicted vs Actual — {best_name}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{MODEL_DIR}/predicted_vs_actual.png", dpi=150)
    plt.close()

    print(f"\nSaved model + plots to {MODEL_DIR}/")


if __name__ == "__main__":
    main()
