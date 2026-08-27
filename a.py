"""
AgricultureCropProductionPrediction.py

Project   : Prediction of Agriculture Crop Production in India
Internship: upskill Campus x The IoT Academy x UniConverge Technologies (UCT)

Description
-----------
Trains and compares regression models to predict crop production (in tonnes)
across Indian states using historical crop data (2001-2014). The pipeline:

    1. Load data
    2. Clean data (handle missing values, standardize text fields)
    3. Feature engineering (year-over-year growth, region/season aggregates)
    4. Encode categorical variables
    5. Train & compare Linear Regression, Decision Tree, Random Forest
    6. Validate the best model with cross-validation on a held-out test set
    7. Report feature importance and save the final model

Expected input CSV columns (rename to match your dataset if different):
    State_Name, District_Name, Crop_Year, Season, Crop, Area, Production
"""

import warnings

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
DATA_PATH = "crop_production.csv"          # <-- update to your dataset path
MODEL_OUTPUT_PATH = "crop_production_random_forest.joblib"


# --------------------------------------------------------------------------
# 1. Load data
# --------------------------------------------------------------------------
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
    return df


# --------------------------------------------------------------------------
# 2. Clean data
# --------------------------------------------------------------------------
def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Standardize text fields
    for col in ["State_Name", "District_Name", "Season", "Crop"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip().str.title()

    # Drop rows with no target value; Production can legitimately be missing
    # for some crop/season combinations, so those rows are not usable for training.
    df = df.dropna(subset=["Production"])

    # Fill missing Area with the median for that crop, as a reasonable estimate.
    if "Area" in df.columns:
        df["Area"] = df.groupby("Crop")["Area"].transform(
            lambda s: s.fillna(s.median())
        )

    # Remove non-physical rows (e.g. zero/negative area or production)
    df = df[(df["Area"] > 0) & (df["Production"] > 0)]

    print(f"After cleaning: {len(df):,} rows")
    return df


# --------------------------------------------------------------------------
# 3. Feature engineering
# --------------------------------------------------------------------------
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Yield as an auxiliary signal (production per unit area)
    df["Yield"] = df["Production"] / df["Area"]

    # Year-over-year growth in average production, per crop
    yearly_avg = (
        df.groupby(["Crop", "Crop_Year"])["Production"].mean().reset_index()
    )
    yearly_avg = yearly_avg.sort_values(["Crop", "Crop_Year"])
    yearly_avg["Prod_YoY_Growth"] = yearly_avg.groupby("Crop")["Production"].pct_change()
    df = df.merge(
        yearly_avg[["Crop", "Crop_Year", "Prod_YoY_Growth"]],
        on=["Crop", "Crop_Year"],
        how="left",
    )
    df["Prod_YoY_Growth"] = df["Prod_YoY_Growth"].fillna(0)

    # Region (state) and season aggregate averages
    df["State_Avg_Production"] = df.groupby("State_Name")["Production"].transform("mean")
    df["Season_Avg_Production"] = df.groupby("Season")["Production"].transform("mean")

    return df


# --------------------------------------------------------------------------
# 4. Encode categorical variables
# --------------------------------------------------------------------------
def encode_features(df: pd.DataFrame, categorical_cols):
    df = df.copy()
    encoders = {}
    for col in categorical_cols:
        le = LabelEncoder()
        df[col + "_enc"] = le.fit_transform(df[col])
        encoders[col] = le
    return df, encoders


# --------------------------------------------------------------------------
# 5. Train & compare models
# --------------------------------------------------------------------------
def train_and_compare(X_train, X_test, y_train, y_test):
    models = {
        "Linear Regression": LinearRegression(),
        "Decision Tree": DecisionTreeRegressor(
            max_depth=12, min_samples_leaf=5, random_state=RANDOM_STATE
        ),
        "Random Forest": RandomForestRegressor(
            n_estimators=300,
            max_depth=16,
            min_samples_leaf=3,
            random_state=RANDOM_STATE,
            n_jobs=-1,
        ),
    }

    results = {}
    for name, model in models.items():
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        r2 = r2_score(y_test, preds)
        results[name] = {"model": model, "rmse": rmse, "r2": r2}
        print(f"{name:>18s} | RMSE: {rmse:,.2f} | R2: {r2:.4f}")

    return results


# --------------------------------------------------------------------------
# 6. Validate best model with cross-validation
# --------------------------------------------------------------------------
def cross_validate_best(model, X, y, cv=5):
    scores = cross_val_score(model, X, y, cv=cv, scoring="r2", n_jobs=-1)
    print(f"Cross-validated R2 ({cv}-fold): mean={scores.mean():.4f}, std={scores.std():.4f}")
    return scores


# --------------------------------------------------------------------------
# 7. Feature importance
# --------------------------------------------------------------------------
def plot_feature_importance(model, feature_names, top_n=10, out_path="feature_importance.png"):
    importances = pd.Series(model.feature_importances_, index=feature_names)
    importances = importances.sort_values(ascending=False).head(top_n)

    plt.figure(figsize=(8, 5))
    importances.sort_values().plot(kind="barh")
    plt.title("Top Feature Importances - Random Forest")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    print(f"Saved feature importance plot to {out_path}")


# --------------------------------------------------------------------------
# Main pipeline
# --------------------------------------------------------------------------
def main():
    df = load_data(DATA_PATH)
    df = clean_data(df)
    df = engineer_features(df)

    categorical_cols = ["State_Name", "Season", "Crop"]
    df, encoders = encode_features(df, categorical_cols)

    feature_cols = [
        "Area",
        "Crop_Year",
        "Prod_YoY_Growth",
        "State_Avg_Production",
        "Season_Avg_Production",
    ] + [c + "_enc" for c in categorical_cols]

    X = df[feature_cols]
    y = df["Production"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    results = train_and_compare(X_train, X_test, y_train, y_test)

    best_name = min(results, key=lambda k: results[k]["rmse"])
    best_model = results[best_name]["model"]
    print(f"\nBest model: {best_name}")

    cross_validate_best(best_model, X, y, cv=5)

    if hasattr(best_model, "feature_importances_"):
        plot_feature_importance(best_model, feature_cols)

    joblib.dump({"model": best_model, "encoders": encoders, "features": feature_cols}, MODEL_OUTPUT_PATH)
    print(f"Saved final model to {MODEL_OUTPUT_PATH}")


if __name__ == "__main__":
    main()