import os
import sys
import time
import warnings
import pickle
warnings.filterwarnings("ignore")
 
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

# Configuration
ML_OUTPUT      = r"C:\Users\user\Desktop\Energy Consumption\spark_work\output\ml"
FEATURE_PATH   = os.path.join(ML_OUTPUT, "features", "feature_table.parquet")
MODEL_OUTPUT   = os.path.join(ML_OUTPUT, "models")
 
# Train on first 80% of dates, test on last 20% (time-based split)
TRAIN_RATIO    = 0.80
RANDOM_STATE   = 42
 
SEP = "=" * 72

# Loading feature table 
def load_features() -> pd.DataFrame:
    print(f"\n[LOAD] Feature table from {FEATURE_PATH} ...")
    if not os.path.isfile(FEATURE_PATH):
        print(f"[ERROR] Not found: {FEATURE_PATH}")
        print("        Run m4_01_feature_engineering.py first.")
        sys.exit(1)
 
    df = pq.read_table(FEATURE_PATH).to_pandas()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df.sort_values("Date").reset_index(drop=True)
    print(f"  {len(df):,} rows | {len(df.columns)} columns")
    print(f"  Date range: {df['Date'].min().date()} → {df['Date'].max().date()}")
    return df

# Feature selection
def select_features(df: pd.DataFrame):
    exclude = {
        "Date", "total_kwh", "avg_kw", "peak_kw",
        "minutes_recorded", "year",
        "kitchen_kwh", "laundry_kwh", "hvac_kwh", "other_kwh",
    }
 
    feature_cols = [c for c in df.columns if c not in exclude
                    and df[c].dtype in [np.float64, np.int64, np.float32, np.int32]
                    and not df[c].isna().all()]
 
    X = df[feature_cols].copy()
    y = df["total_kwh"].copy()
 
    # Fill remaining NaN with column median (affects < 2% of rows)
    X = X.fillna(X.median())
 
    print(f"\n  Feature matrix : {X.shape[0]:,} rows × {X.shape[1]} features")
    print(f"  Target         : total_kwh  "
          f"(mean={y.mean():.2f}, std={y.std():.2f}, "
          f"min={y.min():.2f}, max={y.max():.2f})")
 
    return X, y, feature_cols

# Train-test split
def time_split(df: pd.DataFrame, X: pd.DataFrame, y: pd.Series, ratio: float):
    n_train      = int(len(df) * ratio)
    split_date   = df["Date"].iloc[n_train]
 
    X_train, X_test = X.iloc[:n_train], X.iloc[n_train:]
    y_train, y_test = y.iloc[:n_train], y.iloc[n_train:]
 
    print(f"\n  Train set: {len(X_train):,} days  "
          f"({df['Date'].iloc[0].date()} → {df['Date'].iloc[n_train-1].date()})")
    print(f"  Test set : {len(X_test):,} days  "
          f"({split_date.date()} → {df['Date'].iloc[-1].date()})")
    print(f"  Split ratio: {ratio*100:.0f}% / {(1-ratio)*100:.0f}%")
 
    return X_train, X_test, y_train, y_test, split_date

# Metrics
def compute_metrics(y_true, y_pred, label: str) -> dict:
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
 
    mae  = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / np.maximum(np.abs(y_true), 0.1))) * 100
 
    print(f"\n  [{label}] Evaluation Metrics:")
    print(f"    MAE  : {mae:.4f} kWh  (avg absolute error per day)")
    print(f"    RMSE : {rmse:.4f} kWh  (penalises large errors)")
    print(f"    R²   : {r2:.4f}      (1.0 = perfect fit)")
    print(f"    MAPE : {mape:.2f}%    (scale-independent error)")
 
    return {"model": label, "MAE": mae, "RMSE": rmse, "R2": r2, "MAPE": mape}

# Model A: Random Forest
def train_random_forest(X_train, y_train):
    from sklearn.ensemble import RandomForestRegressor
 
    print("\n  Training Random Forest ...")
    print(f"  Config: 200 trees | max_depth=12 | n_jobs=-1 (all cores)")
 
    t0  = time.time()
    rf  = RandomForestRegressor(
        n_estimators   = 200,
        max_depth      = 12,
        min_samples_leaf = 5,
        max_features   = "sqrt",
        n_jobs         = -1,
        random_state   = RANDOM_STATE,
    )
    rf.fit(X_train, y_train)
    elapsed = time.time() - t0
    print(f" Trained in {elapsed:.1f}s")
    return rf

# Model B: Gradient Boosting
def train_gradient_boosting(X_train, y_train):
    from sklearn.ensemble import HistGradientBoostingRegressor
 
    print("\n  Training Gradient Boosting (HistGBM) ...")
    print(f"  Config: 300 estimators | lr=0.05 | early stopping | n_jobs=-1")
 
    t0  = time.time()
    gbm = HistGradientBoostingRegressor(
        max_iter           = 300,
        learning_rate      = 0.05,
        max_depth          = 5,
        min_samples_leaf   = 10,
        l2_regularization  = 0.1,
        early_stopping     = True,
        validation_fraction = 0.1,
        n_iter_no_change   = 20,
        random_state       = RANDOM_STATE,
    )
    gbm.fit(X_train, y_train)
    elapsed = time.time() - t0
    n_iters = gbm.n_iter_
    print(f" Trained in {elapsed:.1f}s | Early stopped at iteration {n_iters}")
    return gbm

# Cross validation
def time_series_cross_validation(model, X, y, n_splits: int = 5):
    from sklearn.model_selection import TimeSeriesSplit
    from sklearn.metrics import mean_absolute_error, r2_score
 
    print(f"\n  Time-series cross-validation ({n_splits} folds) ...")
    tscv   = TimeSeriesSplit(n_splits=n_splits)
    scores = {"MAE": [], "R2": []}
 
    for fold, (train_idx, test_idx) in enumerate(tscv.split(X), 1):
        X_tr, X_te = X.iloc[train_idx], X.iloc[test_idx]
        y_tr, y_te = y.iloc[train_idx], y.iloc[test_idx]
 
        X_tr_filled = X_tr.fillna(X_tr.median())
        X_te_filled = X_te.fillna(X_tr.median())
 
        model.fit(X_tr_filled, y_tr)
        preds = model.predict(X_te_filled)
 
        mae = mean_absolute_error(y_te, preds)
        r2  = r2_score(y_te, preds)
        scores["MAE"].append(mae)
        scores["R2"].append(r2)
        print(f"    Fold {fold}: MAE={mae:.3f}  R²={r2:.4f}  "
              f"(train={len(X_tr):,}, test={len(X_te):,})")
 
    print(f"  CV MAE : {np.mean(scores['MAE']):.4f} ± {np.std(scores['MAE']):.4f}")
    print(f"  CV R²  : {np.mean(scores['R2']):.4f} ± {np.std(scores['R2']):.4f}")
    return scores

# SHAP explainability
def explain_with_shap(model, X_test: pd.DataFrame, feature_cols: list,model_name: str):
    try:
        import shap
 
        print(f"\n  SHAP explainability for {model_name} ...")
 
        # Sample 100 test points for speed (SHAP is O(n) for tree models)
        sample_size = min(100, len(X_test))
        X_sample    = X_test.sample(sample_size, random_state=RANDOM_STATE)
 
        explainer   = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_sample)
 
        # Mean absolute SHAP value per feature = global importance
        mean_shap = pd.Series(
            np.abs(shap_values).mean(axis=0),
            index=feature_cols
        ).sort_values(ascending=False)
 
        print(f"\n  Top 15 features by mean |SHAP| value ({model_name}):")
        print(f"  {'Feature':<28} {'Mean |SHAP|':>12} {'Direction':>10}")
        print(f"  {'─'*28} {'─'*12} {'─'*10}")
 
        for feat in mean_shap.head(15).index:
            mean_val = mean_shap[feat]
            # Direction: positive SHAP = increases prediction
            feat_idx   = feature_cols.index(feat)
            mean_dir   = np.mean(shap_values[:, feat_idx])
            direction  = "↑ increases" if mean_dir > 0 else "↓ decreases"
            print(f"  {feat:<28} {mean_val:>12.4f} {direction:>10}")
            return mean_shap
 
    except ImportError:
        print("\n  [SKIP] shap not installed — using built-in feature importance instead")
        return None
    except Exception as e:
        print(f"\n  [SKIP] SHAP failed: {e} — using built-in feature importance")
        return None
 
 
def print_builtin_importance(model, feature_cols: list, model_name: str,
                              top_n: int = 15):
    """Fallback: scikit-learn built-in feature importance (impurity-based)."""
    if not hasattr(model, "feature_importances_"):
        return
 
    importances = pd.Series(
        model.feature_importances_, index=feature_cols
    ).sort_values(ascending=False)
 
    print(f"\n  Top {top_n} features by importance ({model_name}):")
    print(f"  {'Feature':<28} {'Importance':>12}")
    print(f"  {'─'*28} {'─'*12}")
    for feat, val in importances.head(top_n).items():
        bar = "█" * int(val * 100)
        print(f"  {feat:<28} {val:>12.4f}  {bar}")

# Saving models
def save_model(model, name: str, metrics: dict, feature_cols: list):
    os.makedirs(MODEL_OUTPUT, exist_ok=True)
    path = os.path.join(MODEL_OUTPUT, f"{name}.pkl")
 
    payload = {
        "model"        : model,
        "feature_cols" : feature_cols,
        "metrics"      : metrics,
        "trained_at"   : time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    with open(path, "wb") as f:
        pickle.dump(payload, f)
 
    size_kb = os.path.getsize(path) / 1024
    print(f"  Saved: {path}  ({size_kb:.1f} KB)")

# Performance report
def print_performance_report(results: list, y_test: pd.Series):
    baseline_mae = np.abs(y_test - y_test.mean()).mean()
    print(f"""Baseline (predict mean every day): MAE = {baseline_mae:.4f} kWh""")
    for r in results:
        improvement = (1 - r["MAE"] / baseline_mae) * 100
        print(f"  │ {r['model']:<24} │ {r['MAE']:>8.4f} │ "
              f"{r['RMSE']:>8.4f} │ {r['R2']:>8.4f} │ {r['MAPE']:>7.2f}% │")

# Main execution
def main():
    t_start = time.time()
    df = load_features()
    X, y, feature_cols = select_features(df)
    X_train, X_test, y_train, y_test, split_date = time_split(df, X, y, TRAIN_RATIO) 
    rf  = train_random_forest(X_train, y_train)
    gbm = train_gradient_boosting(X_train, y_train)
    # Cross validation
    from sklearn.ensemble import RandomForestRegressor
    rf_cv = RandomForestRegressor(
        n_estimators=100, max_depth=10, n_jobs=-1, random_state=RANDOM_STATE
    )
    print("\n  Random Forest cross-validation:")
    cv_scores_rf  = time_series_cross_validation(rf_cv,  X, y, n_splits=5)
 
    from sklearn.ensemble import HistGradientBoostingRegressor
    gbm_cv = HistGradientBoostingRegressor(
        max_iter=100, learning_rate=0.05, random_state=RANDOM_STATE
    )
    print("\n  Gradient Boosting cross-validation:")
    cv_scores_gbm = time_series_cross_validation(gbm_cv, X, y, n_splits=5)
    # Evaluation
    rf_preds  = rf.predict(X_test)
    gbm_preds = gbm.predict(X_test)
 
    rf_metrics  = compute_metrics(y_test, rf_preds,  "Random Forest")
    gbm_metrics = compute_metrics(y_test, gbm_preds, "Gradient Boosting")

    # Sample predictions
    print(f"\n  Sample predictions (first 10 test days):")
    print(f"  {'Date':<12} {'Actual':>8} {'RF Pred':>8} "
          f"{'GBM Pred':>9} {'RF Err':>8} {'GBM Err':>8}")
    print(f"  {'─'*12} {'─'*8} {'─'*8} {'─'*9} {'─'*8} {'─'*8}")
 
    test_dates = df["Date"].iloc[len(X_train):len(X_train)+10]
    for i, (date, actual, rf_p, gbm_p) in enumerate(zip(
        test_dates, y_test.values[:10], rf_preds[:10], gbm_preds[:10]
    )):
        print(f"  {str(date.date()):<12} {actual:>8.3f} {rf_p:>8.3f} "
              f"{gbm_p:>9.3f} {actual-rf_p:>+8.3f} {actual-gbm_p:>+8.3f}")
        
    # SHAP explainability
    shap_rf = explain_with_shap(rf, X_test, feature_cols, "Random Forest")
    if shap_rf is None:
        print_builtin_importance(rf, feature_cols, "Random Forest")
 
    print("\n  Gradient Boosting — Feature Importance:")
    shap_gbm = explain_with_shap(gbm, X_test, feature_cols, "Gradient Boosting")
    if shap_gbm is None:
        print_builtin_importance(gbm, feature_cols, "Gradient Boosting")
    
    # Save models
    save_model(rf,  "random_forest",       rf_metrics,  feature_cols)
    save_model(gbm, "gradient_boosting",   gbm_metrics, feature_cols)
 
    # Also save the best model separately for easy loading in Milestone 5
    best = rf if rf_metrics["R2"] >= gbm_metrics["R2"] else gbm
    best_name = "random_forest" if rf_metrics["R2"] >= gbm_metrics["R2"] \
                else "gradient_boosting"
    save_model(best, "best_model", 
               rf_metrics if best_name == "random_forest" else gbm_metrics,
               feature_cols)
    print(f"\n  Best model: {best_name}  (R²={max(rf_metrics['R2'], gbm_metrics['R2']):.4f})")
    elapsed = time.time() - t_start

if __name__ == "__main__":
    main()




