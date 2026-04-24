import os
import sys
import time
import argparse
import numpy as np
import pandas as pd
import lightgbm as lgb
import mlflow
import mlflow.sklearn
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import onnxmltools
from onnxmltools import convert_lightgbm
from onnxmltools.convert.common.data_types import FloatTensorType
import joblib

# initial_type defined later once training data shape is available
onnx_model = onnxmltools.utils.load_model("lgbm_model.onnx") if os.path.exists("lgbm_model.onnx") else None

def log_cm(cm, classes, tag=""):
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(6, 5))
    disp.plot(ax=ax, cmap='Blues')
    ax.set_title(f"Confusion Matrix ({tag})")
    plt.tight_layout()
    path = f"cm_{tag}.png"
    plt.savefig(path)
    plt.close()
    return path

def main():
    parser = argparse.ArgumentParser(description="DVC-compatible training pipeline")
    parser.add_argument("--data-dir", default="data/processed")
    parser.add_argument("--model-dir", default="models")
    parser.add_argument("--mlflow-experiment", default="lgbm_pipeline")
    # UBAH: Target sekarang disesuaikan dengan output preprocess kita
    parser.add_argument("--target-col", default="stress_label")
    args = parser.parse_args()

    os.makedirs(args.model_dir, exist_ok=True)

    # Load Data
    print("Memuat dataset dari Parquet...")
    train_path = os.path.join(args.data_dir, "train.parquet")
    val_path   = os.path.join(args.data_dir, "val.parquet")
    test_path  = os.path.join(args.data_dir, "test.parquet")

    train_df = pd.read_parquet(train_path)
    val_df   = pd.read_parquet(val_path) if os.path.exists(val_path) else None
    test_df  = pd.read_parquet(test_path) if os.path.exists(test_path) else None

    # Auto-mapping Features
    # UBAH: Mengambil semua kolom yang diawali dengan 'feat_' secara spesifik
    feature_cols = [c for c in train_df.columns if c.startswith('feat_')]
    print(f"Ditemukan {len(feature_cols)} fitur untuk training.")

    X_train, y_train = train_df[feature_cols].values, train_df[args.target_col].values
    X_val,   y_val   = val_df[feature_cols].values,   val_df[args.target_col].values if val_df is not None else None
    X_test,  y_test  = test_df[feature_cols].values,  test_df[args.target_col].values if test_df is not None else None

    # Class Weight Handling
    unique, counts = np.unique(y_train, return_counts=True)
    scale_pos_weight = None
    if len(unique) == 2:
        scale_pos_weight = float(counts[0] / counts[1])
        print(f"Binary classification detected. scale_pos_weight: {scale_pos_weight:.3f}")
    else:
        print("Multiclass detected. LGBM akan menangani bobot kelas secara internal.")

    # MLflow Setup
    mlflow.set_experiment(args.mlflow_experiment)
    with mlflow.start_run(run_name="lgbm_train_run") as run:
        run_id = run.info.run_id
        print(f"MLflow Run ID: {run_id}")

        # Log Hyperparameters
        params = {
            "n_estimators": 500,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "max_depth": -1,
            "feature_fraction": 0.8,
            "bagging_fraction": 0.8,
            "bagging_freq": 5,
            "scale_pos_weight": scale_pos_weight,
            "early_stopping_rounds": 20,
            "feature_count": len(feature_cols)
        }
        mlflow.log_params(params)

        # Train LightGBM
        print("Training LightGBM...")
        model = lgb.LGBMClassifier(
            n_estimators=params["n_estimators"],
            learning_rate=params["learning_rate"],
            num_leaves=params["num_leaves"],
            max_depth=params["max_depth"],
            feature_fraction=params["feature_fraction"],
            bagging_fraction=params["bagging_fraction"],
            bagging_freq=params["bagging_freq"],
            is_unbalance=(scale_pos_weight is not None),
            scale_pos_weight=scale_pos_weight if scale_pos_weight else 1,
            n_jobs=-1,
            verbose=-1,
            random_state=42
        )

        eval_set = [(X_val, y_val)] if X_val is not None else None
        
        # UBAH: Format penulisan early_stopping disesuaikan dengan LGBM versi terbaru
        callbacks = [lgb.early_stopping(stopping_rounds=params["early_stopping_rounds"])] if eval_set else None
        
        model.fit(
            X_train, y_train,
            eval_set=eval_set,
            callbacks=callbacks
        )

        # Evaluation & Logging
        print("Evaluating model...")
        splits = {"train": (X_train, y_train), "val": (X_val, y_val), "test": (X_test, y_test)}
        for name, (X, y) in splits.items():
            if X is None or y is None: continue
            y_pred = model.predict(X)
            
            metrics = {
                f"{name}_accuracy": accuracy_score(y, y_pred),
                f"{name}_f1": f1_score(y, y_pred, average="macro"),
                f"{name}_precision": precision_score(y, y_pred, average="macro"),
                f"{name}_recall": recall_score(y, y_pred, average="macro")
            }
            mlflow.log_metrics(metrics)
            print(f"{name.upper()} Metrics: {metrics}")

            cm = confusion_matrix(y, y_pred)
            cm_path = log_cm(cm, unique, name)
            mlflow.log_artifact(cm_path)
            os.remove(cm_path)

        # Latency Benchmark
        print("Benchmarking inference latency...")
        n_samples = min(1000, len(X_val) if X_val is not None else len(X_train))
        X_bench = X_val[:n_samples] if X_val is not None else X_train[:n_samples]
        
        start = time.perf_counter()
        model.predict(X_bench)
        latency_ms = (time.perf_counter() - start) / n_samples * 1000
        mlflow.log_metric("latency_ms_per_sample", latency_ms)
        print(f"Average latency: {latency_ms:.3f} ms/sample")

        # Checkpoint & Model Export
        model_path = os.path.join(args.model_dir, "lgbm_model.pkl")
        joblib.dump(model, model_path)
        mlflow.sklearn.log_model(sk_model=model, name="sklearn_model")
        
        # ONNX Export
        print("Benchmarking inference latency...")
        onnx_path = os.path.join(args.model_dir, "lgbm_model.onnx")
        initial_type = [('float_input', FloatTensorType([None, X_train.shape[1]]))]
        
        # UBAH convert_sklearn MENJADI convert_lightgbm
        onnx_model = convert_lightgbm(model, initial_types=initial_type)
        
        with open(onnx_path, "wb") as f:
            f.write(onnx_model.SerializeToString())
        mlflow.log_artifact(onnx_path)
        print("Model saved, checkpointed & exported to ONNX!")

    print("Training pipeline completed successfully!")

if __name__ == "__main__":
    main()