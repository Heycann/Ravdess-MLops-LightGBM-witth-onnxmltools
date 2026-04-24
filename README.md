# 🎙️ Audio-Based Emotion & Stress Detection (MLOps Pipeline)

An end-to-end Machine Learning operations (MLOps) pipeline designed to detect and classify human stress and emotion levels directly from raw audio signals. This project implements a modern MLOps architecture, ensuring reproducibility, automated experiment tracking, and production-readiness for integration into game engines or real-time applications.

## ✨ Key Features
* **Advanced Acoustic Feature Extraction:** Extracts **F0 (Pitch)** and **MFCC** from audio using `librosa`. The F0 feature is crucial for capturing the sudden spikes in vocal cord frequency when a person experiences panic or high stress.
* **High-Speed Classification:** Utilizes a **LightGBM** model, achieving **>95% accuracy** with sub-millisecond inference latency (~0.033 ms per sample).
* **MLOps & Automated Orchestration:** Fully orchestrated using **DVC (Data Version Control)** to ensure a reproducible data processing and training pipeline.
* **Experiment Tracking:** Integrated with **MLflow** to automatically log model metrics (Accuracy, F1-Score, Precision, Recall), hyperparameters, and confusion matrices in the background.
* **Production-Ready (ONNX):** The final model is exported to the industry-standard **ONNX** format, allowing seamless integration via C++/C# into game engines like Unreal Engine (e.g., for dynamic Sanity Systems or AI enemies that react to real-world player panic).

## 🗂️ Project Structure

```text
Project-PDM/
├── data/
│   └── processed/         # Processed data (e.g., train.parquet)
├── models/
│   ├── lgbm_model.pkl     # Original Scikit-Learn/LightGBM model
│   └── lgbm_model.onnx    # Production-ready ONNX model
├── src/
│   ├── preprocess.py      # Audio processing & feature extraction script
│   └── train.py           # Model training & MLflow logging script
├── .gitignore             # Ignored files configuration
├── dvc.yaml               # DVC pipeline configuration
├── README.md              # This project documentation
└── requirements.txt       # Python dependencies
