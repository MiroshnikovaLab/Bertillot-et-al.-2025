"""Reusable helpers for the tumor architecture classifier and prediction notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Union

import joblib
import numpy as np
import pandas as pd


PathLike = Union[str, Path]


def load_classifier_artifacts(classifier_dir: PathLike):
    """Load classifier, scaler, and label encoder using canonical filenames."""
    classifier_dir = Path(classifier_dir)
    model = joblib.load(classifier_dir / "model.joblib")
    scaler = joblib.load(classifier_dir / "scaler.joblib")
    label_encoder = joblib.load(classifier_dir / "label_encoder.joblib")
    return model, scaler, label_encoder


def save_classifier_artifacts(model, scaler, label_encoder, classifier_dir: PathLike) -> None:
    """Save classifier artifacts using canonical filenames."""
    classifier_dir = Path(classifier_dir)
    classifier_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, classifier_dir / "model.joblib")
    joblib.dump(scaler, classifier_dir / "scaler.joblib")
    joblib.dump(label_encoder, classifier_dir / "label_encoder.joblib")


def predict_embedding_probabilities(
    embedding_json_path: PathLike,
    classifier_model,
    scaler,
    label_encoder,
) -> pd.DataFrame:
    """Predict architecture probabilities from one embedding JSON file."""
    with Path(embedding_json_path).open("r", encoding="utf-8") as handle:
        data = json.load(handle)

    label_names = label_encoder.inverse_transform(classifier_model.classes_)
    rows = []
    for value in data.values():
        features = np.array(value["embedding"]).reshape(1, -1)
        features_scaled = scaler.transform(features)
        probabilities = classifier_model.predict_proba(features_scaled)[0]
        probabilities_safe = probabilities + 1e-12
        row = {
            "x": value["x"],
            "y": value["y"],
            "shannon_entropy": float(-np.sum(probabilities_safe * np.log2(probabilities_safe))),
        }
        for label_name, probability in zip(label_names, probabilities):
            row[f"prob_{label_name}"] = float(probability)
        rows.append(row)

    return pd.DataFrame(rows)


def batch_predict_embedding_probabilities(
    embedding_paths: Iterable[PathLike],
    classifier_model,
    scaler,
    label_encoder,
    output_dir: PathLike,
) -> None:
    """Write one HDF5 probability table per embedding JSON file."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for embedding_path in embedding_paths:
        embedding_path = Path(embedding_path)
        image_name = embedding_path.name.replace("_embedding.json", "")
        output_path = output_dir / f"{image_name}_probabilities.h5"
        df_pred = predict_embedding_probabilities(embedding_path, classifier_model, scaler, label_encoder)
        df_pred.to_hdf(output_path, key="df", mode="w")
