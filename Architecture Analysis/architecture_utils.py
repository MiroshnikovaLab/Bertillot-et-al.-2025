"""Reusable helpers for tumor architecture embedding and prediction notebooks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Optional, Tuple, Union

import joblib
import numpy as np
import pandas as pd
from skimage import io
from tensorflow.keras.layers import GlobalAveragePooling2D
from tensorflow.keras.models import Model

from UNet_utils import normalizePercentile


PathLike = Union[str, Path]


def make_feature_extractor(model, feature_layer_name: Optional[str] = None) -> Model:
    """Build a model that returns globally pooled bottleneck features."""
    if feature_layer_name is None:
        feature_layer = model.layers[-2].output
    else:
        feature_layer = model.get_layer(feature_layer_name).output
    pooled = GlobalAveragePooling2D()(feature_layer)
    return Model(inputs=model.input, outputs=pooled)


def extract_tile_embedding(feature_extractor: Model, tile: np.ndarray) -> np.ndarray:
    """Normalize one 2D tile and return a flat feature vector."""
    tile = normalizePercentile(tile)
    tile = np.asarray(tile, dtype=np.float32)
    if tile.ndim != 2:
        raise ValueError("Expected a 2D grayscale tile.")
    model_input = tile[np.newaxis, ..., np.newaxis]
    return np.asarray(feature_extractor.predict(model_input, verbose=0)[0], dtype=float)


def iter_tile_coordinates(
    image_shape: Tuple[int, int],
    tile_size: int,
    stride: Optional[int] = None,
) -> list[Tuple[int, int]]:
    """Return top-left tile coordinates covering an image."""
    height, width = image_shape
    stride = stride or tile_size
    coords = []
    for row in range(0, max(height - tile_size + 1, 1), stride):
        for col in range(0, max(width - tile_size + 1, 1), stride):
            coords.append((row, col))
    if height >= tile_size and width >= tile_size:
        last_row = height - tile_size
        last_col = width - tile_size
        edge_coords = {(row, col) for row, col in coords}
        for row in sorted({coord[0] for coord in coords} | {last_row}):
            edge_coords.add((row, last_col))
        for col in sorted({coord[1] for coord in coords} | {last_col}):
            edge_coords.add((last_row, col))
        coords = sorted(edge_coords)
    return coords


def collect_image_embeddings(
    image_path: PathLike,
    feature_extractor: Model,
    tile_size: int,
    stride: Optional[int] = None,
) -> dict[str, dict[str, object]]:
    """Extract feature embeddings for all tiles in one image."""
    image = io.imread(image_path, as_gray=True)
    coords = iter_tile_coordinates(image.shape, tile_size=tile_size, stride=stride)
    embeddings: dict[str, dict[str, object]] = {}
    for idx, (row, col) in enumerate(coords):
        tile = image[row : row + tile_size, col : col + tile_size]
        if tile.shape != (tile_size, tile_size):
            padded = np.zeros((tile_size, tile_size), dtype=tile.dtype)
            padded[: tile.shape[0], : tile.shape[1]] = tile
            tile = padded
        embeddings[str(idx)] = {
            "x": int(row),
            "y": int(col),
            "embedding": extract_tile_embedding(feature_extractor, tile).tolist(),
        }
    return embeddings


def save_embedding_json(embeddings: dict[str, dict[str, object]], output_path: PathLike) -> None:
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(embeddings), encoding="utf-8")


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
