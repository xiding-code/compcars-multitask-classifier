"""CompCars multi-task vehicle make/model classifier.

Multi-task ResNet50V2 trained on the CompCars Web-Nature subset (Kaggle mirror)
to jointly predict vehicle make and model from a single image. Used as a
listing-verification signal for used-car marketplaces.

Team 9 (BU Questrom BA865, Spring 2026): Tianqi Sun, Yanlun Li, Xincheng Ding
"""
from __future__ import annotations

import collections
import json
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow import keras


TARGET_SIZE = (299, 299)
VALID_YEAR_RANGE = (1990, 2024)
AUTOTUNE = tf.data.AUTOTUNE
PAD_VALUE = 114.0


def find_dataset_root(base: Path) -> Path:
    if (base / "image").is_dir() and (base / "train_test_split").is_dir():
        return base

    if (base / "CompCars_raw").is_dir():
        nested = base / "CompCars_raw"
        if (nested / "image").is_dir() and (nested / "train_test_split").is_dir():
            return nested

    for child in sorted(base.iterdir()):
        if child.is_dir() and (child / "image").is_dir() and (child / "train_test_split").is_dir():
            return child

    raise RuntimeError(
        f"Could not locate a valid CompCars root under {base}. "
        "Expected image/ and train_test_split/ directories."
    )


def is_valid_year(year_str: str, valid_year_range: tuple[int, int] = VALID_YEAR_RANGE) -> bool:
    try:
        year = int(year_str)
    except ValueError:
        return False
    return valid_year_range[0] <= year <= valid_year_range[1]


def parse_label_file(label_path: Path) -> dict[str, object]:
    result = {
        "label_path": str(label_path),
        "view_label": None,
        "object_flag": None,
        "bbox_x1": np.nan,
        "bbox_y1": np.nan,
        "bbox_x2": np.nan,
        "bbox_y2": np.nan,
        "has_bbox": False,
    }

    if not label_path.exists():
        return result

    lines = [line.strip() for line in label_path.read_text().splitlines() if line.strip()]
    if len(lines) >= 1:
        try:
            result["view_label"] = int(lines[0])
        except ValueError:
            result["view_label"] = lines[0]
    if len(lines) >= 2:
        try:
            result["object_flag"] = int(lines[1])
        except ValueError:
            result["object_flag"] = lines[1]
    if len(lines) >= 3:
        parts = lines[2].split()
        if len(parts) == 4:
            try:
                x1, y1, x2, y2 = [float(value) for value in parts]
            except ValueError:
                pass
            else:
                result.update(
                    {
                        "bbox_x1": x1,
                        "bbox_y1": y1,
                        "bbox_x2": x2,
                        "bbox_y2": y2,
                        "has_bbox": True,
                    }
                )
    return result


def scan_compcars_metadata(
    dataset_root: Path,
    valid_year_range: tuple[int, int] = VALID_YEAR_RANGE,
    include_bbox: bool = True,
) -> pd.DataFrame:
    dataset_root = find_dataset_root(Path(dataset_root))
    image_root = dataset_root / "image"
    label_root = dataset_root / "label"

    rows: list[dict[str, object]] = []
    for make_dir in sorted(image_root.iterdir()):
        if not make_dir.is_dir():
            continue
        for model_dir in sorted(make_dir.iterdir()):
            if not model_dir.is_dir():
                continue
            for year_dir in sorted(model_dir.iterdir()):
                if not year_dir.is_dir():
                    continue
                year_name = year_dir.name
                if not is_valid_year(year_name, valid_year_range=valid_year_range):
                    continue
                for image_path in sorted(year_dir.glob("*.jpg")):
                    rel_path = image_path.relative_to(image_root)
                    label_path = (label_root / rel_path).with_suffix(".txt")
                    label_meta = parse_label_file(label_path) if include_bbox else {
                        "label_path": str(label_path),
                        "view_label": None,
                        "object_flag": None,
                        "bbox_x1": np.nan,
                        "bbox_y1": np.nan,
                        "bbox_x2": np.nan,
                        "bbox_y2": np.nan,
                        "has_bbox": False,
                    }
                    rows.append(
                        {
                            "image_path": str(image_path),
                            "relative_path": str(rel_path),
                            "make_id": make_dir.name,
                            "model_id": model_dir.name,
                            "model_key": f"{make_dir.name}/{model_dir.name}",
                            "year": int(year_name),
                            "image_name": image_path.name,
                            **label_meta,
                        }
                    )

    return pd.DataFrame(rows)


def filter_models(
    df: pd.DataFrame,
    min_years: int = 3,
    min_images_per_model: int = 1,
) -> pd.DataFrame:
    model_years = df.groupby("model_key")["year"].nunique()
    model_images = df.groupby("model_key").size()
    keep = model_years[
        (model_years >= min_years)
        & (model_images.reindex(model_years.index) >= min_images_per_model)
    ].index
    return df[df["model_key"].isin(keep)].reset_index(drop=True)


def build_training_dataframe(
    dataset_root: Path,
    min_years: int = 3,
    min_images_per_model: int = 1,
    valid_year_range: tuple[int, int] = VALID_YEAR_RANGE,
    include_bbox: bool = True,
) -> pd.DataFrame:
    df = scan_compcars_metadata(
        dataset_root=dataset_root,
        valid_year_range=valid_year_range,
        include_bbox=include_bbox,
    )
    df = filter_models(
        df,
        min_years=min_years,
        min_images_per_model=min_images_per_model,
    )
    return df.reset_index(drop=True)


def summarize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    summary = {
        "rows": len(df),
        "unique_makes": df["make_id"].nunique(),
        "unique_models": df["model_key"].nunique(),
        "min_year": int(df["year"].min()),
        "max_year": int(df["year"].max()),
        "bbox_available_fraction": float(df["has_bbox"].mean()),
    }
    return pd.DataFrame([summary])


def split_dataframe(
    df: pd.DataFrame,
    val_fraction: float = 0.15,
    test_fraction: float = 0.15,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_fraction,
        random_state=random_state,
        stratify=df["model_key"],
    )

    val_ratio_within_train = val_fraction / (1.0 - test_fraction)
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_ratio_within_train,
        random_state=random_state,
        stratify=train_val_df["model_key"],
    )

    for frame, split_name in ((train_df, "train"), (val_df, "val"), (test_df, "test")):
        frame["split"] = split_name

    return train_df.reset_index(drop=True), val_df.reset_index(drop=True), test_df.reset_index(drop=True)


def build_encoders(*frames: pd.DataFrame) -> tuple[dict[str, int], dict[str, int]]:
    merged = pd.concat(frames, ignore_index=True)
    make_encoder = {label: idx for idx, label in enumerate(sorted(merged["make_id"].unique()))}
    model_encoder = {label: idx for idx, label in enumerate(sorted(merged["model_key"].unique()))}
    return make_encoder, model_encoder


def apply_encoders(
    frame: pd.DataFrame,
    make_encoder: dict[str, int],
    model_encoder: dict[str, int],
) -> pd.DataFrame:
    frame = frame.copy()
    frame["make_label"] = frame["make_id"].map(make_encoder).astype("int32")
    frame["model_label"] = frame["model_key"].map(model_encoder).astype("int32")
    return frame


def add_model_sample_weights(frame: pd.DataFrame) -> pd.DataFrame:
    frame = frame.copy()
    counts = frame["model_key"].value_counts()
    frame["model_sample_weight"] = frame["model_key"].map(lambda key: 1.0 / counts[key])
    frame["model_sample_weight"] *= len(frame) / frame["model_sample_weight"].sum()
    return frame


def _crop_with_bbox(image: tf.Tensor, bbox: tf.Tensor, has_bbox: tf.Tensor) -> tf.Tensor:
    def _crop() -> tf.Tensor:
        shape = tf.shape(image)
        height = shape[0]
        width = shape[1]
        x1 = tf.cast(tf.clip_by_value(tf.round(bbox[0]), 0, tf.cast(width - 1, tf.float32)), tf.int32)
        y1 = tf.cast(tf.clip_by_value(tf.round(bbox[1]), 0, tf.cast(height - 1, tf.float32)), tf.int32)
        x2 = tf.cast(tf.clip_by_value(tf.round(bbox[2]), 1, tf.cast(width, tf.float32)), tf.int32)
        y2 = tf.cast(tf.clip_by_value(tf.round(bbox[3]), 1, tf.cast(height, tf.float32)), tf.int32)
        crop_w = tf.maximum(x2 - x1, 1)
        crop_h = tf.maximum(y2 - y1, 1)
        return tf.image.crop_to_bounding_box(image, y1, x1, crop_h, crop_w)

    return tf.cond(tf.cast(has_bbox, tf.bool), _crop, lambda: image)


def letterbox_resize(
    image: tf.Tensor,
    target_size: tuple[int, int] = TARGET_SIZE,
    pad_value: float = PAD_VALUE,
) -> tf.Tensor:
    target_h, target_w = target_size
    image = tf.cast(image, tf.float32)

    original_h = tf.cast(tf.shape(image)[0], tf.float32)
    original_w = tf.cast(tf.shape(image)[1], tf.float32)
    scale = tf.minimum(target_h / original_h, target_w / original_w)

    resized_h = tf.cast(tf.round(original_h * scale), tf.int32)
    resized_w = tf.cast(tf.round(original_w * scale), tf.int32)
    resized = tf.image.resize(image, [resized_h, resized_w], method="bilinear")

    pad_h = target_h - resized_h
    pad_w = target_w - resized_w
    pad_top = pad_h // 2
    pad_bottom = pad_h - pad_top
    pad_left = pad_w // 2
    pad_right = pad_w - pad_left

    padded = tf.pad(
        resized,
        paddings=[[pad_top, pad_bottom], [pad_left, pad_right], [0, 0]],
        constant_values=pad_value,
    )
    padded.set_shape((target_h, target_w, 3))
    return padded


def build_augmenter() -> keras.Sequential:
    return keras.Sequential(
        [
            keras.layers.RandomFlip("horizontal"),
            keras.layers.RandomRotation(0.03),
            keras.layers.RandomZoom(0.10),
            keras.layers.RandomContrast(0.10),
        ],
        name="train_augmentation",
    )


def dataframe_to_dataset(
    frame: pd.DataFrame,
    batch_size: int = 32,
    shuffle: bool = False,
    augment: bool = False,
    use_bbox_crop: bool = True,
    target_size: tuple[int, int] = TARGET_SIZE,
    sample_weight_column: str | None = None,
) -> tf.data.Dataset:
    tensor_slices: dict[str, np.ndarray] = {
        "image_path": frame["image_path"].to_numpy(),
        "make_label": frame["make_label"].to_numpy(dtype=np.int32),
        "model_label": frame["model_label"].to_numpy(dtype=np.int32),
        "has_bbox": frame["has_bbox"].fillna(False).to_numpy(dtype=np.float32),
        "bbox": frame[["bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2"]].fillna(0.0).to_numpy(dtype=np.float32),
    }
    if sample_weight_column is not None:
        tensor_slices[sample_weight_column] = frame[sample_weight_column].to_numpy(dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices(tensor_slices)
    if shuffle:
        ds = ds.shuffle(buffer_size=len(frame), seed=42, reshuffle_each_iteration=True)

    augmenter = build_augmenter() if augment else None

    def _map_row(row: dict[str, tf.Tensor]):
        image = tf.io.decode_jpeg(tf.io.read_file(row["image_path"]), channels=3)
        image = tf.cast(image, tf.float32)
        if use_bbox_crop:
            image = _crop_with_bbox(image, row["bbox"], row["has_bbox"])
        image = letterbox_resize(image, target_size=target_size)

        if augmenter is not None:
            image = augmenter(image, training=True)

        labels = {
            "make_output": row["make_label"],
            "model_output": row["model_label"],
        }

        if sample_weight_column is None:
            return image, labels

        sample_weights = {
            "make_output": tf.constant(1.0, dtype=tf.float32),
            "model_output": row[sample_weight_column],
        }
        return image, labels, sample_weights

    return ds.map(_map_row, num_parallel_calls=AUTOTUNE).batch(batch_size).prefetch(AUTOTUNE)


def build_multitask_model(
    num_makes: int,
    num_models: int,
    input_shape: tuple[int, int, int] = (299, 299, 3),
    dropout_rate: float = 0.40,
    hidden_dim: int = 512,
    backbone_weights: str | None = "imagenet",
) -> tuple[keras.Model, keras.Model]:
    backbone = keras.applications.ResNet50V2(
        include_top=False,
        weights=backbone_weights,
        input_shape=input_shape,
        pooling="avg",
    )
    backbone.trainable = False

    inputs = keras.Input(shape=input_shape, name="image")
    x = keras.applications.resnet_v2.preprocess_input(inputs)
    x = backbone(x, training=False)
    x = keras.layers.Dense(hidden_dim, activation="relu", name="shared_dense")(x)
    x = keras.layers.BatchNormalization(name="shared_batchnorm")(x)
    x = keras.layers.Dropout(dropout_rate, name="shared_dropout")(x)

    make_output = keras.layers.Dense(num_makes, activation="softmax", name="make_output")(x)
    model_output = keras.layers.Dense(num_models, activation="softmax", name="model_output")(x)

    model = keras.Model(
        inputs=inputs,
        outputs={
            "make_output": make_output,
            "model_output": model_output,
        },
        name="compcars_make_model_multitask",
    )
    return backbone, model


def compile_multitask_model(
    model: keras.Model,
    learning_rate: float = 1e-3,
    weight_decay: float = 1e-4,
    make_loss_weight: float = 0.35,
    model_loss_weight: float = 0.65,
) -> None:
    model.compile(
        optimizer=keras.optimizers.AdamW(learning_rate=learning_rate, weight_decay=weight_decay),
        loss={
            "make_output": keras.losses.SparseCategoricalCrossentropy(),
            "model_output": keras.losses.SparseCategoricalCrossentropy(),
        },
        loss_weights={
            "make_output": make_loss_weight,
            "model_output": model_loss_weight,
        },
        metrics={
            "make_output": [keras.metrics.SparseCategoricalAccuracy(name="accuracy")],
            "model_output": [
                keras.metrics.SparseCategoricalAccuracy(name="accuracy"),
                keras.metrics.SparseTopKCategoricalAccuracy(k=5, name="top5_accuracy"),
            ],
        },
    )


def make_callbacks(
    output_dir: Path,
    early_stop_patience: int = 4,
    lr_patience: int = 2,
    lr_factor: float = 0.5,
    min_lr: float = 1e-6,
) -> list[keras.callbacks.Callback]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    return [
        keras.callbacks.ModelCheckpoint(
            filepath=str(output_dir / "best_multitask_model.keras"),
            monitor="val_model_output_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        keras.callbacks.EarlyStopping(
            monitor="val_model_output_accuracy",
            mode="max",
            patience=early_stop_patience,
            restore_best_weights=True,
            verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_model_output_accuracy",
            mode="max",
            factor=lr_factor,
            patience=lr_patience,
            min_lr=min_lr,
            verbose=1,
        ),
        keras.callbacks.CSVLogger(str(output_dir / "training_log.csv")),
    ]


def combine_histories(*histories: keras.callbacks.History) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    epoch_offset = 0
    for idx, history in enumerate(histories, start=1):
        frame = pd.DataFrame(history.history)
        frame["epoch"] = np.arange(epoch_offset, epoch_offset + len(frame))
        frame["phase"] = f"phase_{idx}"
        epoch_offset += len(frame)
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def plot_training_curves(history_df: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(18, 4))
    axes[0].plot(history_df["epoch"], history_df["loss"], label="Train")
    axes[0].plot(history_df["epoch"], history_df["val_loss"], label="Validation")
    axes[0].set_title("Overall Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].legend()

    axes[1].plot(history_df["epoch"], history_df["make_output_accuracy"], label="Train")
    axes[1].plot(history_df["epoch"], history_df["val_make_output_accuracy"], label="Validation")
    axes[1].set_title("Make Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].legend()

    axes[2].plot(history_df["epoch"], history_df["model_output_accuracy"], label="Train Top-1")
    axes[2].plot(history_df["epoch"], history_df["val_model_output_accuracy"], label="Validation Top-1")
    axes[2].plot(history_df["epoch"], history_df["model_output_top5_accuracy"], label="Train Top-5")
    axes[2].plot(history_df["epoch"], history_df["val_model_output_top5_accuracy"], label="Validation Top-5")
    axes[2].set_title("Model Accuracy")
    axes[2].set_xlabel("Epoch")
    axes[2].legend()

    plt.tight_layout()
    plt.show()


def save_training_artifacts(
    output_dir: Path,
    history_df: pd.DataFrame,
    make_encoder: dict[str, int],
    model_encoder: dict[str, int],
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    history_df.to_csv(output_dir / "history_combined.csv", index=False)
    (output_dir / "make_encoder.json").write_text(json.dumps(make_encoder, indent=2))
    (output_dir / "model_encoder.json").write_text(json.dumps(model_encoder, indent=2))


def make_run_id(prefix: str = "run") -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}"


def best_validation_metrics(
    history_df: pd.DataFrame,
    monitor: str = "val_model_output_accuracy",
) -> dict[str, object]:
    if history_df.empty or monitor not in history_df.columns:
        return {}

    best_idx = history_df[monitor].idxmax()
    best_row = history_df.loc[best_idx]
    metric_names = [
        "epoch",
        "phase",
        "loss",
        "val_loss",
        "make_output_accuracy",
        "val_make_output_accuracy",
        "model_output_accuracy",
        "val_model_output_accuracy",
        "model_output_top5_accuracy",
        "val_model_output_top5_accuracy",
        "learning_rate",
    ]
    return {
        f"best_{name}": best_row[name]
        for name in metric_names
        if name in history_df.columns
    }


def final_history_metrics(history_df: pd.DataFrame) -> dict[str, object]:
    if history_df.empty:
        return {}

    final_row = history_df.iloc[-1]
    metric_names = [
        "epoch",
        "phase",
        "loss",
        "val_loss",
        "make_output_accuracy",
        "val_make_output_accuracy",
        "model_output_accuracy",
        "val_model_output_accuracy",
        "model_output_top5_accuracy",
        "val_model_output_top5_accuracy",
        "learning_rate",
    ]
    return {
        f"final_{name}": final_row[name]
        for name in metric_names
        if name in history_df.columns
    }


def save_experiment_summary(
    output_dir: Path,
    config: dict[str, object],
    history_df: pd.DataFrame,
    test_metrics: dict[str, object] | None = None,
    tracking_csv: Path | None = None,
) -> pd.DataFrame:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    record: dict[str, object] = {}
    record.update(config)
    record.update(best_validation_metrics(history_df))
    record.update(final_history_metrics(history_df))
    if test_metrics:
        record.update({f"test_{key}": value for key, value in test_metrics.items()})
    record["output_dir"] = str(output_dir)
    record["logged_at"] = datetime.now().isoformat(timespec="seconds")

    (output_dir / "experiment_config.json").write_text(json.dumps(config, indent=2, default=str))
    (output_dir / "experiment_summary.json").write_text(json.dumps(record, indent=2, default=str))

    tracking_csv = Path(tracking_csv) if tracking_csv is not None else output_dir.parent / "experiment_tracking.csv"
    tracking_csv.parent.mkdir(parents=True, exist_ok=True)

    new_row = pd.DataFrame([record])
    if tracking_csv.exists():
        old_rows = pd.read_csv(tracking_csv)
        combined = pd.concat([old_rows, new_row], ignore_index=True, sort=False)
    else:
        combined = new_row
    combined.to_csv(tracking_csv, index=False)
    return combined
