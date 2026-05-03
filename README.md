# CompCars Multi-Task Vehicle Recognition

A multi-task ResNet50V2 classifier that jointly predicts vehicle **make** and **model** from a single image, framed as a **listing-verification signal** for used-car marketplaces.

> Team 9 — BU Questrom BA865, Spring 2026
> Tianqi Sun · Yanlun Li · Xincheng Ding

---

## Why this exists

Used-car listings rely on seller-reported make/model fields. Wrong or missing metadata hurts search quality and creates fraud risk. We built an image-based classifier that can be plugged in as an independent verification signal — flagging mismatches between what the seller claims and what the photo actually shows.

**Stakeholders served:**
- **Marketplaces** — cleaner metadata + fraud-screening signal
- **Buyers** — more trustworthy search results
- **Sellers** — faster listing creation through auto-fill

---

## Results

We trained two model variants with different deployment trade-offs:

| Variant | Task | Top-1 Accuracy | Top-5 Accuracy | Requires bbox? |
|---------|------|----------------|----------------|----------------|
| **v4ab** | Single-task (model-only, 406 classes) | **91.99%** | — | No — works on raw user uploads |
| **W** | Multi-task (make + model joint) | **95.15%** (model) / **95.99%** (make) | High | Yes — needs bbox crop |

**Business takeaway:**
- `v4ab` is more directly deployable for messy user-uploaded marketplace photos
- `W` reaches higher accuracy when you have a detector or bbox source upstream

---

## Dataset

- **Source:** [CompCars](http://mmlab.ie.cuhk.edu.hk/datasets/comp_cars/index.html) (Web-Nature subset, originally CUHK MMLab) — accessed via a [Kaggle mirror](https://www.kaggle.com/datasets)
- **Raw size:** 1,706 unique make-model pairs
- **Filtered to:** 406 classes / 52,490 images (kept pairs with ≥3 production years and ≥120 images)

**Why we re-scanned the filesystem:** the dataset's official `train_test_split` only covers 431/1,706 make-model pairs. Using it would have discarded most usable classes, so we scanned `image/` directly and applied our own class-keep filters before splitting.

---

## Architecture

```
Input image (299×299, letterbox-padded)
        │
        ▼
ResNet50V2 backbone (ImageNet pretrained, frozen)
        │
        ▼
Dense(512, ReLU) → BatchNorm → Dropout(0.40)
        │
   ┌────┴────┐
   ▼         ▼
make_output   model_output
(softmax)    (softmax)
```

**Key engineering choices:**
- **Letterbox preprocessing** (preserve aspect ratio + pad to 299×299) instead of crop-and-stretch
- **Optional bbox crop** when CompCars label files contain bounding boxes
- **Multi-task loss weighting:** 0.35 for `make_output`, 0.65 for `model_output` (model is harder)
- **Class imbalance handling:** inverse-frequency `sample_weight` on the model head only
- **Optimizer:** AdamW with `weight_decay=1e-4`
- **Data augmentation:** RandomFlip / RandomRotation(0.03) / RandomZoom(0.10) / RandomContrast(0.10)
- **Callbacks:** ModelCheckpoint, EarlyStopping (patience 4), ReduceLROnPlateau (factor 0.5, patience 2), CSVLogger
- **Stratified 70/15/15** train/val/test split by `model_key`

---

## Repo layout

```
compcars-multitask-classifier/
├── compcars_kaggle_multitask.py   # All pipeline + model code
├── README.md                       # This file
└── .gitignore
```

The code is structured as a single module with named, composable functions. To go end-to-end:

```python
from compcars_kaggle_multitask import (
    build_training_dataframe,
    split_dataframe,
    build_encoders,
    apply_encoders,
    add_model_sample_weights,
    dataframe_to_dataset,
    build_multitask_model,
    compile_multitask_model,
    make_callbacks,
    combine_histories,
    save_training_artifacts,
    save_experiment_summary,
)

# 1. Scan + filter
df = build_training_dataframe("/path/to/CompCars_root", min_years=3, min_images_per_model=120)

# 2. Split + encode
train_df, val_df, test_df = split_dataframe(df)
make_enc, model_enc = build_encoders(train_df, val_df, test_df)
train_df = add_model_sample_weights(apply_encoders(train_df, make_enc, model_enc))
val_df  = apply_encoders(val_df, make_enc, model_enc)
test_df = apply_encoders(test_df, make_enc, model_enc)

# 3. tf.data pipelines
train_ds = dataframe_to_dataset(train_df, shuffle=True, augment=True, sample_weight_column="model_sample_weight")
val_ds   = dataframe_to_dataset(val_df)
test_ds  = dataframe_to_dataset(test_df)

# 4. Model
backbone, model = build_multitask_model(num_makes=len(make_enc), num_models=len(model_enc))
compile_multitask_model(model)

# 5. Train
callbacks = make_callbacks("./runs/run_001")
history = model.fit(train_ds, validation_data=val_ds, epochs=20, callbacks=callbacks)
```

---

## Reproducing

**Requirements:** Python ≥ 3.10. The code uses TF 2.x with `tf.data`, `tf.keras`, scikit-learn, pandas, numpy, matplotlib.

```bash
pip install tensorflow pandas numpy scikit-learn matplotlib
```

**Data:** download the CompCars Web-Nature subset from the Kaggle mirror and unpack it so the layout looks like:

```
CompCars_root/
├── image/<make_id>/<model_id>/<year>/*.jpg
└── label/<make_id>/<model_id>/<year>/*.txt   # optional, used for bbox crop
```

Then run the snippet under [Repo layout](#repo-layout).

---

## What's not in this repo

- **Trained model weights** (`best_multitask_model.keras`, several hundred MB) — host on Hugging Face / GitHub Releases
- **Raw CompCars images** — download from Kaggle, see above
- **Notebook with training curves and confusion matrices** — TODO

---

## Next steps (post-course)

- **Deployable pipeline:** train a no-bbox variant or pair the W model with an upstream detector
- **Stricter eval:** out-of-time test set with newer vehicle classes
- **Business extension:** image-text mismatch flagging for marketplace fraud detection
- **Pricing-aware retraining:** flag listings with anomalous price for the predicted make/model
