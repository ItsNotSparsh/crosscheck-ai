"""Train separate claim-veracity and image-consistency heads on M4FC."""
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from PIL import Image, UnidentifiedImageError
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, f1_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from model_utils import load_clip

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data" / "m4fc"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def encode_rows(rows, processor, model, batch_size=24):
    rows = rows.reset_index(drop=True)
    features, indices = [], []
    for start in range(0, len(rows), batch_size):
        batch_rows = rows.iloc[start:start + batch_size]
        images, valid = [], []
        for idx, row in batch_rows.iterrows():
            try:
                with Image.open(DATA / "images" / row.image_file) as image:
                    if image.width < 32 or image.height < 32:
                        continue
                    images.append(image.convert("RGB"))
                valid.append(idx)
            except (OSError, UnidentifiedImageError):
                continue
        if not valid:
            continue
        texts = [str(rows.iloc[idx].claim)[:1000] or " " for idx in valid]
        inputs = processor(text=texts, images=images, return_tensors="pt",
                           padding=True, truncation=True)
        inputs = {key: value.to(DEVICE) for key, value in inputs.items()}
        with torch.inference_mode():
            output = model(**inputs)
            text_vec = torch.nn.functional.normalize(output.text_embeds, dim=-1)
            image_vec = torch.nn.functional.normalize(output.image_embeds, dim=-1)
            similarity = (text_vec * image_vec).sum(dim=1, keepdim=True)
            batch_features = torch.cat(
                [text_vec, image_vec, text_vec * image_vec, similarity], dim=1
            ).cpu().numpy()
        features.extend(batch_features)
        indices.extend(valid)
        if (start // batch_size + 1) % 20 == 0 or start + batch_size >= len(rows):
            print(f"Encoded {min(start + batch_size, len(rows))}/{len(rows)} rows", flush=True)
    return np.asarray(features, dtype=np.float32), np.asarray(indices, dtype=int)


def train_head(X, y, train_idx, dev_idx, name):
    train_mask = np.isin(np.arange(len(y)), train_idx)
    dev_mask = np.isin(np.arange(len(y)), dev_idx)
    Xtr, ytr = X[train_mask], y[train_mask]
    Xdev, ydev = X[dev_mask], y[dev_mask]
    if len(np.unique(ytr)) < 2:
        raise RuntimeError(f"{name}: training split has fewer than two classes")
    base = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=1500, class_weight="balanced", C=0.5)
    )
    clf = CalibratedClassifierCV(base, method="sigmoid", cv=3)
    clf.fit(Xtr, ytr)
    pred = clf.predict(Xdev)
    print(f"\n{name} dev report:\n{classification_report(ydev, pred, zero_division=0)}")
    return clf


def main():
    manifest = pd.read_csv(DATA / "manifest.csv").fillna("")
    manifest = manifest[manifest.image_file.map(lambda f: (DATA / "images" / f).is_file())].reset_index(drop=True)
    if manifest.empty:
        raise SystemExit("No downloaded M4FC images found. Run prepare_m4fc.py --download first.")
    processor, model = load_clip(DEVICE)
    X, encoded_idx = encode_rows(manifest, processor, model)
    rows = manifest.iloc[encoded_idx].reset_index(drop=True)
    train_idx = np.flatnonzero(rows.official_split.eq("train").to_numpy())
    dev_idx = np.flatnonzero(rows.official_split.eq("dev").to_numpy())
    if len(train_idx) == 0 or len(dev_idx) == 0:
        raise SystemExit("M4FC official train/dev split has no usable images; refusing a random leakage-prone split.")

    # Claim veracity uses fact-checker verdicts. The consistency head is trained
    # only where M4FC explicitly distinguishes true pairs from out-of-context pairs.
    fact_rows = rows.factual_label.isin(["true", "false"]).to_numpy()
    fact_y = np.where(rows.loc[fact_rows, "factual_label"].to_numpy() == "true", 1, 0)
    fact_original = np.flatnonzero(fact_rows)
    fact_clf = train_head(X[fact_rows], fact_y,
                          np.flatnonzero(np.isin(fact_original, train_idx)),
                          np.flatnonzero(np.isin(fact_original, dev_idx)), "Claim veracity")

    consistency_rows = rows.consistency_label.isin(["consistent", "mismatched"]).to_numpy()
    consistency_y = np.where(rows.loc[consistency_rows, "consistency_label"].to_numpy() == "consistent", 1, 0)
    consistency_original = np.flatnonzero(consistency_rows)
    consistency_clf = train_head(X[consistency_rows], consistency_y,
                                 np.flatnonzero(np.isin(consistency_original, train_idx)),
                                 np.flatnonzero(np.isin(consistency_original, dev_idx)),
                                 "Image-claim consistency")

    output = ROOT / "artifacts" / "m4fc_dual_classifier.joblib"
    joblib.dump({
        "veracity": fact_clf,
        "consistency": consistency_clf,
        "classes": {"fake": 0, "real": 1},
        "consistency_classes": {"mismatched": 0, "consistent": 1},
        "training_data": "M4FC English subset",
        "train_rows": int(len(train_idx)),
        "dev_rows": int(len(dev_idx)),
    }, output)
    (ROOT / "artifacts" / "m4fc_training_summary.json").write_text(
        json.dumps({"usable_images": len(rows), "train": len(train_idx), "dev": len(dev_idx),
                    "device": DEVICE, "artifact": str(output)}, indent=2),
        encoding="utf-8"
    )
    print(f"Saved dual-head model to {output}", flush=True)


if __name__ == "__main__":
    main()
