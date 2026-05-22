import numpy as np
import pandas as pd
import joblib
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

EMBED_DIR       = Path("data") / "processed" / "embeddings"
PIXEL_EMBED_DIR = Path("data") / "processed" / "embeddings_pixel"
FACTOR_DIR      = Path("data") / "processed" / "factors"
FACTOR_DIR.mkdir(parents=True, exist_ok=True)

N_COMPONENTS   = 10
TARGET_ARCHIVE = "embeddings_20d_V2_month.npz"
FACTOR_COLS    = [f"PC{i+1}" for i in range(N_COMPONENTS)]


def load_embeddings(embed_dir, archive_name):
    E    = np.load(embed_dir / archive_name)["embeddings"]
    meta = pd.read_csv(embed_dir / archive_name.replace(".npz", "_meta.csv"))
    meta["date"] = pd.to_datetime(meta["date"])
    assert len(E) == len(meta)
    return E, meta


def make_train_test_split(meta):
    all_dates    = sorted(meta["date"].unique())
    split_idx    = int(len(all_dates) * 0.80)
    train_cutoff = all_dates[split_idx]
    train_mask   = meta["date"] < train_cutoff
    test_mask    = meta["date"] >= train_cutoff
    return train_mask, test_mask, train_cutoff


def fit_pca(E, train_mask):
    pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("pca",    PCA(n_components=N_COMPONENTS, random_state=42)),
    ])
    pipeline.fit(E[train_mask.values])
    return pipeline


def build_factor_df(E, meta, pipeline, train_cutoff):
    F = pipeline.transform(E)
    factors = meta.copy()
    for j, col in enumerate(FACTOR_COLS):
        factors[col] = F[:, j]
    factors["split"] = factors["date"].apply(
        lambda d: "train" if d < train_cutoff else "test")
    factors["month"] = factors["date"].dt.to_period("M").astype(str)
    return factors


def run_pca(embed_dir, pixel_embed_dir, archive_name):
    E_clip, meta_clip = load_embeddings(embed_dir, archive_name)
    train_mask, test_mask, train_cutoff = make_train_test_split(meta_clip)

    print(f"Train: {train_mask.sum()}  Test: {test_mask.sum()}  Split: {pd.Timestamp(train_cutoff).date()}")

    pca_clip = fit_pca(E_clip, train_mask)
    ev       = pca_clip.named_steps["pca"].explained_variance_ratio_
    cumev    = np.cumsum(ev)
    print(f"CLIP variance explained: {cumev[-1]:.4f}")

    factors_clip = build_factor_df(E_clip, meta_clip, pca_clip, train_cutoff)
    factors_clip.to_parquet(FACTOR_DIR / "clip_pca_factors_20d_month.parquet", index=False)
    factors_clip.to_csv(FACTOR_DIR / "clip_pca_factors_20d_month.csv", index=False)
    joblib.dump(pca_clip, FACTOR_DIR / "pca_pipeline_clip_20d_month.pkl")
    print("CLIP factors saved.")

    pixel_archive = archive_name.replace("embeddings_", "embeddings_pixel_")
    E_pix, meta_pix = load_embeddings(pixel_embed_dir, pixel_archive)
    pca_pixel    = fit_pca(E_pix, train_mask)
    ev_pix       = pca_pixel.named_steps["pca"].explained_variance_ratio_
    cumev_pix    = np.cumsum(ev_pix)
    print(f"Pixel variance explained: {cumev_pix[-1]:.4f}")

    factors_pixel = build_factor_df(E_pix, meta_pix, pca_pixel, train_cutoff)
    factors_pixel.to_parquet(FACTOR_DIR / "pixel_pca_factors_20d_month.parquet", index=False)
    joblib.dump(pca_pixel, FACTOR_DIR / "pca_pipeline_pixel_20d_month.pkl")
    print("Pixel factors saved.")

    return factors_clip, factors_pixel, train_cutoff


if __name__ == "__main__":
    factors_clip, factors_pixel, train_cutoff = run_pca(
        EMBED_DIR, PIXEL_EMBED_DIR, TARGET_ARCHIVE)
    print("\nStage 2 complete.")
    print(factors_clip[FACTOR_COLS + ["ticker","month","split"]].head())
