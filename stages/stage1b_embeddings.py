import os
import time
import numpy as np
import pandas as pd
import torch
import open_clip
from PIL import Image
from pathlib import Path

IMAGE_DIR       = Path("data") / "processed" / "chart_images"
EMBED_DIR       = Path("data") / "processed" / "embeddings"
PIXEL_EMBED_DIR = Path("data") / "processed" / "embeddings_pixel"

EMBED_DIR.mkdir(parents=True, exist_ok=True)
PIXEL_EMBED_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "ViT-B-32"
PRETRAINED = "laion2b_s34b_b79k"
EMBED_DIM  = 512
BATCH_SIZE = 64
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
NORMALIZE  = True

IMAGE_HEIGHT = {5: 32, 10: 48, 20: 64, 60: 96}
IMAGE_WIDTH  = {5: 15, 10: 30, 20: 60, 60: 180}


def load_clip_model():
    print(f"Loading {MODEL_NAME} ({PRETRAINED})...")
    model, _, preprocess = open_clip.create_model_and_transforms(
        MODEL_NAME, pretrained=PRETRAINED)
    model = model.to(DEVICE).eval()
    print(f"Model loaded on {DEVICE}")
    return model, preprocess


def extract_clip_embeddings(npz_path, meta_path, model, preprocess, window_size):
    images = np.load(npz_path)["images"]
    meta   = pd.read_csv(meta_path)
    n      = len(images)
    h      = IMAGE_HEIGHT[window_size]
    w      = IMAGE_WIDTH[window_size]

    assert len(meta) == n

    embeddings = np.zeros((n, EMBED_DIM), dtype=np.float32)

    for start in range(0, n, BATCH_SIZE):
        end        = min(start + BATCH_SIZE, n)
        batch_imgs = []

        for i in range(start, end):
            img_pil = Image.fromarray(images[i].reshape(h, w), mode="L").convert("RGB")
            batch_imgs.append(preprocess(img_pil))

        batch_tensor = torch.stack(batch_imgs).to(DEVICE)

        with torch.no_grad():
            emb = model.encode_image(batch_tensor)
            if NORMALIZE:
                emb = emb / emb.norm(dim=-1, keepdim=True)

        embeddings[start:end] = emb.cpu().numpy()

        if (start // BATCH_SIZE) % 10 == 0:
            print(f"    {end:>6,} / {n:,}", end="\r")

    print(f"    {n:>6,} / {n:,} done.")
    return embeddings, meta


def extract_pixel_embeddings(npz_path, meta_path):
    images = np.load(npz_path)["images"].astype(np.float32)
    meta   = pd.read_csv(meta_path)

    assert len(meta) == len(images)

    images   = images / 255.0
    row_mean = images.mean(axis=1, keepdims=True)
    row_std  = images.std(axis=1, keepdims=True)
    row_std[row_std < 1e-8] = 1.0
    images   = (images - row_mean) / row_std

    norms = np.linalg.norm(images, axis=1, keepdims=True)
    norms[norms < 1e-8] = 1.0
    images = images / norms

    return images, meta


def run_clip_extraction(model, preprocess):
    archives = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.startswith("images_") and f.endswith(".npz")
    ])

    print(f"\nFound {len(archives)} archives")
    results = []

    for archive in archives:
        npz_path  = IMAGE_DIR / archive
        meta_path = IMAGE_DIR / archive.replace(".npz", "_meta.csv")

        if not meta_path.exists():
            print(f"  Missing meta for {archive}, skipping")
            continue

        parts   = archive.replace(".npz", "").split("_")
        ws      = int(parts[1].replace("d", ""))
        ver     = parts[2]
        freq    = parts[3]

        out_npz = EMBED_DIR / archive.replace("images_", "embeddings_")
        out_csv = EMBED_DIR / archive.replace("images_", "embeddings_").replace(".npz", "_meta.csv")

        if out_npz.exists():
            n = len(np.load(out_npz)["embeddings"])
            print(f"  {ws:>2}d {ver} {freq}: cached ({n:,})")
            results.append({"archive": archive, "window": ws, "freq": freq, "n": n, "status": "cached"})
            continue

        print(f"  {ws:>2}d {ver} {freq}:")
        t0 = time.time()

        embeddings, meta = extract_clip_embeddings(
            str(npz_path), str(meta_path), model, preprocess, ws)

        np.savez_compressed(out_npz, embeddings=embeddings)
        meta.to_csv(out_csv, index=False)

        elapsed = time.time() - t0
        print(f"    {embeddings.shape[0]:,} embeddings in {elapsed:.1f}s")
        results.append({"archive": archive, "window": ws, "freq": freq,
                        "n": embeddings.shape[0], "time_s": round(elapsed,1), "status": "new"})

    return pd.DataFrame(results)


def run_pixel_extraction():
    archives = sorted([
        f for f in os.listdir(IMAGE_DIR)
        if f.startswith("images_") and f.endswith(".npz")
    ])

    results = []

    for archive in archives:
        npz_path  = IMAGE_DIR / archive
        meta_path = IMAGE_DIR / archive.replace(".npz", "_meta.csv")

        if not meta_path.exists():
            continue

        parts     = archive.replace(".npz", "").split("_")
        ws        = int(parts[1].replace("d", ""))
        ver       = parts[2]
        freq      = parts[3]
        pixel_dim = IMAGE_HEIGHT[ws] * IMAGE_WIDTH[ws]

        out_npz = PIXEL_EMBED_DIR / archive.replace("images_", "embeddings_pixel_")
        out_csv = PIXEL_EMBED_DIR / archive.replace("images_", "embeddings_pixel_").replace(".npz", "_meta.csv")

        if out_npz.exists():
            n = len(np.load(out_npz)["embeddings"])
            print(f"  {ws:>2}d {ver} {freq}: cached ({n:,}, dim={pixel_dim})")
            results.append({"archive": archive, "window": ws, "freq": freq,
                            "n": n, "dim": pixel_dim, "status": "cached"})
            continue

        t0 = time.time()
        embeddings, meta = extract_pixel_embeddings(str(npz_path), str(meta_path))
        elapsed = time.time() - t0

        np.savez_compressed(out_npz, embeddings=embeddings)
        meta.to_csv(out_csv, index=False)

        print(f"  {ws:>2}d {ver} {freq}: {embeddings.shape[0]:,} (dim={pixel_dim}) in {elapsed:.1f}s")
        results.append({"archive": archive, "window": ws, "freq": freq,
                        "n": embeddings.shape[0], "dim": pixel_dim,
                        "time_s": round(elapsed,1), "status": "new"})

    return pd.DataFrame(results)


if __name__ == "__main__":
    model, preprocess = load_clip_model()
    clip_summary  = run_clip_extraction(model, preprocess)
    pixel_summary = run_pixel_extraction()

    print("\nCLIP summary:")
    print(clip_summary.to_string(index=False))

    print("\nPixel summary:")
    print(pixel_summary.to_string(index=False))

    print("\nStage 1b complete.")
