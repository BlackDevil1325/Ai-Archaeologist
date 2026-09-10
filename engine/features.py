"""Visual feature extraction for the AI Archaeologist.

A lightweight, fully-offline computer-vision pipeline built on Pillow +
NumPy + scikit-learn (no GPU / no large model downloads required):

  1. Colour signature  — 8x8x8 RGB histogram (512-d, L1 normalised)
  2. Style vector      — 19-d vector of colour/texture/shape/material cues
  3. Dominant colours  — k-means palette (for the UI + material cues)
  4. Material cues     — heuristic scores for terracotta, bronze, stone...

Backdrop pixels are masked via border-connected flood-fill, then
similarity = 0.65 * hist_intersection + 0.35 * whitened_style_cosine.
"""

import io
from collections import deque

import numpy as np
from PIL import Image, ImageOps
from sklearn.cluster import KMeans

HIST_BINS = 8
HIST_DIM = HIST_BINS ** 3          # 512
PROBE_SIZE = (256, 256)
PALETTE_SIZE = (64, 64)


# --------------------------------------------------------------------------- #
# loading / colour conversion
# --------------------------------------------------------------------------- #
def load_image(file_bytes: bytes, max_side: int = 512) -> Image.Image:
    img = Image.open(io.BytesIO(file_bytes))
    img = ImageOps.exif_transpose(img)
    img = img.convert("RGB")
    img.thumbnail((max_side, max_side), Image.LANCZOS)
    return img


def rgb_to_hsv(arr: np.ndarray):
    """Vectorised RGB(0-1) -> H,S,V (0-1)."""
    r, g, b = arr[..., 0], arr[..., 1], arr[..., 2]
    mx = np.maximum(np.maximum(r, g), b)
    mn = np.minimum(np.minimum(r, g), b)
    diff = mx - mn
    h = np.zeros_like(mx)
    m = diff > 1e-6
    rmax = m & (mx == r)
    gmax = m & (mx == g)
    bmax = m & (mx == b)
    h[rmax] = ((g[rmax] - b[rmax]) / diff[rmax]) % 6.0
    h[gmax] = (b[gmax] - r[gmax]) / diff[gmax] + 2.0
    h[bmax] = (r[bmax] - g[bmax]) / diff[bmax] + 4.0
    h = (h / 6.0) % 1.0
    s = np.where(mx > 1e-6, diff / np.maximum(mx, 1e-6), 0.0)
    return h, s, mx


def flood_background_mask(img: Image.Image, size: int = 64,
                            thresh: float = 50.0) -> np.ndarray:
    """Border-connected backdrop mask (white AND black studio backdrops).

    Estimates the backdrop colour from the four corners, then flood-fills
    from the image border through similarly-coloured pixels only.
    Returns a size x size boolean array (True = background).
    """
    small = np.asarray(img.resize((size, size), Image.BILINEAR), dtype=np.float64)
    corners = np.concatenate([
        small[0:3, 0:3].reshape(-1, 3), small[0:3, -3:].reshape(-1, 3),
        small[-3:, 0:3].reshape(-1, 3), small[-3:, -3:].reshape(-1, 3),
    ])
    bg_color = np.median(corners, axis=0)
    dist = np.sqrt(((small - bg_color) ** 2).sum(axis=2))
    candidate = dist < thresh
    seen = np.zeros((size, size), dtype=bool)
    dq = deque()
    for x in range(size):
        for y in (0, size - 1):
            if candidate[y, x]:
                dq.append((y, x))
    for y in range(size):
        for x in (0, size - 1):
            if candidate[y, x]:
                dq.append((y, x))
    while dq:
        cy, cx = dq.popleft()
        if seen[cy, cx]:
            continue
        seen[cy, cx] = True
        for ny, nx in ((cy - 1, cx), (cy + 1, cx), (cy, cx - 1), (cy, cx + 1)):
            if 0 <= ny < size and 0 <= nx < size and not seen[ny, nx] and candidate[ny, nx]:
                dq.append((ny, nx))
    if seen.mean() > 0.92:  # degenerate (object fills the frame) — mask nothing
        return np.zeros((size, size), dtype=bool)
    return seen


# --------------------------------------------------------------------------- #
# descriptors
# --------------------------------------------------------------------------- #
def rgb_histogram(img: Image.Image, bins: int = HIST_BINS,
                  mask: np.ndarray | None = None) -> np.ndarray:
    """Colour histogram, optionally restricted to foreground pixels.

    Museum photos share near-white studio backgrounds; masking them out
    keeps the signature about the *object*, not the backdrop.
    """
    small = img.resize(PROBE_SIZE, Image.BILINEAR)
    arr = np.asarray(small, dtype=np.float32) / 255.0
    px = arr[mask] if mask is not None and mask.sum() > 100 else arr.reshape(-1, 3)
    q = np.clip((px * bins).astype(np.int32), 0, bins - 1)
    idx = (q[..., 0] * bins * bins + q[..., 1] * bins + q[..., 2]).ravel()
    hist = np.bincount(idx, minlength=bins ** 3).astype(np.float64)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def texture_features(img: Image.Image) -> dict:
    g = np.asarray(img.convert("L").resize((128, 128)), dtype=np.float64) / 255.0
    gy, gx = np.gradient(g)
    mag = np.sqrt(gx * gx + gy * gy)
    edge_density = float((mag > mag.mean() + mag.std()).mean())

    inner = g[1:-1, 1:-1]
    lap = (-4.0 * inner + g[:-2, 1:-1] + g[2:, 1:-1]
           + g[1:-1, :-2] + g[1:-1, 2:])
    sharp_raw = float(lap.var())
    sharpness = float(np.clip(np.log1p(1000.0 * sharp_raw) / 4.0, 0.0, 1.0))

    left = g[:, :64]
    right = np.fliplr(g[:, 64:])
    a = (left - left.mean()).ravel()
    b = (right - right.mean()).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    symmetry = float(a @ b / denom) if denom > 1e-9 else 0.0

    return {
        "edge_density": edge_density,
        "sharpness": sharpness,
        "symmetry": symmetry,
        "brightness": float(g.mean()),
        "contrast": float(g.std()),
    }


def dominant_colors(img: Image.Image, k: int = 4, bg_extra: np.ndarray | None = None):
    small = img.resize(PALETTE_SIZE, Image.BILINEAR)
    px = np.asarray(small, dtype=np.float64)
    _h, _s, _v = rgb_to_hsv(px / 255.0)
    _bg = ((_v > 0.93) & (_s < 0.12))
    if bg_extra is not None and bg_extra.shape == _bg.shape:
        _bg = _bg | bg_extra
    _fg = ~_bg
    data = px[_fg] if _fg.sum() >= 200 else px.reshape(-1, 3)
    k = min(k, len(data))
    km = KMeans(n_clusters=k, n_init=5, random_state=0)
    labels = km.fit_predict(data)
    counts = np.bincount(labels, minlength=k).astype(float)
    order = np.argsort(-counts)
    palette = []
    for i in order:
        rgb = tuple(int(v) for v in np.clip(km.cluster_centers_[i], 0, 255))
        palette.append({
            "hex": "#{:02x}{:02x}{:02x}".format(*rgb),
            "rgb": list(rgb),
            "pct": round(float(counts[i] / counts.sum()), 3),
        })
    return palette


def material_scores(arr: np.ndarray, h: np.ndarray, s: np.ndarray, v: np.ndarray,
                    fg_ratio: float) -> dict:
    """Heuristic material cues, each 0..1 (independent, foreground-normalised)."""
    fg = max(float(fg_ratio), 0.25)

    def frac(mask):
        return float(mask.mean() / fg)

    is_red_orange = (h < 0.09) | (h > 0.95)
    terracotta = frac(is_red_orange & (s > 0.25) & (s < 0.85) & (v > 0.22) & (v < 0.78))
    pale_ceramic = frac((s < 0.28) & (v > 0.52))
    dark_stone = frac((s < 0.32) & (v < 0.38))
    bronze = frac((h > 0.03) & (h < 0.14) & (s > 0.18) & (s < 0.75)
                  & (v > 0.15) & (v < 0.68))
    gold = frac((h > 0.07) & (h < 0.18) & (s > 0.42) & (v > 0.5))
    specular = frac((v > 0.9) & (s < 0.45))
    dark_frac = float((v < 0.28).mean() / fg)
    orange_frac = frac((h > 0.02) & (h < 0.11) & (s > 0.3) & (v > 0.3) & (v < 0.75))
    black_figure = float(np.clip(dark_frac * orange_frac * 4.0, 0.0, 1.0))

    # metallic sheen boosts bronze/gold slightly
    bronze = float(np.clip(bronze * 1.15 + specular * 0.35, 0.0, 1.0))

    return {
        "terracotta": round(min(terracotta, 1.0), 3),
        "pale_ceramic": round(min(pale_ceramic, 1.0), 3),
        "dark_stone": round(min(dark_stone, 1.0), 3),
        "bronze": round(min(bronze, 1.0), 3),
        "gold": round(min(gold, 1.0), 3),
        "black_figure_paint": round(black_figure, 3),
    }


# --------------------------------------------------------------------------- #
# main entry point
# --------------------------------------------------------------------------- #
def analyze_image(file_bytes: bytes) -> dict:
    img = load_image(file_bytes)
    w, hgt = img.size

    probe = img.resize(PROBE_SIZE, Image.BILINEAR)
    arr = np.asarray(probe, dtype=np.float64) / 255.0
    h, s, v = rgb_to_hsv(arr)

    white_bg = (v > 0.93) & (s < 0.12)        # near-white studio background
    flood64 = flood_background_mask(img)    # any border-connected backdrop
    flood256 = np.asarray(Image.fromarray(
        (flood64 * 255).astype(np.uint8)).resize(PROBE_SIZE, Image.NEAREST)) > 127
    bg_mask = white_bg | flood256
    fg_mask = ~bg_mask
    fg_ratio = float(np.clip(1.0 - bg_mask.mean(), 0.0, 1.0))

    hist = rgb_histogram(img, mask=fg_mask if fg_mask.sum() > 100 else None)
    tex = texture_features(img)
    palette = dominant_colors(img, bg_extra=flood64)
    mats = material_scores(arr, h, s, v, fg_ratio)

    fg_px = arr[fg_mask] if fg_mask.sum() > 100 else arr.reshape(-1, 3)
    mean_rgb = fg_px.mean(axis=0)
    s_mean = float(s[fg_mask].mean()) if fg_mask.sum() > 100 else float(s.mean())
    v_mean = float(v[fg_mask].mean()) if fg_mask.sum() > 100 else float(v.mean())
    ratio = min(w, hgt) / max(w, hgt)
    style = np.array([
        float(mean_rgb[0]), float(mean_rgb[1]), float(mean_rgb[2]),
        s_mean, v_mean,
        tex["brightness"], tex["contrast"],
        tex["edge_density"], tex["sharpness"],
        float((tex["symmetry"] + 1.0) / 2.0),
        float(ratio), 1.0 if w >= hgt else 0.0, float(fg_ratio),
        mats["terracotta"], mats["pale_ceramic"], mats["dark_stone"],
        mats["bronze"], mats["gold"], mats["black_figure_paint"],
    ], dtype=np.float64)

    return {
        "width": w,
        "height": hgt,
        "aspect": round(w / hgt, 3),
        "fg_ratio": round(float(fg_ratio), 3),
        "hist": hist,
        "style": style,
        "palette": palette,
        "materials": mats,
        "texture": {k: round(float(vv), 3) for k, vv in tex.items()},
        "mean_rgb": [round(float(c), 3) for c in mean_rgb],
    }


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom < 1e-12:
        return 0.0
    return float(np.clip(a @ b / denom, -1.0, 1.0))


def hist_intersection(a: np.ndarray, b: np.ndarray) -> float:
    """Histogram intersection for L1-normalised hists (1.0 = identical)."""
    return float(np.minimum(a, b).sum())


def visual_similarity(fa: dict, fb: dict) -> float:
    """Blended colour + style similarity in 0..1.

    Colour uses histogram intersection (more discriminative than cosine);
    style uses cosine on whitened (z-scored) vectors mapped to 0..1.
    """
    ch = hist_intersection(fa["hist"], fb["hist"])
    if "zstyle" in fa and "zstyle" in fb:
        cs = (cosine(fa["zstyle"], fb["zstyle"]) + 1.0) / 2.0
    else:  # pragma: no cover - fallback for unindexed comparisons
        cs = cosine(fa["style"], fb["style"])
    return float(0.65 * ch + 0.35 * cs)
