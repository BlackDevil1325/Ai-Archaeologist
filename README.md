# 🏺 AI Archaeologist

Upload a photo of an archaeological artefact → get an AI hypothesis of its
**approximate period**, **category**, **possible region**, and **similarity to known artefacts** —
with a full explanation of *why*.

100% offline. No GPU, no API keys, no model downloads. Built with Flask + Pillow + NumPy + scikit-learn.

## ✨ Features

- 🔍 **4 predictions per photo** — period (with timeline), category, region/culture, material
- 🏛️ **Similar-artefact ranking** — top-5 closest known artefacts with similarity scores
- 💡 **Explainable AI** — every prediction ships with human-readable reasoning + visual fingerprint
- 🗺️ **Provenance-aware** — optional find-spot, material and free-text notes fuse into the vote
- 📚 **Reference library** — 14 artefacts across Indus Valley, Chola South India, Egypt, Greece, Rome, Mesopotamia
- 🖱️ Drag-drop / paste / sample-click input, mobile-friendly sandstone-and-gold UI

## 🚀 Quick start

```bash
cd ai-archaeologist
pip install -r requirements.txt
python app.py
# open http://127.0.0.1:5000
```

## 🧠 How the model works

A transparent **CBIR (content-based image retrieval) + heuristic fusion** pipeline:

1. **See** — each image becomes a 512-bin RGB colour signature + a 19-d style vector
   (colour stats, edge density, sharpness, symmetry, brightness, contrast, aspect,
   foreground ratio, material cues) + a k-means dominant-colour palette.
2. **Sense material** — heuristic vision rules score terracotta, pale ceramic,
   dark stone, bronze, gold and black-figure-paint surface signatures.
3. **Compare** — blended similarity (`0.65·histogram-intersection + 0.35·whitened-style`) ranks every
   reference artefact; the top-5 matches vote for category/region/culture/date
   weighted by `similarity³` (replica photos vote at 0.92×). Studio
   backgrounds are masked out so the colour signature describes the object.
4. **Fuse & explain** — user context boosts matching references; the date is a
   similarity-weighted average of reference dates mapped onto a historical timeline;
   confidence reflects top-match strength + voter agreement.

## 📁 Project structure

```
ai-archaeologist/
├── app.py                  # Flask server + REST API
├── artifacts_db.json       # Reference knowledge base (edit to grow the library)
├── requirements.txt
├── engine/
│   ├── features.py         # Vision pipeline (histograms, texture, palette, materials)
│   └── classifier.py       # Voting, dating, context fusion, explanations
├── static/
│   ├── css/style.css
│   ├── js/app.js
│   └── ref/                # Reference artefact photos (14)
├── templates/index.html
└── sample_images/          # Copies for quick manual testing
```

## 🔌 API

| Endpoint | Method | Input | Output |
|---|---|---|---|
| `/` | GET | — | Web app |
| `/api/artifacts` | GET | — | Reference library JSON |
| `/api/analyze` | POST | `multipart`: `image` + `found_region`, `material_hint`, `notes` | Full prediction JSON |
| `/api/analyze-sample/<file>` | POST | JSON context | Full prediction JSON |

Example:

```bash
curl -X POST -F image=@coin.jpg -F found_region=rome \
  http://127.0.0.1:5000/api/analyze
```

## ➕ Growing the library (recommended!)

Accuracy scales with references. To add an artefact:

1. Drop a photo (≤640px) into `static/ref/`
2. Append an entry to `artifacts_db.json`:
   `id, name, category, culture, region, region_group, period_label,`
   `date_start, date_end, period_mid (negative = BCE), material, description, image`
3. Restart `app.py` — embeddings rebuild automatically at startup.

## ⚠️ Honest limits

Educational stylistic-guessing demo, **not** professional authentication. Real
archaeology requires excavation context, lab analysis (TL/OSL, C-14, XRF) and
expert eyes. Treat low-confidence / “tentative” outputs accordingly.

## 📜 Sources

Reference photos: museum collections & open archives (Met Museum, Egyptian Museum,
Wikimedia Commons, study replicas) — see per-card source notes in the app.
