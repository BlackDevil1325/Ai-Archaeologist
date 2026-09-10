"""Prediction layer — turns visual similarity into archaeological hypotheses.

Approach (transparent, explainable, fully offline):
  * Compare the upload against every reference artefact (colour + style).
  * The top-5 matches vote for category / region / culture / date,
    weighted by (similarity^3) to emphasise close matches.
    Replicas vote at 0.92x.
  * Optional user context (find-spot, material, free-text notes) boosts
    matching references — exactly like a human expert using provenance.
  * A small heuristic prior (material + shape cues) is blended in so the
    model can gesture at categories outside the reference library.
"""

import json
import os
import time

import numpy as np

from .features import analyze_image, visual_similarity

TIMELINE = [
    ("Neolithic & Chalcolithic", -7000, -3300),
    ("Bronze Age", -3300, -1200),
    ("Iron Age", -1200, -600),
    ("Classical Antiquity", -600, 500),
    ("Early Medieval", 500, 1300),
    ("Late Medieval", 1300, 1600),
    ("Early Modern", 1600, 1900),
    ("Modern Era", 1900, 2100),
]

FOUND_REGION_MAP = {
    "south-asia-indus": "South Asia · Indus Valley",
    "south-asia-south": "South Asia · South India",
    "south-asia-other": "South Asia",
    "west-asia": "West Asia",
    "egypt": "North Africa · Egypt",
    "greece": "Mediterranean · Greece",
    "rome": "Mediterranean · Rome",
    "east-asia": "East Asia",
    "unknown": None,
}

MATERIAL_TOKEN = {
    "terracotta": "terracotta",
    "ceramic": "ceramic",
    "clay": "clay",
    "stone": "stone",
    "steatite": "steatite",
    "bronze": "bronze",
    "gold": "gold",
    "unknown": None,
}

# keyword groups found in free-text notes -> attributes they support
NOTE_KEYWORDS = [
    ({"chola", "tamil", "nataraja", "shiva", "dravid", "pallava", "pandya",
      "tanjore", "thanjavur", "madurai", "tamilnadu", "tamil nadu"},
     {"region_group": "South Asia · South India", "culture": "Chola",
      "category": "Sculpture / Statue"}),
    ({"indus", "harapp", "mohenjo", "mohendro", "dholavira", "lothal",
      "steatite", "unicorn seal"}, 
     {"region_group": "South Asia · Indus Valley", "culture": "Indus",
      "category": "Seal / Stamp Seal"}),
    ({"egypt", "pharaoh", "hieroglyph", "khafre", "chephren", "giza",
      "nile", "mummy", "luxor", "cairo"},
     {"region_group": "North Africa · Egypt", "culture": "Egypt",
      "category": "Sculpture / Statue"}),
    ({"greek", "greece", "athens", "attic", "amphora", "black-figure",
      "red-figure", "kylix", "krater"},
     {"region_group": "Mediterranean · Greece", "culture": "Greece",
      "category": "Pottery / Vessel"}),
    ({"roman", "rome", "pompeii", "imperial", "latium", "shipwreck"},
     {"region_group": "Mediterranean · Rome", "culture": "Rome",
      "category": "Pottery / Vessel"}),
    ({"mesopotam", "babylon", "sumer", "cuneiform", "assyria", "iraq",
      "tablet", "gilgamesh"},
     {"region_group": "West Asia · Mesopotamia", "culture": "Babylon",
      "category": "Clay Tablet"}),
    ({"bronze", "lost-wax", "lost wax", "metal statue", "cast metal"},
     {"material": "bronze"}),
    ({"terracotta", "pottery", "potsherd", "ceramic", "earthenware", "vase"},
     {"category": "Pottery / Vessel"}),
    ({"seal", "stamp", "inscription", "script"},
     {"category": "Seal / Stamp Seal"}),
    ({"coin", "gold", "jewellery", "jewelry", "ornament", "necklace"},
     {"material": "gold"}),
    ({"stone", "granite", "sandstone", "diorite", "basalt", "carved"},
     {"material": "stone"}),
]

MATERIAL_LABELS = {
    "terracotta": "fired terracotta clay",
    "pale_ceramic": "pale ceramic / limestone",
    "dark_stone": "dark polished stone",
    "bronze": "aged bronze metal",
    "gold": "gold-toned metal",
    "black_figure_paint": "black-and-orange painted pottery",
}


def format_year(y: int) -> str:
    y = int(round(y))
    if y < 0:
        return f"{abs(y):,} BCE".replace(",", "")
    return f"{y:,} CE".replace(",", "")


def format_range(a: int, b: int) -> str:
    return f"{format_year(a)} – {format_year(b)}"


def timeline_label(year: float) -> str:
    for label, start, end in TIMELINE:
        if start <= year <= end:
            return label
    # nearest fallback
    return min(TIMELINE, key=lambda t: abs((t[1] + t[2]) / 2 - year))[0]


class ArchaeologyEngine:
    def __init__(self, db_path: str, ref_dir: str):
        with open(db_path, "r", encoding="utf-8") as fh:
            self.db = json.load(fh)["artifacts"]
        self.ref_dir = ref_dir
        self.refs = []
        for art in self.db:
            img_path = os.path.join(ref_dir, art["image"])
            try:
                with open(img_path, "rb") as fh:
                    feats = analyze_image(fh.read())
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[engine] WARNING: could not index {art['id']}: {exc}")
                continue
            self.refs.append({"meta": art, "feats": feats})
        if not self.refs:
            raise RuntimeError("Reference library is empty — no images indexed.")
        styles = np.stack([r["feats"]["style"] for r in self.refs])
        self.style_mean = styles.mean(axis=0)
        self.style_std = styles.std(axis=0) + 1e-6
        for r in self.refs:
            r["feats"]["zstyle"] = ((r["feats"]["style"] - self.style_mean)
                                    / self.style_std)
        print(f"[engine] indexed {len(self.refs)} reference artefacts.")

    # ------------------------------------------------------------------ #
    def library(self):
        return {"count": len(self.refs),
                "artifacts": [r["meta"] for r in self.refs]}

    # ------------------------------------------------------------------ #
    def predict(self, file_bytes: bytes, context: dict | None = None) -> dict:
        t0 = time.time()
        context = context or {}
        try:
            up = analyze_image(file_bytes)
        except Exception:
            raise ValueError("Could not read that file as an image. "
                             "Please upload a JPG/PNG photo of the artefact.")
        up["zstyle"] = (up["style"] - self.style_mean) / self.style_std

        found_region = FOUND_REGION_MAP.get(context.get("found_region", "unknown"))
        material_hint = MATERIAL_TOKEN.get(context.get("material_hint", "unknown"))
        notes = (context.get("notes") or "").lower()

        note_hits = [attrs for keys, attrs in NOTE_KEYWORDS
                     if any(k in notes for k in keys)]

        # ---- similarity + boosted voting weights ---- #
        scored = []
        for r in self.refs:
            meta = r["meta"]
            sim = visual_similarity(up, r["feats"])
            w = (max(sim, 0.0) ** 3) * (0.92 if meta.get("replica") else 1.0)
            w = max(w, 1e-6)
            if found_region and meta["region_group"].startswith(found_region):
                w *= 1.6
            if material_hint and material_hint in meta["material"].lower():
                w *= 1.45
            note_mult = 1.0
            for attrs in note_hits:
                ok = all(meta.get(k, "").find(v) >= 0 for k, v in attrs.items()
                         if k in ("region_group", "culture", "category", "material"))
                if ok:
                    note_mult = min(note_mult * 1.3, 1.8)
            w *= note_mult
            scored.append({"meta": meta, "sim": sim, "weight": w})
        scored.sort(key=lambda d: d["sim"], reverse=True)
        top = scored[:5]

        def vote(key):
            acc: dict[str, float] = {}
            for d in top:
                acc[d["meta"][key]] = acc.get(d["meta"][key], 0.0) + d["weight"]
            total = sum(acc.values()) or 1.0
            ranked = sorted(acc.items(), key=lambda kv: kv[1], reverse=True)
            return ranked, total

        cat_ranked, _ = vote("category")
        grp_ranked, _ = vote("region_group")
        reg_ranked, _ = vote("region")
        cul_ranked, _ = vote("culture")
        mat_ranked, _ = vote("material")

        # ---- heuristic prior from material + shape cues (18% blend) ---- #
        priors = self._heuristic_prior(up)
        cats = [c for c, _ in cat_ranked]
        vote_total = sum(w for _, w in cat_ranked)
        blended = {}
        for c in set(cats) | set(priors):
            v = dict(cat_ranked).get(c, 0.0) / (vote_total or 1.0)
            blended[c] = 0.82 * v + 0.18 * priors.get(c, 0.0)
        win_cat = max(blended, key=blended.get)

        # ---- dating: similarity-weighted average of reference dates ---- #
        years = np.array([d["meta"]["period_mid"] for d in top], dtype=float)
        wts = np.array([d["weight"] for d in top], dtype=float)
        sims = np.array([max(d["sim"], 0.0) for d in top], dtype=float)
        dw = wts * sims * sims  # sim^5-equivalent: dating follows close matches
        wavg = float(np.average(years, weights=dw))
        wstd = float(np.sqrt(np.average((years - wavg) ** 2, weights=dw)))
        half = max(200.0, min(wstd * 1.15, 1200.0))
        low, high = int(round((wavg - half) / 50) * 50), int(round((wavg + half) / 50) * 50)
        period_name = timeline_label(wavg)

        # ---- confidences ---- #
        s1 = top[0]["sim"]
        agree = top[0]["weight"] / (sum(d["weight"] for d in top) or 1.0)
        tentative = s1 < 0.50
        overall = 0.30 + 0.45 * min(s1, 1.0) + 0.25 * agree
        if wstd > 1200:
            overall *= 0.88
        if tentative:
            overall = min(overall, 0.55)
        overall = round(float(min(max(overall, 0.05), 0.93)), 3)

        def share_conf(ranked):
            tot = sum(w for _, w in ranked) or 1.0
            return round(float(min(0.35 + 0.55 * (ranked[0][1] / tot), 0.92)), 3)

        win_group = grp_ranked[0][0]
        win_region = next((r for r, _ in reg_ranked
                           if self._group_of(r) == win_group), reg_ranked[0][0])

        similar = [{
            "id": d["meta"]["id"],
            "name": d["meta"]["name"],
            "image": d["meta"]["image"],
            "similarity": round(float(d["sim"] * 100), 1),
            "period": d["meta"]["period_label"],
            "region": d["meta"]["region"],
            "category": d["meta"]["category"],
            "culture": d["meta"]["culture"],
            "replica": bool(d["meta"].get("replica")),
        } for d in top]

        result = {
            "period": {
                "label": period_name,
                "date_range": format_range(low, high),
                "year_mid": int(round(wavg)),
                "year_mid_label": format_year(wavg),
                "confidence": overall,
            },
            "category": {
                "label": win_cat,
                "confidence": share_conf([(c, blended[c]) for c in blended]),
                "all": [{"label": c, "score": round(float(s), 3)}
                        for c, s in sorted(blended.items(), key=lambda kv: kv[1],
                                           reverse=True)[:4]],
            },
            "region": {
                "label": win_region,
                "group": win_group,
                "culture": cul_ranked[0][0],
                "confidence": share_conf(grp_ranked),
                "all": [{"label": g, "score": round(float(w / (sum(x[1] for x in grp_ranked) or 1.0)), 3)}
                        for g, w in grp_ranked[:4]],
            },
            "material": {
                "label": mat_ranked[0][0],
                "confidence": share_conf(mat_ranked),
            },
            "similar": similar,
            "visual": {
                "dominant_colors": up["palette"][:4],
                "material_cues": up["materials"],
                "texture": up["texture"],
                "aspect": up["aspect"],
                "dimensions": {"width": up["width"], "height": up["height"]},
            },
            "timeline": {"min": -3000, "max": 2000, "predicted": int(round(wavg)),
                         "low": low, "high": high},
            "tentative": tentative,
            "reasoning": self._reasoning(up, top, win_cat, win_region, period_name,
                                         format_range(low, high), context,
                                         found_region, material_hint, tentative),
            "meta": {
                "model": "AI-Archaeologist v1.0 (CBIR + heuristic fusion, offline)",
                "library_size": len(self.refs),
                "latency_ms": int((time.time() - t0) * 1000),
                "note": ("Educational demo — stylistic hypothesis only, "
                         "not a professional authentication."),
            },
        }
        return result

    # ------------------------------------------------------------------ #
    def _group_of(self, region_label: str) -> str:
        for r in self.refs:
            if r["meta"]["region"] == region_label:
                return r["meta"]["region_group"]
        return ""

    def _heuristic_prior(self, up: dict) -> dict:
        m = up["materials"]
        t = up["texture"]
        ratio = min(up["aspect"], 1 / max(up["aspect"], 1e-6))  # 0..1 squareness
        vessel = 1.0 if 0.55 <= up["aspect"] <= 1.8 else 0.5
        scores = {
            "Sculpture / Statue": (m["bronze"] * 0.7 + m["dark_stone"] * 0.5
                                   + t["edge_density"] * 0.6 + (1 - ratio) * 0.2),
            "Pottery / Vessel": ((m["terracotta"] * 0.6 + m["pale_ceramic"] * 0.5
                                  + m["black_figure_paint"] * 1.2) * vessel),
            "Seal / Stamp Seal": ((ratio ** 2) * 0.5 + t["edge_density"] * 0.4
                                  + max(m["dark_stone"], m["terracotta"]) * 0.4),
            "Clay Tablet": (m["pale_ceramic"] * 0.6 + m["terracotta"] * 0.3
                            + up["fg_ratio"] * 0.3 + t["edge_density"] * 0.2),
        }
        tot = sum(scores.values()) or 1.0
        return {k: v / tot for k, v in scores.items()}

    def _reasoning(self, up, top, win_cat, win_region, period_name,
                   date_range, context, found_region, material_hint, tentative):
        r = []
        t0, t1 = top[0], top[1]
        r.append(f"Closest visual match is “{t0['meta']['name']}” "
                 f"({t0['sim']*100:.1f}%) — {t0['meta']['culture']}, "
                 f"{t0['meta']['period_label']}."
                 + (" (Reference photo is a modern replica.)"
                    if t0['meta'].get('replica') else ""))
        if t1["sim"] > 0.40:
            r.append(f"Runner-up: “{t1['meta']['name']}” ({t1['sim']*100:.1f}%) — "
                     f"the prediction blends the top matches by similarity weight.")
        mats = up["materials"]
        best_mat = max(mats, key=mats.get)
        if mats[best_mat] > 0.05:
            r.append(f"Surface cues read as {MATERIAL_LABELS[best_mat]} "
                     f"(~{mats[best_mat]*100:.0f}% of the object area), which supports "
                     f"a “{win_cat}” reading.")
        pal = ", ".join(c["hex"] for c in up["palette"][:3])
        tone = ("warm earthen" if up["mean_rgb"][0] > up["mean_rgb"][2]
                else "cool / grey")
        r.append(f"Dominant palette {pal} — {tone} tones typical of aged, "
                 f"unrestored surfaces.")
        t = up["texture"]
        detail = "rich carved detail" if t["edge_density"] > 0.28 else (
            "moderate surface detail" if t["edge_density"] > 0.18 else "smooth, worn surfaces")
        sym = "balanced, symmetric composition" if t["symmetry"] > 0.55 else "asymmetric composition"
        r.append(f"Texture analysis: {detail} (edge density {t['edge_density']:.2f}), "
                 f"{sym} (symmetry {t['symmetry']:.2f}).")
        if found_region or material_hint or (context.get("notes") or "").strip():
            bits = []
            if found_region:
                bits.append(f"find-spot “{found_region}”")
            if material_hint:
                bits.append(f"material “{material_hint}”")
            if (context.get("notes") or "").strip():
                bits.append("your descriptive notes")
            r.append(f"Your context ({', '.join(bits)}) was fused into the vote "
                     f"— provenance-aware, like a human expert.")
        if up["materials"]["gold"] > 0.45 and up["aspect"] > 0.8:
            r.append("Note: strong gold-toned, compact signature — a coin, seal-matrix "
                     "or jewellery item is also possible, but those categories are not "
                     "yet in the reference library.")
        if tentative:
            r.append(f"No close match exists in the {len(self.refs)}-artefact reference "
                     f"library, so this {period_name} / {win_region} hypothesis "
                     f"({date_range}) is tentative — try adding find-spot context.")
        else:
            r.append(f"Date {date_range} comes from similarity-weighted averaging of "
                     f"reference dates, landing in the {period_name} horizon.")
        return r
