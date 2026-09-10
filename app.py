"""AI Archaeologist — Flask server.

Run:
    pip install -r requirements.txt
    python app.py
Then open http://127.0.0.1:5000
"""
import os
import time

from flask import Flask, jsonify, render_template, request
from werkzeug.utils import secure_filename

from engine import ArchaeologyEngine

BASE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE, "artifacts_db.json")
REF_DIR = os.path.join(BASE, "static", "ref")

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB uploads

print("[server] loading AI Archaeologist engine ...")
engine = ArchaeologyEngine(DB_PATH, REF_DIR)
print("[server] ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/artifacts")
def artifacts():
    return jsonify(engine.library())


@app.route("/api/analyze", methods=["POST"])
def analyze():
    t0 = time.time()
    upload = request.files.get("image")
    if upload is None or not upload.filename:
        return jsonify({"error": "No image uploaded."}), 400
    data = upload.read()
    if not data:
        return jsonify({"error": "Empty file uploaded."}), 400
    context = {
        "found_region": request.form.get("found_region", "unknown"),
        "material_hint": request.form.get("material_hint", "unknown"),
        "notes": request.form.get("notes", ""),
    }
    try:
        result = engine.predict(data, context)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[server] predict failed: {exc}")
        return jsonify({"error": "Analysis failed. Please try another image."}), 500
    result["meta"]["server_ms"] = int((time.time() - t0) * 1000)
    return jsonify(result)


@app.route("/api/analyze-sample/<name>", methods=["POST"])
def analyze_sample(name):
    fname = secure_filename(name)
    path = os.path.join(REF_DIR, fname)
    if not os.path.isfile(path):
        return jsonify({"error": "Unknown sample image."}), 404
    with open(path, "rb") as fh:
        data = fh.read()
    payload = request.get_json(silent=True) or {}
    context = {
        "found_region": payload.get("found_region", "unknown"),
        "material_hint": payload.get("material_hint", "unknown"),
        "notes": payload.get("notes", ""),
    }
    try:
        result = engine.predict(data, context)
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[server] sample predict failed: {exc}")
        return jsonify({"error": "Analysis failed."}), 500
    result["sample_image"] = f"/static/ref/{fname}"
    return jsonify(result)


@app.errorhandler(413)
def too_large(_e):
    return jsonify({"error": "Image too large — please keep uploads under 10 MB."}), 413


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
