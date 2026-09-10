# 🏺 AI Archaeologist

**AI Archaeologist** is an offline AI-powered archaeological artefact analysis system.
Users can upload an image of an archaeological artefact, and the system analyzes its visual characteristics to provide an estimated:

* 🕰️ Historical period
* 🏺 Artefact category
* 🌍 Possible region and culture
* 🧱 Material
* 🔍 Similar known artefacts
* 📊 Prediction confidence
* 💡 Explainable reasoning

The system is designed as an **educational and research-oriented prototype** for exploring how computer vision and content-based image retrieval can support archaeological analysis.

> ⚠️ **Note:** This system provides an AI-based hypothesis and is not intended for professional archaeological authentication.

---

## ✨ Features

### 🔍 Artefact Image Analysis

Upload a JPG or PNG image of an archaeological artefact and analyze its visual characteristics.

### 🕰️ Period Prediction

The system estimates the approximate historical period and provides a date range such as:

```text
500 BCE – 300 BCE
```

### 🏺 Category Classification

The system predicts categories such as:

* Seal / Stamp Seal
* Pottery / Vessel
* Clay Tablet
* Metal Statue
* Stone Artefact
* Other archaeological artefacts

### 🌍 Region & Culture Prediction

The system compares the uploaded artefact with reference artefacts from different historical regions and cultures.

### 🧱 Material Detection

The vision pipeline identifies visual cues associated with:

* Terracotta
* Ceramic
* Stone
* Bronze
* Gold
* Painted pottery

### 🔎 Similar Artefact Search

The system ranks the **top 5 visually similar artefacts** from its reference library.

### 💡 Explainable AI

Instead of only providing a prediction, the system also displays visual information used during analysis, including:

* Dominant colours
* Texture
* Material cues
* Image dimensions
* Visual similarity
* Confidence score

### 🗺️ Context-Aware Prediction

Additional information can be provided by the user:

* Found region
* Material hint
* Additional notes

This contextual information is combined with visual similarity to improve the final prediction.

### 💻 Fully Offline

The application does not require:

* GPU
* API keys
* Cloud AI services
* Internet connection
* Large pre-trained model downloads

---

# 🧠 Methodology

The project uses a **Content-Based Image Retrieval (CBIR) + Heuristic Fusion** approach.

The overall workflow is:

```text
                 Input Artefact Image
                         │
                         ▼
                  Image Preprocessing
                         │
                         ▼
              Background Removal/Masking
                         │
                         ▼
             ┌───────────┴───────────┐
             │                       │
             ▼                       ▼
       Colour Features          Style Features
             │                       │
             │                       ├── Texture
             │                       ├── Sharpness
             │                       ├── Symmetry
             │                       ├── Brightness
             │                       ├── Contrast
             │                       └── Shape/Aspect
             │
             ▼
       Material Features
             │
             ├── Terracotta
             ├── Bronze
             ├── Gold
             ├── Stone
             └── Painted Pottery
             │
             └───────────┬───────────┘
                         ▼
                Reference Library
                         │
                         ▼
                Visual Similarity
                         │
                         ▼
                  Top-5 Artefacts
                         │
                         ▼
                   Weighted Voting
                         │
             ┌───────────┼───────────┐
             ▼           ▼           ▼
          Category     Region       Period
             │           │           │
             └───────────┼───────────┘
                         ▼
                Contextual Fusion
                         │
                         ▼
                  Final Prediction
                         │
                         ▼
                Explanation + Results
```

---

# 🔬 Image Feature Extraction

Each uploaded image is converted into multiple visual representations.

## 1. RGB Colour Histogram

A **512-dimensional RGB histogram** is extracted using:

```text
8 × 8 × 8 = 512 bins
```

The histogram represents the colour distribution of the artefact.

---

## 2. Style Features

A **19-dimensional style vector** is generated containing visual characteristics such as:

* Mean RGB values
* Saturation
* Brightness
* Contrast
* Edge density
* Sharpness
* Symmetry
* Aspect ratio
* Foreground ratio
* Material scores

---

## 3. Texture Analysis

The system calculates:

* Edge density
* Image sharpness
* Symmetry
* Brightness
* Contrast

These features help distinguish objects with different surface and structural characteristics.

---

## 4. Dominant Colour Detection

The project uses **K-Means clustering** to identify dominant colours in the artefact image.

These colours are displayed as part of the explanation.

---

## 5. Material Heuristics

Visual colour characteristics are used to estimate material cues.

For example:

```text
Orange / reddish surface
        ↓
Possible terracotta

Dark polished surface
        ↓
Possible stone

Brown/golden metallic surface
        ↓
Possible bronze
```

These are heuristic visual indicators rather than laboratory material identification.

---

# 📐 Similarity Calculation

The system combines two types of visual similarity:

```text
Visual Similarity
      =
65% Colour Similarity
+
35% Style Similarity
```

The implementation uses:

```text
0.65 × Histogram Intersection
+
0.35 × Whitened Style Cosine Similarity
```

The reference artefacts are then ranked according to their similarity with the uploaded image.

---

# 🗳️ Prediction Process

The top 5 visually similar reference artefacts are used for weighted voting.

The system considers:

```text
Visual Similarity
       +
Found Region
       +
Material Hint
       +
User Notes
       ↓
Weighted Voting
       ↓
Final Prediction
```

Higher similarity produces a higher voting weight.

The system then estimates:

* Category
* Region
* Culture
* Material
* Historical period

---

# 📚 Reference Dataset

The project contains a reference library of **14 archaeological artefacts**.

The reference collection includes examples associated with:

* Indus Valley
* Chola / South India
* Egypt
* Greece
* Rome
* Mesopotamia

The reference images are stored in:

```text
static/ref/
```

The corresponding metadata is stored in:

```text
artifacts_db.json
```

---

# 📁 Project Structure

```text
ai-archaeologist/
│
├── app.py
│
├── artifacts_db.json
├── requirements.txt
├── README.md
│
├── engine/
│   ├── __init__.py
│   ├── features.py
│   └── classifier.py
│
├── static/
│   ├── css/
│   │   └── style.css
│   │
│   ├── js/
│   │   └── app.js
│   │
│   └── ref/
│       └── reference artefact images
│
├── templates/
│   └── index.html
│
├── sample_images/
│   └── sample artefact images
│
└── .vscode/
    ├── launch.json
    └── extensions.json
```

---

# 🛠️ Technologies Used

| Technology   | Purpose                      |
| ------------ | ---------------------------- |
| Python       | Core programming             |
| Flask        | Web application and REST API |
| Pillow       | Image processing             |
| NumPy        | Numerical computation        |
| Scikit-learn | K-Means clustering           |
| HTML         | Frontend structure           |
| CSS          | User interface               |
| JavaScript   | Frontend interaction         |
| JSON         | Artefact knowledge base      |

---

# 📦 Installation

## Step 1: Clone the Repository

```bash
git clone <your-repository-url>
```

## Step 2: Enter the Project Directory

```bash
cd ai-archaeologist
```

## Step 3: Create a Virtual Environment

### macOS / Linux

```bash
python3 -m venv venv
source venv/bin/activate
```

### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

---

# 📥 Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:

```text
Flask
Pillow
NumPy
Scikit-learn
```

---

# ▶️ Run the Application

Run:

```bash
python app.py
```

The application will start on:

```text
http://127.0.0.1:5000
```

Open the address in your browser.

---

# 🖼️ How to Use

### Step 1

Open the web application.

### Step 2

Upload an archaeological artefact image.

### Step 3

Optionally provide:

```text
Found Region
Material Hint
Additional Notes
```

### Step 4

Click **Analyze**.

### Step 5

The system extracts visual features and compares the image with the reference library.

### Step 6

The application displays:

```text
Predicted Period
Predicted Category
Possible Region
Culture
Material
Confidence
Top Similar Artefacts
Visual Fingerprint
```

---

# 🔌 API Endpoints

## Get Reference Artefacts

```http
GET /api/artifacts
```

Returns the available reference artefact library.

---

## Analyze an Uploaded Image

```http
POST /api/analyze
```

### Input

Multipart form data:

```text
image
found_region
material_hint
notes
```

### Example

```bash
curl -X POST \
  -F image=@artefact.jpg \
  -F found_region=rome \
  http://127.0.0.1:5000/api/analyze
```

---

## Analyze a Sample Image

```http
POST /api/analyze-sample/<filename>
```

Example:

```bash
curl -X POST \
  http://127.0.0.1:5000/api/analyze-sample/indus_seal_unicorn.jpg
```

---

# ➕ Adding New Artefacts

The reference library can be expanded.

### 1. Add the image

Place the image inside:

```text
static/ref/
```

### 2. Add metadata

Add the corresponding artefact information to:

```text
artifacts_db.json
```

Important fields include:

```text
id
name
category
culture
region
region_group
period_label
date_start
date_end
period_mid
material
description
image
```

### 3. Restart the application

```bash
python app.py
```

The reference features are automatically rebuilt when the application starts.

---

# ⚠️ Limitations

This project is a **prototype / educational AI system**.

The predictions should not be considered professional archaeological authentication.

Accuracy can be affected by:

* Image quality
* Lighting conditions
* Background
* Camera angle
* Missing reference artefacts
* Limited reference dataset
* Similar-looking artefacts from different cultures

Professional archaeological identification may require:

* Excavation context
* Archaeological records
* Expert examination
* Radiocarbon dating
* Thermoluminescence dating
* XRF analysis
* Other laboratory techniques

---

# 🚀 Future Enhancements

Possible improvements include:

* 📚 Larger archaeological dataset
* 🧠 Deep learning models such as CNNs
* 🔥 Transfer learning using ResNet/EfficientNet
* 🔎 CLIP-based image-text retrieval
* 🌍 Larger global artefact database
* 🗺️ Interactive archaeological map
* 📈 Model evaluation with accuracy, precision, recall and F1-score
* 🔐 User authentication
* ☁️ Cloud deployment
* 📱 Improved mobile application
* 🧪 Integration with scientific material-analysis data
* 🤖 Large Language Model-based archaeological explanations

---

# 🎯 Project Objective

The main objective of **AI Archaeologist** is to demonstrate how computer vision and AI techniques can assist in the preliminary analysis of archaeological artefacts.

Instead of manually comparing every uploaded image with a large collection, the system automatically extracts visual features, searches a reference library, identifies similar artefacts, and generates an explainable archaeological hypothesis.

---

# 📜 Disclaimer

This project is intended for **educational, research, and demonstration purposes only**.

The generated predictions are AI-based estimates and should not be used as definitive evidence for archaeological dating, authentication, provenance, or cultural attribution.
