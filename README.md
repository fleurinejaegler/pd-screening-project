# Parkinson's Disease Screening — Drawings + Voice

Two independent PD-vs-healthy classifiers (hand-drawn spirals/waves, and sustained-vowel voice
recordings), each compared across several models with cross-validation, plus a small app that
combines both into one screening tool.

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Data — you need to download these yourself (not included in the repo)

**Voice** — UCI "Parkinsons" dataset (195 rows, 31 subjects, `class`/`status` column: 1 = PD, 0 = healthy).
- https://www.kaggle.com/datasets/porinitahoque/parkinsons-disease-pd-data-analysis
- or directly from UCI: https://archive.ics.uci.edu/dataset/174/parkinsons

Download the CSV and put it at:
```
data/voice/parkinsons.csv
```

**Drawings** — Kaggle spiral/wave drawings dataset.
- https://www.kaggle.com/datasets/kmader/parkinsons-drawings

Unzip it so you end up with:
```
data/drawings/spiral/training/healthy/*.png
data/drawings/spiral/training/parkinson/*.png
data/drawings/spiral/testing/healthy/*.png
data/drawings/spiral/testing/parkinson/*.png
data/drawings/wave/training/healthy/*.png
data/drawings/wave/training/parkinson/*.png
data/drawings/wave/testing/healthy/*.png
data/drawings/wave/testing/parkinson/*.png
```
(This matches the folder layout the Kaggle dataset ships with. If your download differs, adjust
`DRAWINGS_DIR` / the glob patterns at the top of `src/drawings_pipeline.py`.)

## Run

```bash
python src/voice_pipeline.py
python src/drawings_pipeline.py --test spiral
python src/drawings_pipeline.py --test wave
streamlit run src/app.py
```

## What each script does

- `src/voice_pipeline.py` — loads the voice CSV, auto-detects the target column (`class` or
  `status`), scales features, runs stratified k-fold cross-validation comparing Logistic
  Regression, Random Forest, SVM and an MLP, prints a comparison table, and saves the best model.
- `src/drawings_pipeline.py` — loads spiral or wave images, extracts classical hand-crafted
  features (contour smoothness, stroke width variance, tremor-like direction changes) with
  OpenCV, runs the same model comparison, and saves the best model + the feature extractor
  settings.
- `src/features.py` — the image feature-extraction functions shared by the drawings pipeline and
  the app.
- `src/app.py` — a small Streamlit app: draw a spiral/wave on a canvas, or record/upload a short
  voice clip, and get a live prediction from each trained model.

## Notes / limitations

- The voice dataset only has 195 rows total — cross-validation is used specifically because a
  single train/test split would be unreliable at that size.
- The app's voice feature extraction uses `parselmouth` (a Python wrapper around Praat) to
  approximate jitter/shimmer/HNR from a live recording. These won't be numerically identical to
  the original MDVP-derived features in the training CSV, so treat the app's voice prediction as
  a demo, not a clinical-grade measurement — worth saying explicitly in your presentation.
- These are two **independent** models (drawing-based, voice-based) shown side by side in the
  app, not a fused multimodal model — there's no patient-level pairing between the two datasets.
