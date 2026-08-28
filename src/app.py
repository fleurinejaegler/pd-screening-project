"""
Small demo app combining both screening models:
  - draw a spiral/wave -> drawing-based prediction
  - record/upload a short sustained-vowel voice clip -> voice-based prediction

Run with:
    streamlit run src/app.py

Requires both models to have been trained first:
    python src/voice_pipeline.py
    python src/drawings_pipeline.py --test spiral
"""

from pathlib import Path

import cv2
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from streamlit_drawable_canvas import st_canvas

from features import extract_features

VOICE_MODEL_PATH = Path("models/voice_best_model.joblib")
DRAWING_MODEL_PATH = Path("models/drawings_spiral_best_model.joblib")

st.set_page_config(page_title="Parkinson's Screening Demo", layout="centered")
st.title("Parkinson's Disease Screening Demo")
st.caption(
    "Two independent, separately trained models -- not a fused multimodal model. "
    "This is a class project demo, not a medical device."
)

tab_drawing, tab_voice = st.tabs(["Drawing screening", "Voice screening"])


# ---------------------------------------------------------------- drawing tab
with tab_drawing:
    st.subheader("Draw a spiral")
    st.write("Draw a spiral below, roughly filling the canvas, then click **Predict**.")

    canvas_result = st_canvas(
        stroke_width=3,
        stroke_color="#000000",
        background_color="#FFFFFF",
        height=300,
        width=300,
        drawing_mode="freedraw",
        key="canvas",
    )

    if st.button("Predict from drawing"):
        if not DRAWING_MODEL_PATH.exists():
            st.error(f"No trained model found at {DRAWING_MODEL_PATH}. Run drawings_pipeline.py first.")
        elif canvas_result.image_data is None:
            st.warning("Draw something first.")
        else:
            img = canvas_result.image_data.astype(np.uint8)
            img_gray = cv2.cvtColor(img, cv2.COLOR_RGBA2GRAY)

            bundle = joblib.load(DRAWING_MODEL_PATH)
            pipe, feature_names = bundle["pipeline"], bundle["feature_names"]

            feats = extract_features(img_gray)
            X = pd.DataFrame([feats])[feature_names]

            proba = pipe.predict_proba(X)[0, 1] if hasattr(pipe, "predict_proba") else None
            pred = pipe.predict(X)[0]

            label = "Parkinson's-like pattern" if pred == 1 else "Healthy-like pattern"
            st.metric("Prediction", label, f"model: {bundle['model_name']}")
            if proba is not None:
                st.progress(float(proba))
                st.write(f"Predicted probability of Parkinson's-like pattern: {proba:.2f}")


# ------------------------------------------------------------------ voice tab
with tab_voice:
    st.subheader("Upload a short voice recording")
    st.write(
        "Upload a short .wav recording of a sustained vowel (e.g. saying 'aaaah' for "
        "2-3 seconds). Live browser microphone recording isn't wired up in this demo -- "
        "upload a file instead."
    )

    audio_file = st.file_uploader("Voice recording (.wav)", type=["wav"])

    if st.button("Predict from voice") and audio_file is not None:
        if not VOICE_MODEL_PATH.exists():
            st.error(f"No trained model found at {VOICE_MODEL_PATH}. Run voice_pipeline.py first.")
        else:
            try:
                import parselmouth
                from parselmouth.praat import call
            except ImportError:
                st.error("praat-parselmouth isn't installed. Run: pip install praat-parselmouth")
                st.stop()

            tmp_path = Path("_tmp_recording.wav")
            tmp_path.write_bytes(audio_file.read())

            sound = parselmouth.Sound(str(tmp_path))
            point_process = call(sound, "To PointProcess (periodic, cc)", 75, 500)
            jitter_local = call(point_process, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
            shimmer_local = call(
                [sound, point_process],
                "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6,
            )
            harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
            hnr = call(harmonicity, "Get mean", 0, 0)
            tmp_path.unlink(missing_ok=True)

            bundle = joblib.load(VOICE_MODEL_PATH)
            pipe, feature_names = bundle["pipeline"], bundle["feature_names"]

            # Best-effort mapping: only override columns we can actually estimate live;
            # everything else falls back to a fixed neutral value (0). This is a coarse
            # demo, NOT equivalent to the original MDVP-derived training features --
            # say so explicitly when presenting this.
            row = {name: 0.0 for name in feature_names}
            for name in feature_names:
                lname = name.lower()
                if "jitter" in lname and "%" in lname:
                    row[name] = jitter_local * 100
                elif "shimmer" in lname and "db" not in lname:
                    row[name] = shimmer_local * 100
                elif lname == "hnr":
                    row[name] = hnr

            X = pd.DataFrame([row])[feature_names]
            proba = pipe.predict_proba(X)[0, 1] if hasattr(pipe, "predict_proba") else None
            pred = pipe.predict(X)[0]

            label = "Parkinson's-like voice pattern" if pred == 1 else "Healthy-like voice pattern"
            st.metric("Prediction", label, f"model: {bundle['model_name']}")
            if proba is not None:
                st.progress(float(proba))
                st.write(f"Predicted probability of Parkinson's-like pattern: {proba:.2f}")
            st.caption(
                "Note: only jitter/shimmer/HNR are estimated live from your recording; "
                "the rest of the model's input features are held at a neutral default, "
                "so treat this as an illustrative demo rather than an accurate reading."
            )
