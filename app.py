import streamlit as st
import os
import ast
import pandas as pd

# ---------------------------------------------------
# 1️⃣ Load Available Movie Analyses
# ---------------------------------------------------
OUTPUT_DIR = "output"

st.set_page_config(page_title="🎬 Movie Review Analysis Dashboard", layout="wide")

st.title("🎬 Movie Review Emotion & Sentiment Analysis")

# Get all movie folders
movie_folders = [f for f in os.listdir(OUTPUT_DIR) if os.path.isdir(os.path.join(OUTPUT_DIR, f))]

if not movie_folders:
    st.warning("⚠️ No analyzed movie data found in the 'output' folder.")
    st.stop()

selected_movie = st.selectbox("🎥 Choose a Movie Review", movie_folders)

# ---------------------------------------------------
# 2️⃣ Load analysis.txt for selected movie
# ---------------------------------------------------
analysis_file = os.path.join(OUTPUT_DIR, selected_movie, "analysis.txt")

if not os.path.exists(analysis_file):
    st.error("❌ No analysis.txt file found for this movie.")
    st.stop()

with open(analysis_file, "r", encoding="utf-8") as f:
    content = f.read()

# ---------------------------------------------------
# 3️⃣ Parse the content
# ---------------------------------------------------
def parse_analysis(text):
    sections = text.split("\n\n")
    data = {}
    for sec in sections:
        if sec.startswith("Emotion:"):
            data["Emotion"] = ast.literal_eval(sec.replace("Emotion:", "").strip())
        elif sec.startswith("Sentiment:"):
            data["Sentiment"] = ast.literal_eval(sec.replace("Sentiment:", "").strip())
        elif sec.startswith("Intent:"):
            data["Intent"] = ast.literal_eval(sec.replace("Intent:", "").strip())
        elif sec.startswith("Summary:"):
            data["Summary"] = sec.replace("Summary:", "").strip()
    return data

analysis_data = parse_analysis(content)

# ---------------------------------------------------
# 4️⃣ Display in attractive layout
# ---------------------------------------------------
st.subheader(f"🎞 Analysis for: `{selected_movie}`")

# Emotion Table
st.markdown("### 😍 Emotion Analysis")
if "Emotion" in analysis_data:
    emotion_df = pd.DataFrame(analysis_data["Emotion"])
    emotion_df["score"] = emotion_df["score"].round(3)
    st.table(emotion_df)
else:
    st.info("No emotion data found.")

# Sentiment Table
st.markdown("### 💬 Sentiment Analysis")
if "Sentiment" in analysis_data:
    sentiment_df = pd.DataFrame(analysis_data["Sentiment"])
    sentiment_df["score"] = sentiment_df["score"].round(3)
    st.table(sentiment_df)
else:
    st.info("No sentiment data found.")

# Intent Table
st.markdown("### 🎯 Intent Classification")
if "Intent" in analysis_data:
    intent_df = pd.DataFrame({
        "Label": analysis_data["Intent"]["labels"],
        "Confidence": [round(x, 3) for x in analysis_data["Intent"]["scores"]]
    })
    st.bar_chart(intent_df.set_index("Label"))
else:
    st.info("No intent data found.")

# Summary Card
st.markdown("### 🧾 Summary")
st.markdown(
    f"<div style='background-color:#f5f5f5;padding:15px;border-radius:10px;'>"
    f"<b>📝 Summary:</b><br>{analysis_data.get('Summary', 'No summary available.')}"
    f"</div>", unsafe_allow_html=True
)

# ---------------------------------------------------
# 5️⃣ Optional — Show transcript or translated text
# ---------------------------------------------------
st.markdown("### 📜 Additional Files")

transcript_path = os.path.join(OUTPUT_DIR, selected_movie, "transcript.txt")
translated_path = os.path.join(OUTPUT_DIR, selected_movie, "translated.txt")

col1, col2 = st.columns(2)

with col1:
    if os.path.exists(transcript_path):
        with open(transcript_path, "r", encoding="utf-8") as f:
            transcript = f.read()[:2000]
        st.text_area("🎧 Transcript (partial view)", transcript, height=250)
    else:
        st.info("No transcript available.")

with col2:
    if os.path.exists(translated_path):
        with open(translated_path, "r", encoding="utf-8") as f:
            translated = f.read()[:2000]
        st.text_area("🌐 Translated Text (partial view)", translated, height=250)
    else:
        st.info("No translated text available.")

st.success("✅ Analysis visualization complete!")
