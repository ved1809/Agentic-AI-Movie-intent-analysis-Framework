# display_results.py
import streamlit as st
import os

# Streamlit Page Title
st.set_page_config(page_title="🎬 Movie Review NLP Analysis", layout="wide")
st.title("🎧 Movie Review Analysis Dashboard")

# Let user select which output folder to display
base_dir = "output"
if not os.path.exists(base_dir):
    st.error("❌ No output directory found. Please run your analysis script first.")
    st.stop()

folders = [f for f in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, f))]
if not folders:
    st.error("❌ No results found in the output folder.")
    st.stop()

selected_folder = st.selectbox("📂 Select an output folder to view results:", folders)
output_path = os.path.join(base_dir, selected_folder)

st.write("---")

# Function to read text files safely
def read_text_file(file_path):
    if os.path.exists(file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            return f.read()
    return "⚠️ File not found."

# Display transcript
st.subheader("📝 Transcription")
transcript_path = os.path.join(output_path, "transcript.txt")
transcript_text = read_text_file(transcript_path)
st.text_area("Transcribed Text:", transcript_text, height=200)

# Display translation if available
translated_path = os.path.join(output_path, "translated.txt")
if os.path.exists(translated_path):
    st.subheader("🌐 Translation (English)")
    translated_text = read_text_file(translated_path)
    st.text_area("Translated Text:", translated_text, height=200)
else:
    st.info("No translation file found (audio was likely already in English).")

# Display analysis results
st.subheader("🔍 NLP Analysis Results")
analysis_path = os.path.join(output_path, "analysis.txt")
analysis_text = read_text_file(analysis_path)
st.text_area("Analysis Output:", analysis_text, height=250)

# Download buttons
st.write("---")
col1, col2, col3 = st.columns(3)
with col1:
    st.download_button("⬇️ Download Transcript", transcript_text, file_name="transcript.txt")
with col2:
    if os.path.exists(translated_path):
        st.download_button("⬇️ Download Translation", translated_text, file_name="translated.txt")
with col3:
    st.download_button("⬇️ Download Analysis", analysis_text, file_name="analysis.txt")

st.success("✅ All analysis files displayed successfully!")
