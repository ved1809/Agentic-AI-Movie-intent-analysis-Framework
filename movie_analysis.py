import os
import warnings
import torch
import mimetypes
import ffmpeg
from functools import lru_cache

warnings.filterwarnings("ignore")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_TYPE = "float16" if DEVICE == "cuda" else "int8"

print(f"✅ Device: {DEVICE}, Compute type: {COMPUTE_TYPE}")

# ---------------------------------------------------
# 1️⃣ Create Output Directory
# ---------------------------------------------------
def create_output_directory(media_path):
    filename = os.path.splitext(os.path.basename(media_path))[0]
    folder_name = filename.replace("-", "_").replace(" ", "_")
    output_dir = os.path.join("output", folder_name)
    os.makedirs(output_dir, exist_ok=True)
    print(f"📁 Output directory created: {output_dir}")
    return output_dir

# ---------------------------------------------------
# 2️⃣ Load Whisper Model
# ---------------------------------------------------
@lru_cache(maxsize=1)
def get_whisper_model(model_size="small", device=DEVICE, compute_type=COMPUTE_TYPE):
    from faster_whisper import WhisperModel, BatchedInferencePipeline
    print(f"🎧 Loading Whisper model: {model_size} on {device}...")
    model = WhisperModel(model_size, device=device, compute_type=compute_type)
    return BatchedInferencePipeline(model=model)

# ---------------------------------------------------
# 3️⃣ Transformer Pipelines
# ---------------------------------------------------
@lru_cache(maxsize=1)
def get_transformer_models():
    from transformers import pipeline
    print("🤗 Loading Hugging Face models (this may take a moment)...")
    models = {
        "emotion": pipeline("text-classification", model="j-hartmann/emotion-english-distilroberta-base",
                            device=0 if DEVICE == "cuda" else -1),
        "sentiment": pipeline("text-classification", model="cardiffnlp/twitter-roberta-base-sentiment-latest",
                              device=0 if DEVICE == "cuda" else -1),
        "summarizer": pipeline("summarization", model="facebook/bart-large-cnn",
                               device=0 if DEVICE == "cuda" else -1),
        "zero_shot": pipeline("zero-shot-classification", model="facebook/bart-large-mnli",
                              device=0 if DEVICE == "cuda" else -1)
    }
    print("✅ All models loaded!")
    return models

# ---------------------------------------------------
# 4️⃣ Detect File Type
# ---------------------------------------------------
def detect_file_type(file_path):
    mime_type, _ = mimetypes.guess_type(file_path)
    if mime_type and mime_type.startswith("audio"):
        return "audio"
    elif mime_type and mime_type.startswith("video"):
        return "video"
    ext = os.path.splitext(file_path)[1].lower()
    return "video" if ext in [".mp4", ".mov", ".mkv", ".avi"] else "audio"

# ---------------------------------------------------
# 5️⃣ Extract Audio
# ---------------------------------------------------
def extract_audio_from_video(video_path, output_dir):
    output_path = os.path.join(output_dir, "extracted_audio.mp3")
    print(f"🎬 Extracting audio from {os.path.basename(video_path)}...")
    (
        ffmpeg
        .input(video_path)
        .output(output_path, acodec='libmp3lame', qscale=2)
        .overwrite_output()
        .run(quiet=True)
    )
    return output_path

# ---------------------------------------------------
# 6️⃣ Convert to WAV
# ---------------------------------------------------
def convert_to_wav_16k_mono(src_audio_path, output_dir):
    dst_audio_path = os.path.join(output_dir, "audio_16k_mono.wav")
    (
        ffmpeg
        .input(src_audio_path)
        .output(dst_audio_path, ac=1, ar=16000, acodec='pcm_s16le')
        .overwrite_output()
        .run(quiet=True)
    )
    print("🎧 Converted to 16kHz mono WAV")
    return dst_audio_path

# ---------------------------------------------------
# 7️⃣ Transcribe + Translate
# ---------------------------------------------------
def transcribe_and_translate_whisper(wav_file, output_dir, model_size="small"):
    batched_model = get_whisper_model(model_size)
    print("📝 Transcribing audio...")
    segments, info = batched_model.transcribe(wav_file, batch_size=8, task="transcribe", vad_filter=True)
    raw_text = " ".join([seg.text.strip() for seg in segments])
    lang = info.language

    print(f"🌐 Detected language: {lang}")
    transcript_path = os.path.join(output_dir, "transcript.txt")
    with open(transcript_path, "w", encoding="utf-8") as f:
        f.write(raw_text)

    if lang.lower() not in ("en", "english"):
        print("🔁 Translating to English...")
        segments, _ = batched_model.transcribe(wav_file, batch_size=8, task="translate")
        translated = " ".join([seg.text.strip() for seg in segments])
        translated_path = os.path.join(output_dir, "translated.txt")
        with open(translated_path, "w", encoding="utf-8") as f:
            f.write(translated)
    else:
        translated = raw_text
    return translated

# ---------------------------------------------------
# 8️⃣ NLP Analysis
# ---------------------------------------------------
def analyze_emotion_sentiment_intent(text, output_dir):
    models = get_transformer_models()
    emotion_res = models["emotion"](text[:512])
    sentiment_res = models["sentiment"](text[:512])
    summary = models["summarizer"](text[:1000], max_length=100, min_length=20)[0]["summary_text"]
    intents = ["Educational/Tutorial", "Entertainment", "Informative/News", "Motivational", "Review/Opinion", "Story/Narrative"]
    intent_res = models["zero_shot"](text[:1000], candidate_labels=intents)

    print("✨ Emotion:", emotion_res[0])
    print("💬 Sentiment:", sentiment_res[0])
    print("🎯 Intent:", intent_res["labels"][0])
    print("🧩 Summary:", summary)

    with open(os.path.join(output_dir, "analysis.txt"), "w", encoding="utf-8") as f:
        f.write(f"Emotion: {emotion_res}\n\nSentiment: {sentiment_res}\n\nIntent: {intent_res}\n\nSummary:\n{summary}")

    return emotion_res, sentiment_res, summary, intent_res

# ---------------------------------------------------
# 🏁 MAIN LOOP FOR MULTIPLE VIDEOS
# ---------------------------------------------------
if __name__ == "__main__":
    MOVIE_DIR = r"D:\NLP Mini Project\Movie"
    print(f"🎞 Scanning for video files in {MOVIE_DIR}...")

    video_files = [os.path.join(MOVIE_DIR, f) for f in os.listdir(MOVIE_DIR)
                   if f.lower().endswith((".mp4", ".mkv", ".avi", ".mov"))]

    if not video_files:
        print("⚠️ No video files found.")
    else:
        for idx, MEDIA_PATH in enumerate(video_files, 1):
            print(f"\n{'='*50}\n▶️ Processing Video {idx}/{len(video_files)}: {os.path.basename(MEDIA_PATH)}\n{'='*50}")
            output_dir = create_output_directory(MEDIA_PATH)
            file_type = detect_file_type(MEDIA_PATH)
            audio_file = extract_audio_from_video(MEDIA_PATH, output_dir) if file_type == "video" else MEDIA_PATH
            wav_file = convert_to_wav_16k_mono(audio_file, output_dir)
            transcribed_text = transcribe_and_translate_whisper(wav_file, output_dir)
            analyze_emotion_sentiment_intent(transcribed_text, output_dir)
            print(f"✅ Completed analysis for: {MEDIA_PATH}\nResults saved in: {output_dir}")
