import whisper
import json
import os

os.environ["PATH"] += r";C:\Users\alexb\ffmpeg\bin"

AUDIO_DIR      = "audio_recordings"
OUTPUT_PATH    = r"C:\Users\alexb\Downloads\noisy_test_sets.json"
MODEL_SIZE     = "base"   # tiny | base | small | medium | large
SAMPLE_SIZE    = 4500

audio_paths = [f"{AUDIO_DIR}/sample_{i:04d}.mp3" for i in range(SAMPLE_SIZE)]

print(f"Loading Whisper model: {MODEL_SIZE}...")
model = whisper.load_model(MODEL_SIZE)

results = []
for i, audio_path in enumerate(audio_paths):
    try:
        result      = model.transcribe(audio_path, language="en")
        transcribed = result["text"].strip()
        results.append(transcribed)
    except Exception as e:
        print(f"[{i+1}] Failed: {e} — keeping empty string")
        results.append("")

    if (i + 1) % 100 == 0:
        print(f"  {i+1}/{SAMPLE_SIZE} done...")

print("Saving to noisy_test_sets.json...")
with open(OUTPUT_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)

data["whisper"] = results

with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
    json.dump(data, f, ensure_ascii=False, indent=2)

print(f"Done! 'whisper' key added with {len(results)} entries.")
