import edge_tts
import asyncio
import os
import json


clinc_path = "oos-eval/data/data_full.json"
with open(clinc_path, "r") as f:
    data = json.load(f)

# Build test_texts (and the rest, just in case)
test_texts  = [item[0] for item in data["test"]]
test_labels = [item[1] for item in data["test"]]


VOICE      = "en-AU-WilliamNeural"   # <-- swap accent here
OUTPUT_DIR = "audio_recordings"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Sample de 4500 de intents
SAMPLE_SIZE = 4500
texts_to_record = test_texts[:SAMPLE_SIZE]

SEM = asyncio.Semaphore(20)  # 20 concurrent requests

async def tts_single(i, text):
    path = f"{OUTPUT_DIR}/sample_{i:04d}.mp3"
    if os.path.exists(path):
        return
    for attempt in range(5):
        try:
            async with SEM:
                communicate = edge_tts.Communicate(text, VOICE)
                await communicate.save(path)
            return
        except Exception:
            await asyncio.sleep(2 ** attempt)
    if i % 100 == 0:
        print(f"  {i+1}/{SAMPLE_SIZE} done...")

async def generate_all():
    tasks = [tts_single(i, text) for i, text in enumerate(texts_to_record)]
    await asyncio.gather(*tasks)
    print("All audio generated!")

asyncio.run(generate_all())