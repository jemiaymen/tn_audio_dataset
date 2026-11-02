#%%
import json
from transformers import pipeline

#%%

pipe = pipeline(
    "automatic-speech-recognition",
    model="TuniSpeech-AI/whisper-tunisian-dialect"
)
#%%
result = pipe("../raw_data/1/raw.wav" , return_timestamps=True, language="ar")
#%%
for row in result["chunks"]:
    print(row)
# %%
yt_code = "1"
json_file = {"text" : result["text"],"segments": [] }
for row in result["chunks"]:
    json_file["segments"].append({
        "text": row["text"],
        "start": row["timestamp"][0],
        "end": row["timestamp"][1],
    })

with open(f"../raw_data/{yt_code}/output.json", "w",encoding="utf8" ) as file:
    json.dump(json_file, file, ensure_ascii=False, indent=2)
# %%
def transcribe(yt_code):
    pipe = pipeline(
        "automatic-speech-recognition",
        model="TuniSpeech-AI/whisper-tunisian-dialect"
    )
    result = pipe(f"../raw_data/{yt_code}/raw.wav" , return_timestamps=True, language="ar")
    json_file = {"text" : result["text"],"segments": [] }
    for row in result["chunks"]:
        json_file["segments"].append({
            "text": row["text"],
            "start": row["timestamp"][0],
            "end": row["timestamp"][1],
        })

    with open(f"../raw_data/{yt_code}/output.json", "w",encoding="utf8") as file:
        json.dump(json_file, file, ensure_ascii=False, indent=2)