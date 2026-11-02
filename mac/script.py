#%%
import os
import json
import argparse
from abc import ABC, abstractmethod
from yt_dlp import YoutubeDL
from mlx_audio.stt.generate import generate_transcription
from pydub import AudioSegment
import jsonlines

#%%
class YoutubeAudioDownloader:
    def __init__(self, yt_code):
        self.yt_code = yt_code
        self.yt_url = f'https://www.youtube.com/watch?v={yt_code}'
        os.makedirs(f'../raw_data/{yt_code}', exist_ok=True)
        self.download()

    def download(self):
        ydl_opts = {
            'outtmpl': f'../raw_data/{self.yt_code}/raw',
            'format': 'bestaudio/best',
            'postprocessors': [{
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
            }],
            'postprocessor_args': {
                'ExtractAudio': ['-ar', '16000']
            },
        }
        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.yt_url])

class Transcribe:
    def __init__(self, yt_code, model="TuniSpeech-AI/whisper-tunisian-dialect", format="json"):
        self.yt_code = yt_code
        self.audio_path = f'../raw_data/{yt_code}/raw.wav'
        self.transcription_path = f'../raw_data/{yt_code}/output'
        self.transcribe(model, format)


    def transcribe(self,model,format):
        result = generate_transcription(
            model=model,
            audio= self.audio_path,
            format=format,
            verbose=True,
            language="ar",
            output_file=self.transcription_path )
        return result

class AudioSegmenter:
    def __init__(self, yt_code):
        self.yt_code = yt_code
        self.audio_path = f'../raw_data/{yt_code}/raw.wav'
        self.segments_dir = f'../corpus/{yt_code}/wavs'
        os.makedirs(self.segments_dir, exist_ok=True)
        self.audio = AudioSegment.from_wav(self.audio_path)

    def segment_audio(self,name,start_time, end_time=None):
        wav_path = f'../corpus/{self.yt_code}/wavs/{name}.wav'
        start_sample = int(start_time * 1000)  
        end_sample = int(end_time * 1000) if end_time is not None else len(self.audio)
        segment = self.audio[start_sample:end_sample]
        segment.export(wav_path, format='wav')

class TransformDataset(ABC):
    
    def __init__(self, yt_code:str):
        self.yt_code = yt_code  
        self.segmenter = AudioSegmenter(yt_code)
    @abstractmethod
    def transform(self):
        pass

class Qwen3ASRTransform(TransformDataset):

    def transform(self):
        data = json.load(open(f'../raw_data/{self.yt_code}/output.json', 'r', encoding="utf-8"))
        with jsonlines.open(f'../corpus/{self.yt_code}/metadata.jsonl' , mode="w") as writer:
            for row in data["segments"]:
                start_time = row['start']
                end_time = row['end']
                text = row['text']
                name = f"{start_time:.2f}_{end_time:.2f}"
                self.segmenter.segment_audio(name, start_time, end_time)
                writer.write({"audio": f'/corpus/{self.yt_code}/wavs/{name}.wav', "text": f"language Arabic<asr_text>{text}"})

#%%
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube audio, transcribe it, and transform the dataset.")
    parser.add_argument("-y","--yt_code", type=str, help="The YouTube video code.")
    argv = parser.parse_args()
    if not argv.yt_code:
        raise ValueError("YouTube video code is required. Use -y or --yt_code to provide it.")
    yt_code = argv.yt_code
    YoutubeAudioDownloader(yt_code)
    Transcribe(yt_code,"TuniSpeech-AI/whisper-tunisian-dialect", "json")
    Qwen3ASRTransform(yt_code).transform()

#%%
yt_code = "FePX4oe19E0"
Transcribe(yt_code,"TuniSpeech-AI/whisper-tunisian-dialect", "json")
Qwen3ASRTransform(yt_code).transform()
