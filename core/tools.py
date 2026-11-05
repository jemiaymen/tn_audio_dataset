import os
import json
import platform
import csv
import uuid
import argparse
from abc import ABC, abstractmethod
import math

import numpy as np
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import jsonlines
from faster_whisper import WhisperModel


current_os = platform.system()

if current_os == "Darwin":
    from mlx_audio.stt.generate import generate_transcription
else:
    from transformers import pipeline

def get_confidence_percentage(avg_logprob):
    probability = min(1.0, math.exp(avg_logprob))
    return round(probability * 100, 2)

class AudioDownloader:
    def __init__(self, code):
        self.code =code
        self.yt_url = f'https://www.youtube.com/watch?v={code}'
        self.fb_url = f'https://www.facebook.com/reel/{code}'
        self.ig_url = f'https://www.instagram.com/reels/{code}/'
        os.makedirs(f'../raw_data/{code}', exist_ok=True)

    def download_yt(self):
        ydl_opts = {
            'outtmpl': f'../raw_data/{self.code}/raw',
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

    def download_fb(self):
            ydl_opts = {
                'outtmpl': f'../raw_data/{self.code}/raw',
                'format': 'bestaudio/best',
                'postprocessors': [
                    {
                    'key': 'FFmpegExtractAudio',
                    'preferredcodec': 'wav',
                    } 
                ],
                'postprocessor_args': {
                    'ExtractAudio': ['-ar', '16000']
                },
                'ignoreerrors': False,
                'logtostderr': False,
                'quiet': False
            }

            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([self.fb_url])

    def download_ig(self):
        ydl_opts = {
            'outtmpl': f'../raw_data/{self.code}/raw',
            'format': 'bestaudio/best',
            'postprocessors': [
                {
                'key': 'FFmpegExtractAudio',
                'preferredcodec': 'wav',
                } 
            ],
            'postprocessor_args': {
                'ExtractAudio': ['-ar', '16000']
            },
            'ignoreerrors': False,
            'logtostderr': False,
            'quiet': False
        }

        with YoutubeDL(ydl_opts) as ydl:
            ydl.download([self.ig_url])

class Transcribe:
    def __init__(self, code, model="../models/whisper-tunisian-dialect-ct2", format="json"):
        self.code = code
        self.audio_path = f'../raw_data/{self.code}/raw.wav'
        self.transcription_path = f'../raw_data/{self.code}/output'
        self.transcribe(model, format)


    def transcribe(self,model,format):
        if current_os == "Darwin":
            result = generate_transcription(
                model=model,
                audio= self.audio_path,
                format=format,
                verbose=True,
                language="ar",
                output_file=self.transcription_path )
            return result

        wmodel = WhisperModel(model, device="cpu", compute_type="int8")

        result, info = wmodel.transcribe(f"../raw_data/{self.code}/raw.wav", 
                                        language="ar", 
                                        vad_filter=True,
                                        word_timestamps=True,
                                        log_prob_threshold=-1.0)

        json_file = {"segments": [] }
        for segment in result:

            if segment.words:
                word_logprobs = [math.log(max(w.probability, 1e-5)) for w in segment.words]
                true_segment_avg_logprob = sum(word_logprobs) / len(word_logprobs)
            else:
                true_segment_avg_logprob = segment.avg_logprob

            confidence_pct = get_confidence_percentage(true_segment_avg_logprob)

            json_file["segments"].append({
                            "text": segment.text,
                            "start": segment.start,
                            "end": segment.end,
                            "logprob" : true_segment_avg_logprob,
                            "confidence_percentage" : confidence_pct
                        }) 
    
        with open(f"../raw_data/{self.code}/output.json", "w",encoding="utf8") as file:
            json.dump(json_file, file, ensure_ascii=False, indent=2)

        return result        

class AudioSegmenter:
    def __init__(self, code):
        self.code = code
        self.audio_path = f'../raw_data/{code}/raw.wav'
        self.segments_dir = f'../corpus/{code}/wavs'
        os.makedirs(self.segments_dir, exist_ok=True)
        self.audio = AudioSegment.from_wav(self.audio_path)
        with open(f'../raw_data/{code}/len.txt','w',encoding="utf8") as file:
            file.write( f'{self.audio.duration_seconds:.2f}\n')

    def segment_audio(self,name,start_time, end_time=None,base_path=None):
        if base_path:
            wav_path = base_path + f"{name}.wav"
        else:
            wav_path = f'../corpus/{self.code}/wavs/{name}.wav'
        start_sample = int(start_time * 1000)  
        end_sample = int(end_time * 1000) if end_time is not None else len(self.audio)
        segment = self.audio[start_sample:end_sample]
        segment.export(wav_path, format='wav')

class TransformDataset(ABC):
    
    def __init__(self, code:str):
        self.code = code  
        self.segmenter = AudioSegmenter(code)
    @abstractmethod
    def transform(self):
        pass

class Qwen3ASRTransform(TransformDataset):

    def transform(self):
        data = json.load(open(f'../raw_data/{self.code}/output.json', 'r', encoding="utf-8"))
        with jsonlines.open(f'../corpus/{self.code}/metadata.jsonl' , mode="w") as writer:
            for row in data["segments"]:
                start_time = row['start']
                end_time = row['end']
                text = row['text']
                if not text.strip() :
                    continue
                name = f"{start_time:.2f}_{end_time:.2f}"
                self.segmenter.segment_audio(name, start_time, end_time)
                writer.write({"audio": f'/corpus/{self.code}/wavs/{name}.wav', "text": f"language Arabic<asr_text>{text}"})

class HFDatasetTransform(TransformDataset):
    def transform(self):
        os.makedirs('../hf_dataset/train/', exist_ok=True)
        os.makedirs('../hf_dataset/test/', exist_ok=True)
        data = json.load(open(f'../raw_data/{self.code}/output.json', 'r', encoding="utf-8"))
        with open(f'../hf_dataset/train/metadata.csv',mode="a",newline='', encoding="utf8") as trainfile , open(f'../hf_dataset/test/metadata.csv',mode="a",newline='', encoding="utf8") as testfile :

            train_writer = csv.writer(trainfile)
            test_writer = csv.writer(testfile)
            i = 0
            for row in data["segments"]:
                start_time = row['start']
                end_time = row['end']
                text = row['text']
                if not text.strip():
                    continue
                unique_id = uuid.uuid4()
                name = unique_id.hex
                # 10 % test 90 % train
                if i % 10 == 0:
                    self.segmenter.segment_audio(name,start_time,end_time,base_path='../hf_dataset/test/')
                    test_writer.writerow([f'{name}.wav',text])
                else:
                    self.segmenter.segment_audio(name,start_time,end_time,base_path='../hf_dataset/train/')
                    train_writer.writerow([f'{name}.wav',text])
                i+= 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audio, transcribe it, and transform the dataset.")
    parser.add_argument("-y","--yt_code", type=str, help="The YouTube video code.")
    parser.add_argument("-f","--fb_code", type=str, help="The Facebook video code.")
    parser.add_argument("-i","--ig_code", type=str, help="The Instagram video code.")
    parser.add_argument("-g" ,"--hugface", action="store_true" , help ="Activate hugeface dataset transformation")
    parser.add_argument("-q" , "--qwen", action="store_true", help ="Activate Qwen dataset transformation")
    parser.add_argument("-t" , "--trans", action="store_true", help ="Activate Transcribe")
    parser.add_argument("-n" , "--no_download_need", action="store_true", help ="-n or --no_download_need for exist wav. ")
    argv = parser.parse_args()

    code = ''
    if not argv.yt_code and not argv.fb_code and not argv.ig_code:
        raise ValueError("YouTube, Facebook or Instagram video code is required. Use -y | --yt_code, -f | --fb_code or -i --ig_code .")

    
    if argv.yt_code:
        code = argv.yt_code
        if argv.no_download_need is False:
            AudioDownloader(code).download_yt()
    elif argv.fb_code:
        code = argv.fb_code
        if argv.no_download_need is False:
            AudioDownloader(code).download_fb()
    elif argv.ig_code:
        code = argv.ig_code
        if argv.no_download_need is False:
            AudioDownloader(code).download_ig()
    if argv.trans:
        Transcribe(code)
    if argv.qwen:
        Qwen3ASRTransform(code).transform()
    if argv.hugface:
        HFDatasetTransform(code).transform()
        