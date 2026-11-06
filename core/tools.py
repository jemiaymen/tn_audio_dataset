import os
import json
import csv
import argparse
from abc import ABC, abstractmethod
import math
from pathlib import Path
from yt_dlp import YoutubeDL
from pydub import AudioSegment
import jsonlines
from faster_whisper import WhisperModel



def get_confidence_percentage(avg_logprob):
    probability = min(1.0, math.exp(avg_logprob))
    return round(probability * 100, 2)

class AudioDownloader:
    def __init__(self, code,default_path='../raw_data'):
        self.code =code
        self.default_path = default_path
        self.yt_url = f'https://www.youtube.com/watch?v={code}'
        self.fb_url = f'https://www.facebook.com/reel/{code}'
        self.ig_url = f'https://www.instagram.com/reels/{code}/'
        os.makedirs(f'{default_path}/{code}', exist_ok=True)

    def download_yt(self):
        ydl_opts = {
            'outtmpl': f'{self.default_path}/{self.code}/raw',
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
                'outtmpl': f'{self.default_path}/{self.code}/raw',
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
            'outtmpl': f'{self.default_path}/{self.code}/raw',
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
    def __init__(self, code, model="../models/whisper-tunisian-dialect-ct2"):
        self.code = code
        self.audio_path = f'../raw_data/{self.code}/raw.wav'
        self.transcription_path = f'../raw_data/{self.code}/output'
        self.transcribe(model)


    def transcribe(self,model):

        wmodel = WhisperModel(model, device="cpu", compute_type="int8")
        result, info = wmodel.transcribe(f"../raw_data/{self.code}/raw.wav", 
                                        language="ar", 
                                        vad_filter=True,
                                        word_timestamps=True,
                                        log_prob_threshold=-1.0)
        # more then 85%
        high_confidence = {"total" : 0.0 ,"segments": []  }
        # between 50 and less 85%
        low_confidence = {"segments": [] }
        total_s = 0.0
        for segment in result:

            if segment.words:
                word_logprobs = [math.log(max(w.probability, 1e-5)) for w in segment.words]
                true_segment_avg_logprob = sum(word_logprobs) / len(word_logprobs)
            else:
                true_segment_avg_logprob = segment.avg_logprob

            confidence_pct = get_confidence_percentage(true_segment_avg_logprob)

            if confidence_pct >= 85.0:
                high_confidence["segments"].append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "logprob" : true_segment_avg_logprob,
                    "confidence_percentage" : confidence_pct
                })
                total_s += (segment.end - segment.start)

            elif confidence_pct >= 50.0:
                low_confidence["segments"].append({
                    "text": segment.text,
                    "start": segment.start,
                    "end": segment.end,
                    "logprob" : true_segment_avg_logprob,
                    "confidence_percentage" : confidence_pct
                })

        high_confidence["total"] = total_s

        with open(f"../raw_data/{self.code}/output.json", "w",encoding="utf8") as high , open(f"../raw_data/{self.code}/low.json", "w",encoding="utf8") as low:
            json.dump(high_confidence, high, ensure_ascii=False, indent=2)
            json.dump(low_confidence, low, ensure_ascii=False, indent=2)

        return result        

class AudioSegmenter:
    def __init__(self, path_in):

        self.audio_path = f'{path_in}/raw.wav'
        self.audio = AudioSegment.from_wav(self.audio_path)

    def segment_audio(self,path_out,start_time, end_time=None):
        wav_path = f'{path_out}.wav'
        start_sample = int(start_time * 1000)  
        end_sample = int(end_time * 1000) if end_time is not None else len(self.audio)
        segment = self.audio[start_sample:end_sample]
        segment.export(wav_path, format='wav')

class TransformDataset(ABC):
    
    def __init__(self, base_dir:str):
        p = Path(base_dir)
        self.base_dir = base_dir
        self.folders = [x for x in p.iterdir() if x.is_dir()]  
        
    @abstractmethod
    def transform(self):
        pass

class Qwen3ASRTransform(TransformDataset):

    def transform(self,i=1):
        qwen_path = '../qwen3_dataset'
        os.makedirs(qwen_path,exist_ok=True)
        for folder in self.folders:

            data = json.load(open(f'{folder}/output.json', 'r', encoding="utf-8"))
            segmenter = AudioSegmenter(str(folder))
            metadata = []
            with jsonlines.open(f'{qwen_path}/metadata.jsonl' , mode="a") as writer:
                for row in data["segments"]:
                    start_time = row['start']
                    end_time = row['end']
                    text = row['text']
                    if not text.strip() :
                        continue
                    name = f"{i}"
                    segmenter.segment_audio(f'{qwen_path}/{name}', start_time, end_time)
                    metadata.append({"audio": f'{name}.wav', "text": f"language Arabic<asr_text>{text}"})
                    i+= 1
                writer.write_all(metadata)

class HFDatasetTransform(TransformDataset):
    def transform(self,i=1):
        train_path = '../hf_dataset/train'
        test_path = '../hf_dataset/test'
        os.makedirs(train_path, exist_ok=True)
        os.makedirs(test_path, exist_ok=True)
        for folder in self.folders:
            data = json.load(open(f'{folder}/output.json', 'r', encoding="utf-8"))
            segmenter = AudioSegmenter(str(folder))
            with open(f'{train_path}/metadata.csv',mode="a",newline='', encoding="utf8") as trainfile , open(f'{test_path}/metadata.csv',mode="a",newline='', encoding="utf8") as testfile :

                train_writer = csv.writer(trainfile)
                test_writer = csv.writer(testfile)
                train_data = []
                test_data = []
                for row in data["segments"]:
                    start_time = row['start']
                    end_time = row['end']
                    text = row['text']
                    if not text.strip():
                        continue
                    # 10 % test 90 % train
                    if i % 10 == 0:
                        segmenter.segment_audio(f'{test_path}/{i}',start_time,end_time)
                        test_data.append([f'{i}.wav' ,  text])
                    else:
                        segmenter.segment_audio(f'{train_path}/{i}',start_time,end_time)
                        train_data.append([f'{i}.wav' ,  text])
                    i+= 1
                train_writer.writerows(train_data)
                test_writer.writerows(test_data)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download audio, transcribe it, and transform the dataset.")
    parser.add_argument("-y","--yt_code", type=str, help="The YouTube video code.")
    parser.add_argument("-f","--fb_code", type=str, help="The Facebook video code.")
    parser.add_argument("-i","--ig_code", type=str, help="The Instagram video code.")
    parser.add_argument("-g" ,"--hugface", action="store_true" , help ="Activate hugeface dataset transformation")
    parser.add_argument("-q" , "--qwen", action="store_true", help ="Activate Qwen dataset transformation")
    parser.add_argument("-t" , "--trans", action="store_true", help ="Activate Transcribe")
    parser.add_argument("-n" , "--no_download_need", action="store_true", help ="-n or --no_download_need for exist wav. ")
    parser.add_argument("-c","--counter", type=int,default=1, help="Dataset transform counter.")
    parser.add_argument("-d","--default_path", type=str, default="../raw_data", help="Default path to download audio.")
    argv = parser.parse_args()

    code = ''
    default_path = argv.default_path
    if not argv.yt_code and not argv.fb_code and not argv.ig_code and argv.no_download_need is False:
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
        Qwen3ASRTransform(default_path).transform(argv.counter)
    if argv.hugface:
        HFDatasetTransform(default_path).transform(argv.counter)
        