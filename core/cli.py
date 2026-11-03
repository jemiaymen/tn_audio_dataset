import argparse
import platform

from ..core.tools import (YoutubeAudioDownloader,
                    AudioSegment,
                    Qwen3ASRTransform,
                    HFDatasetTransform,
                    Transcribe)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download YouTube audio, transcribe it, and transform the dataset.")
    parser.add_argument("-y","--yt_code", type=str, help="The YouTube video code.")
    parser.add_argument("-f" ,"--hugface", type=bool, default=False, help ="Activate hugeface dataset transformation")
    argv = parser.parse_args()
    if not argv.yt_code:
        raise ValueError("YouTube video code is required. Use -y or --yt_code to provide it.")
    yt_code = argv.yt_code
    YoutubeAudioDownloader(yt_code)
    transcribe(yt_code)
    Qwen3ASRTransform(yt_code).transform()
    if argv.hugface:
        HFDatasetTransform(yt_code)
        