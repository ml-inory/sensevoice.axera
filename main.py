import os
import argparse
from SenseVoiceAx import SenseVoiceAx
from download_utils import download_model
import librosa
import numpy as np
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str, help="Input audio file")
    parser.add_argument("--language", "-l", required=False, type=str, default="auto", choices=["auto", "zh", "en", "yue", "ja", "ko"])
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()

    input_audio = args.input
    language = args.language
    use_itn = True # 标点符号预测
    model_path_root = download_model("SenseVoice")
    if not args.streaming:
        max_len = 256
        model_path = os.path.join(model_path_root, "sensevoice_ax650", "sensevoice.axmodel")
    else:
        max_len = 26
        model_path = os.path.join(model_path_root, "sensevoice_ax650", "streaming_sensevoice.axmodel")

    assert os.path.exists(model_path), f"model {model_path} not exist"

    print(f"input_audio: {input_audio}")
    print(f"language: {language}")
    print(f"use_itn: {use_itn}")
    print(f"model_path: {model_path}")
    print(f"streaming: {args.streaming}")

    pipeline = SenseVoiceAx(model_path, 
                            max_len=max_len, 
                            beam_size=3,
                            language="auto", 
                            hot_words=None,
                            use_itn=True, 
                            streaming=args.streaming)
    
    if not args.streaming:
        asr_res = pipeline.infer(input_audio, print_rtf=True)
        print("ASR result: " + asr_res)
    else:
        samples, sr = librosa.load(input_audio, sr=16000)
        samples = (samples * 32768).tolist()
        duration = len(samples) / 16000

        start = time.time()
        step = int(0.1 * sr)
        for i in range(0, len(samples), step):
            is_last = i + step >= len(samples)
            for res in pipeline.stream_infer(samples[i : i + step], is_last):
                print(res)
        
        end = time.time()
        cost_time = end - start

        print(f"RTF: {cost_time / duration}")

if __name__ == "__main__":
    main()