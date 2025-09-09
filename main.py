import os, sys
import argparse
from SenseVoiceAx import SenseVoiceAx
from tokenizer import SentencepiecesTokenizer
from print_utils import rich_transcription_postprocess, rich_print_asr_res
from download_utils import download_model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, type=str, help="Input audio file")
    parser.add_argument("--language", "-l", required=False, type=str, default="auto", choices=["auto", "zh", "en", "yue", "ja", "ko"])
    return parser.parse_args()


def main():
    args = get_args()

    input_audio = args.input
    language = args.language
    use_itn = True # 标点符号预测
    max_len = 256

    model_path_root = download_model("SenseVoice")
    model_path = os.path.join(model_path_root, "sensevoice_ax650", "sensevoice.axmodel")
    bpemodel = os.path.join(model_path_root, "chn_jpn_yue_eng_ko_spectok.bpe.model")

    assert os.path.exists(model_path), f"model {model_path} not exist"

    print(f"input_audio: {input_audio}")
    print(f"language: {language}")
    print(f"use_itn: {use_itn}")
    print(f"model_path: {model_path}")

    tokenizer = SentencepiecesTokenizer(bpemodel=bpemodel)
    pipeline = SenseVoiceAx(model_path, 
                            max_len=max_len,
                            language=language, 
                            use_itn=use_itn, 
                            tokenizer=tokenizer)
    asr_res = pipeline.infer(input_audio, print_rtf=True)
    print([rich_transcription_postprocess(i) for i in asr_res])
    # rich_print_asr_res(asr_res)

if __name__ == "__main__":
    main()