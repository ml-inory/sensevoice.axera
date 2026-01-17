import os
import argparse
from SenseVoiceAx import SenseVoiceAx
import librosa
from download_utils import download_model
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", "-i", required=True, type=str, help="Input audio file"
    )
    parser.add_argument(
        "--language",
        "-l",
        required=False,
        type=str,
        default="auto",
        choices=["auto", "zh", "en", "yue", "ja", "ko"],
    )
    parser.add_argument("--streaming", action="store_true")
    return parser.parse_args()


def main():
    args = get_args()
    print(vars(args))

    input_audio = args.input
    language = args.language
    model_root = download_model("SenseVoice")
    model_root = os.path.join(model_root, "sensevoice_ax650")
    if not args.streaming:
        max_seq_len = 256
        model_path = os.path.join(model_root, "sensevoice.axmodel")
    else:
        max_seq_len = 26
        model_path = os.path.join(model_root, "streaming_sensevoice.axmodel")

    assert os.path.exists(model_path), f"model {model_path} not exist"

    cmvn_file = os.path.join(model_root, "am.mvn")
    bpe_model = os.path.join(model_root, "chn_jpn_yue_eng_ko_spectok.bpe.model")
    token_file = os.path.join(model_root, "tokens.txt")

    model = SenseVoiceAx(
        model_path,
        cmvn_file,
        token_file,
        bpe_model,
        max_seq_len=max_seq_len,
        beam_size=3,
        hot_words=None,
        streaming=args.streaming,
    )

    if not args.streaming:
        asr_res = model.infer(input_audio, language, print_rtf=True)
        print("ASR result: " + asr_res)
    else:
        samples, sr = librosa.load(input_audio, sr=16000)
        samples = (samples * 32768).tolist()
        duration = len(samples) / 16000

        start = time.time()
        step = int(0.1 * sr)
        for i in range(0, len(samples), step):
            is_last = i + step >= len(samples)
            for res in model.stream_infer(samples[i : i + step], is_last, language):
                print(res)

        end = time.time()
        cost_time = end - start

        print(f"RTF: {cost_time / duration}")


if __name__ == "__main__":
    main()
