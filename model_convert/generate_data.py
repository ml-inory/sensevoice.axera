import argparse
import os
from SenseVoiceOnnx import SenseVoiceOnnx
from SenseVoicePth import SenseVoicePth
from model import SenseVoiceSmall
import tarfile
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", type=str, default="output_dir/model.onnx")
    parser.add_argument("--cmvn", type=str, default="output_dir/am.mvn")
    parser.add_argument("--tokens", type=str, default="output_dir/tokens.txt")
    parser.add_argument("--use_torch", action="store_true")
    parser.add_argument("--calib_dataset", type=str, default="calibration_dataset")
    args = parser.parse_args()
    return args


def main():
    args = get_args()
    print(vars(args))

    if args.use_torch:
        model_dir = "iic/SenseVoiceSmall"
        orig_model, kwargs = SenseVoiceSmall.from_pretrained(
            model=model_dir, device="cpu"
        )
        model = SenseVoicePth(
            orig_model, max_seq_len=256, cmvn_file=args.cmvn, token_file=args.tokens
        ).eval()
    else:
        model = SenseVoiceOnnx(args.onnx, args.cmvn, args.tokens)

    en_list = ["../example/en.mp3"]
    ja_list = ["../example/ja.mp3"]
    ko_list = ["../example/ko.mp3"]
    yue_list = ["../example/yue.mp3"]
    zh_list = ["../example/zh.mp3"]
    data_dir = {
        "en": en_list,
        "ja": ja_list,
        "ko": ko_list,
        "yue": yue_list,
        "zh": zh_list,
        "auto": en_list + ja_list + ko_list + yue_list + zh_list,
    }

    os.makedirs(args.calib_dataset, exist_ok=True)

    speech_dir = os.path.join(args.calib_dataset, "speech")
    mask_dir = os.path.join(args.calib_dataset, "mask")
    language_dir = os.path.join(args.calib_dataset, "language")

    os.makedirs(speech_dir, exist_ok=True)
    os.makedirs(mask_dir, exist_ok=True)
    os.makedirs(language_dir, exist_ok=True)

    tf_speech = tarfile.open(f"{args.calib_dataset}/speech.tar.gz", "w:gz")
    tf_mask = tarfile.open(f"{args.calib_dataset}/mask.tar.gz", "w:gz")
    tf_language = tarfile.open(f"{args.calib_dataset}/language.tar.gz", "w:gz")

    data_num = 0

    def save_data(slice_index, speech, mask, language_token):
        np.save(os.path.join(speech_dir, f"{data_num}_slice_{slice_index}.npy"), speech)
        np.save(os.path.join(mask_dir, f"{data_num}_slice_{slice_index}.npy"), mask)
        np.save(
            os.path.join(language_dir, f"{data_num}_slice_{slice_index}.npy"),
            language_token,
        )

    for language in data_dir.keys():
        for audio_path in data_dir[language]:
            text = model.infer(audio_path, language, save_data)
            print(audio_path)
            print(text)
            print()

            data_num += 1

    tf_speech.add(speech_dir)
    tf_mask.add(mask_dir)
    tf_language.add(language_dir)

    tf_speech.close()
    tf_mask.close()
    tf_language.close()


if __name__ == "__main__":
    main()
