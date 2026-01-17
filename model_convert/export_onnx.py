import argparse
import os
import torch
from model import SenseVoiceSmall
from SenseVoicePth import SenseVoicePth
import random
import onnx
from typing import Dict, Any
import shutil
import sentencepiece as spm
import numpy as np


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_seq_len", type=int, default=256)
    parser.add_argument("--output_dir", type=str, default="output_dir")
    parser.add_argument("--onnx_name", type=str, default="model.onnx")
    parser.add_argument("--opset", type=int, default=17)
    args = parser.parse_args()
    return args


def add_meta_data(filename: str, meta_data: Dict[str, Any]):
    """Add meta data to an ONNX model. It is changed in-place.

    Args:
      filename:
        Filename of the ONNX model to be changed.
      meta_data:
        Key-value pairs.
    """
    model = onnx.load(filename)

    while len(model.metadata_props):
        model.metadata_props.pop()

    for key, value in meta_data.items():
        meta = model.metadata_props.add()
        meta.key = key
        meta.value = str(value)

    onnx.save(model, filename)


def sequence_mask(max_seq_len, actual_seq_len):
    mask = np.zeros((1, 1, max_seq_len), dtype=np.int32)
    mask[:, :, :actual_seq_len] = 1
    return mask


def main():
    args = get_args()
    print(vars(args))

    max_seq_len = args.max_seq_len

    model_dir = "iic/SenseVoiceSmall"
    orig_model, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cpu")

    os.makedirs(args.output_dir, exist_ok=True)

    origin_model_path = os.path.dirname(kwargs.get("init_param"))
    shutil.copy(
        os.path.join(origin_model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"),
        args.output_dir,
    )
    shutil.copy(os.path.join(origin_model_path, "am.mvn"), args.output_dir)

    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(origin_model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"))

    tokens = [sp.id_to_piece(i).replace("▁", " ") for i in range(sp.vocab_size())]
    with open(os.path.join(args.output_dir, "tokens.txt"), "w") as f:
        for t in tokens:
            f.write(f"{t}\n")

    model = SenseVoicePth(orig_model, max_seq_len=max_seq_len).eval()

    with torch.no_grad():
        speech = torch.randn(1, args.max_seq_len, 560, dtype=torch.float32)
        speech_len = random.randint(5, args.max_seq_len)
        mask = torch.from_numpy(sequence_mask(max_seq_len + 4, speech_len))
        language_token = torch.IntTensor([model.get_language_token("auto")])

        input_names = ["speech", "mask", "language"]
        output_names = ["ctc_logits", "encoder_out_lens"]

        onnx_filename = os.path.join(args.output_dir, args.onnx_name)

        torch.onnx.export(
            model,
            (speech, mask, language_token),
            onnx_filename,
            opset_version=args.opset,
            input_names=input_names,
            output_names=output_names,
            # do_constant_folding=True,
            # export_params=True,
            dynamic_axes=None,
        )

        meta_data = {
            "max_seq_len": args.max_seq_len,
            "vocab_size": sp.vocab_size(),
            "unk_symbol": "<unk>",
        }
        add_meta_data(filename=onnx_filename, meta_data=meta_data)

        print(f"Input name: {input_names}")
        print(f"Output names: {output_names}")
        print(f"Export model to {onnx_filename}")


if __name__ == "__main__":
    main()
