#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""SenseVoiceSmall 静态 ONNX 导出（适配 funasr>=1.3）。

输入: speech (1, max_seq_len, 560) FP32, mask (1,1,max_seq_len+4) S32, language (1) S32
输出: ctc_logits (1, max_seq_len+4, vocab) FP32, encoder_out_lens (1) S32
"""
import argparse, json, os, random, shutil
import numpy as np
import torch
from torch import nn

from funasr import AutoModel
import sentencepiece as spm
import onnx
import onnxruntime as ort


def get_args():
    p = argparse.ArgumentParser()
    p.add_argument("--max_seq_len", type=int, default=256)
    p.add_argument("--model_dir", type=str, default="iic/SenseVoiceSmall")
    p.add_argument("--output_dir", type=str, default="output_dir")
    p.add_argument("--onnx_name", type=str, default="model.onnx")
    p.add_argument("--opset", type=int, default=17)
    return p.parse_args()


def sequence_mask(max_len: int, actual_len: int) -> np.ndarray:
    m = np.zeros((1, 1, max_len), dtype=np.int32)
    m[:, :, :actual_len] = 1
    return m


class SenseVoiceExport(nn.Module):
    """静态导出包装：把 query token 拼接到 speech 前，直接吃外部 mask。"""

    def __init__(self, orig_model: nn.Module, max_seq_len: int = 256):
        super().__init__()
        self.orig = orig_model
        self.max_seq_len = max_seq_len
        self.query_num = 4
        wo_itn = orig_model.textnorm_dict["withitn"]  # 非流式: 带标点
        with torch.no_grad():
            textnorm_query = orig_model.embed(torch.LongTensor([wo_itn])).unsqueeze(1)
            event_emo_query = orig_model.embed(torch.LongTensor([[1, 2]]))
        self.register_buffer("textnorm_query", textnorm_query.detach())
        self.register_buffer("event_emo_query", event_emo_query.detach())

    def forward(self, speech: torch.Tensor, mask: torch.Tensor, language: torch.Tensor):
        enc = self.orig.encoder
        language_query = self.orig.embed(language.long()).unsqueeze(1)
        input_query = torch.cat((self.textnorm_query, language_query, self.event_emo_query), dim=1)
        xs = torch.cat((input_query, speech), dim=1)

        xs = xs * (enc.output_size() ** 0.5)
        xs = enc.embed(xs)
        masks = mask
        for layer in enc.encoders0:
            outs = layer(xs, masks)
            xs, masks = outs[0], outs[1]
        for layer in enc.encoders:
            outs = layer(xs, masks)
            xs, masks = outs[0], outs[1]
        xs = enc.after_norm(xs)
        olens = masks.squeeze(1).sum(1).int()
        for layer in enc.tp_encoders:
            outs = layer(xs, masks)
            xs, masks = outs[0], outs[1]
        xs = enc.tp_norm(xs)

        ctc_logits = self.orig.ctc.ctc_lo(xs)
        return ctc_logits, olens


def add_meta(filename: str, meta: dict):
    m = onnx.load(filename)
    while len(m.metadata_props):
        m.metadata_props.pop()
    for k, v in meta.items():
        p = m.metadata_props.add()
        p.key, p.value = k, str(v)
    onnx.save(m, filename)


def main():
    args = get_args()
    os.makedirs(args.output_dir, exist_ok=True)

    print("[1/5] 加载 SenseVoiceSmall 权重 ...")
    orig_model, kwargs = AutoModel.build_model(model=args.model_dir, trust_remote_code=True, device="cpu")
    model = SenseVoiceExport(orig_model, max_seq_len=args.max_seq_len).eval()

    origin_model_path = os.path.dirname(kwargs.get("init_param"))
    for f in ["chn_jpn_yue_eng_ko_spectok.bpe.model", "am.mvn"]:
        shutil.copy(os.path.join(origin_model_path, f), args.output_dir)
    sp = spm.SentencePieceProcessor()
    sp.load(os.path.join(origin_model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"))
    tokens = [sp.id_to_piece(i).replace("▁", " ") for i in range(sp.vocab_size())]
    with open(os.path.join(args.output_dir, "tokens.txt"), "w") as f:
        f.writelines(f"{t}\n" for t in tokens)

    print("[2/5] Torch 前向 sanity check (vs funasr encoder 原生 forward) ...")
    with torch.no_grad():
        speech = torch.randn(1, args.max_seq_len, 560, dtype=torch.float32)
        real_len = random.randint(5, args.max_seq_len)
        mask = torch.from_numpy(sequence_mask(args.max_seq_len + 4, real_len))
        language = torch.IntTensor([0])
        out, olens = model(speech, mask, language)
        enc = orig_model.encoder
        lang_q = orig_model.embed(language.long()).unsqueeze(1)
        q = torch.cat((model.textnorm_query, lang_q, model.event_emo_query), dim=1)
        xs = torch.cat((q, speech), dim=1)
        ref, ref_olens = enc(xs, torch.tensor([real_len], dtype=torch.int32))
        ref = orig_model.ctc.ctc_lo(ref)
        cos = float(torch.nn.functional.cosine_similarity(out.flatten(), ref.flatten(), dim=0))
        print(f"    torch-vs-native cosine={cos:.6f}, olens {olens.tolist()} vs {ref_olens.tolist()}")
        assert cos > 0.9999 and olens.item() == ref_olens.item()

    print("[3/5] torch.onnx.export (静态 shape) ...")
    onnx_path = os.path.join(args.output_dir, args.onnx_name)
    torch.onnx.export(
        model,
        (speech, mask, language),
        onnx_path,
        opset_version=args.opset,
        input_names=["speech", "mask", "language"],
        output_names=["ctc_logits", "encoder_out_lens"],
        dynamic_axes=None,
    )
    add_meta(onnx_path, {"max_seq_len": args.max_seq_len, "vocab_size": sp.vocab_size(),
                         "unk_symbol": "<unk>"})
    onnx.checker.check_model(onnx.load(onnx_path))
    print("    onnx.checker OK")

    print("[4/5] ONNX Runtime 对分 ...")
    sess = ort.InferenceSession(onnx_path, providers=["CPUExecutionProvider"])
    cos_sum, n = 0.0, 3
    for i in range(n):
        with torch.no_grad():
            s = torch.randn(1, args.max_seq_len, 560, dtype=torch.float32)
            rl = random.randint(10, args.max_seq_len)
            m = torch.from_numpy(sequence_mask(args.max_seq_len + 4, rl))
            lg = torch.IntTensor([i % 5])
            torch_ctc, torch_len = model(s, m, lg)
        onnx_ctc, onnx_len = sess.run(None, {"speech": s.numpy(), "mask": m.numpy(),
                                              "language": lg.numpy()})
        c = float(torch.nn.functional.cosine_similarity(
            torch_ctc.flatten(), torch.from_numpy(onnx_ctc).flatten(), dim=0))
        cos_sum += c
        print(f"    case {i}: cosine={c:.6f} len={torch_len.item()}/{onnx_len[0]}")
        assert c > 0.9999 and torch_len.item() == int(onnx_len[0])
    print(f"    mean cosine={cos_sum/n:.6f}")

    print("[5/5] 写 model_meta.json ...")
    meta = {
        "model_name": "sensevoice-small",
        "framework": "funasr",
        "max_seq_len": args.max_seq_len,
        "inputs": [
            {"name": "speech", "shape": [1, args.max_seq_len, 560], "dtype": "float32", "layout": "NTC"},
            {"name": "mask", "shape": [1, 1, args.max_seq_len + 4], "dtype": "int32", "layout": "NTC"},
            {"name": "language", "shape": [1], "dtype": "int32", "layout": "N"},
        ],
        "outputs": [
            {"name": "ctc_logits", "shape": [1, args.max_seq_len + 4, sp.vocab_size()], "dtype": "float32"},
            {"name": "encoder_out_lens", "shape": [1], "dtype": "int32"},
        ],
        "opset": args.opset,
        "torch_onnx_mean_cosine": float(cos_sum / n),
    }
    with open(os.path.join(args.output_dir, "model_meta.json"), "w") as f:
        json.dump(meta, f, indent=2)
    print("DONE:", onnx_path)


if __name__ == "__main__":
    main()
