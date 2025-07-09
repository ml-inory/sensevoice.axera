#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os
import torch
from model import SenseVoiceSmall, SinusoidalPositionEncoder
from utils import export_utils
from utils.model_bin import SenseVoiceSmallONNX
from funasr.utils.postprocess_utils import rich_transcription_postprocess
import torchaudio
import numpy as np
import shutil
import tarfile

torchaudio.set_audio_backend("sox_io")

quantize = False
force_export = True
seq_len = 68

model_dir = "iic/SenseVoiceSmall"
model, kwargs = SenseVoiceSmall.from_pretrained(model=model_dir, device="cpu")
model.seq_len = seq_len

embed = SinusoidalPositionEncoder()
position_encoding = embed.get_position_encoding(torch.randn(1, seq_len, 560)).numpy()

rebuilt_model = model.export(type="onnx", quantize=False)
model_path = "./output_dir"
kwargs["output_dir"] = model_path
os.makedirs(model_path, exist_ok=True)
origin_model_path = os.path.dirname(kwargs.get("init_param"))
shutil.copy(os.path.join(origin_model_path, "config.yaml"), model_path)
shutil.copy(os.path.join(origin_model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"), model_path)
shutil.copy(os.path.join(origin_model_path, "am.mvn"), model_path)

model_file = os.path.join(model_path, "model.onnx")
if quantize:
    model_file = os.path.join(model_path, "model_quant.onnx")

# export model
if not os.path.exists(model_file) or force_export:
    with torch.no_grad():
        del kwargs['model']
        export_dir = export_utils.export(model=rebuilt_model, **kwargs)
        print("Export model onnx to {}".format(model_file))
        
# export model init
model_bin = SenseVoiceSmallONNX(model_path, seq_len=seq_len)

# build tokenizer
try:
    from funasr.tokenizer.sentencepiece_tokenizer import SentencepiecesTokenizer
    tokenizer = SentencepiecesTokenizer(bpemodel=os.path.join(model_path, "chn_jpn_yue_eng_ko_spectok.bpe.model"))
except:
    tokenizer = None

# inference
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
    "auto": en_list + ja_list + ko_list + yue_list + zh_list
}

with torch.no_grad():
    for language, value in model.lid_dict.items():
        language_query = model.embed(torch.LongTensor([value])).unsqueeze(1)
        language_query = language_query.numpy()
        np.save(f"{model_path}/{language}.npy", language_query)

    for textnorm, value in model.textnorm_dict.items():
        textnorm_query = model.embed(torch.LongTensor([value])).unsqueeze(1)
        textnorm_query = textnorm_query.numpy()
        np.save(f"{model_path}/{textnorm}.npy", textnorm_query)

    event_emo_query = model.embed(torch.LongTensor([[1, 2]]).to(model.device)).repeat(
        1, 1, 1
    )
    np.save(f"{model_path}/event_emo.npy", event_emo_query.numpy())
    np.save(f"{model_path}/position_encoding.npy", position_encoding)


    dataset = "dataset"
    os.makedirs(dataset, exist_ok=True)
    speech_dir = os.path.join(dataset, "speech")
    mask_dir = os.path.join(dataset, "masks")
    pe_dir = os.path.join(dataset, "position_encoding")

    tf_speech = tarfile.open(f"{dataset}/speech.tar.gz", "w:gz")
    tf_masks = tarfile.open(f"{dataset}/masks.tar.gz", "w:gz")
    tf_pe = tarfile.open(f"{dataset}/position_encoding.tar.gz", "w:gz")

    for withitn in [True, False]:
        for language in data_dir.keys():
            for wav_file in data_dir[language]:
                res = model_bin(wav_file, language, withitn, position_encoding, tokenizer=tokenizer)
                print(wav_file)
                print([rich_transcription_postprocess(i) for i in res])
                print("\n")

    tf_speech.add(speech_dir)
    tf_masks.add(mask_dir)
    tf_pe.add(pe_dir)

    tf_speech.close()
    tf_masks.close()
    tf_pe.close()
