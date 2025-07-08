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
wav_or_scp = "../example/yue.mp3"
language_list = [0]
textnorm_list = [14]

with torch.no_grad():
    language = torch.LongTensor(language_list).to(model.device)
    textnorm = torch.LongTensor(textnorm_list).to(model.device)

    language_query = model.embed(language.to(model.device)).unsqueeze(1)
    textnorm_query = model.embed(textnorm.to(model.device)).unsqueeze(1)

    # speech = torch.cat((textnorm_query, speech), dim=1)
    # speech_lengths += 1

    event_emo_query = model.embed(torch.LongTensor([[1, 2]]).to(model.device)).repeat(
        1, 1, 1
    )
    # textnorm language event_emo speech
    input_query = torch.cat((textnorm_query, language_query, event_emo_query), dim=1)
    # speech = torch.cat((input_query, speech), dim=1)
    # speech_lengths += 3

    res = model_bin(wav_or_scp, input_query.numpy(), position_encoding, tokenizer=tokenizer)
    print([rich_transcription_postprocess(i) for i in res])


    for language, value in model.lid_dict.items():
        language_query = model.embed(torch.LongTensor([value])).unsqueeze(1)
        language_query = language_query.numpy()
        np.save(f"output_dir/{language}.npy", language_query)

    for textnorm, value in model.textnorm_dict.items():
        textnorm_query = model.embed(torch.LongTensor([value])).unsqueeze(1)
        textnorm_query = textnorm_query.numpy()
        np.save(f"output_dir/{textnorm}.npy", textnorm_query)

    np.save(f"output_dir/event_emo.npy", event_emo_query.numpy())
    np.save("output_dir/position_encoding.npy", position_encoding)