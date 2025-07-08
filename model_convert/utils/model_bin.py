#!/usr/bin/env python3
# -*- encoding: utf-8 -*-
# Copyright FunASR (https://github.com/FunAudioLLM/SenseVoice). All Rights Reserved.
#  MIT License  (https://opensource.org/licenses/MIT)

import os.path
from pathlib import Path
from typing import List, Union, Tuple
import torch
import librosa
import numpy as np

from utils.infer_utils import (
    CharTokenizer,
    Hypothesis,
    ONNXRuntimeError,
    OrtInferSession,
    TokenIDConverter,
    get_logger,
    read_yaml,
)
from utils.frontend import WavFrontend
from utils.infer_utils import pad_list

logging = get_logger()


def sequence_mask(lengths, maxlen=None, dtype=torch.float32, device=None):
    if maxlen is None:
        maxlen = lengths.max()
    row_vector = torch.arange(0, maxlen, 1).to(lengths.device)
    matrix = torch.unsqueeze(lengths, dim=-1)
    mask = row_vector < matrix
    mask = mask.detach()

    return mask.type(dtype).to(device) if device is not None else mask.type(dtype)


class SenseVoiceSmallONNX:
    """
    Author: Speech Lab of DAMO Academy, Alibaba Group
    Paraformer: Fast and Accurate Parallel Transformer for Non-autoregressive End-to-End Speech Recognition
    https://arxiv.org/abs/2206.08317
    """

    def __init__(
        self,
        model_dir: Union[str, Path] = None,
        batch_size: int = 1,
        device_id: Union[str, int] = "-1",
        plot_timestamp_to: str = "",
        quantize: bool = False,
        intra_op_num_threads: int = 4,
        cache_dir: str = None,
        seq_len: int = 68,
        **kwargs,
    ):
        if quantize:
            model_file = os.path.join(model_dir, "model_quant.onnx")
        else:
            model_file = os.path.join(model_dir, "model.onnx")

        config_file = os.path.join(model_dir, "config.yaml")
        cmvn_file = os.path.join(model_dir, "am.mvn")
        config = read_yaml(config_file)
        self.model_dir = model_dir
        # token_list = os.path.join(model_dir, "tokens.json")
        # with open(token_list, "r", encoding="utf-8") as f:
        #     token_list = json.load(f)

        # self.converter = TokenIDConverter(token_list)
        self.tokenizer = CharTokenizer()
        config["frontend_conf"]['cmvn_file'] = cmvn_file
        self.frontend = WavFrontend(**config["frontend_conf"])
        self.ort_infer = OrtInferSession(
            model_file, device_id, intra_op_num_threads=intra_op_num_threads
        )
        self.batch_size = batch_size
        self.blank_id = 0
        self.seq_len = seq_len

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}

    def __call__(self, 
                 wav_content: Union[str, np.ndarray, List[str]], 
                #  input_query: np.ndarray,
                 language: str,
                 withitn: bool,
                 position_encoding: np.ndarray,
                 tokenizer=None,
                 **kwargs) -> List:
        waveform_list = self.load_data(wav_content, self.frontend.opts.frame_opts.samp_freq)
        waveform_nums = len(waveform_list)
        asr_res = []

        if isinstance(wav_content, str):
            wav_name = os.path.splitext(os.path.basename(wav_content))[0]

        language_query = np.load(os.path.join(self.model_dir, f"{language}.npy"))
        textnorm_query = np.load(os.path.join(self.model_dir, "withitn.npy") if withitn 
                                 else os.path.join(self.model_dir, "woitn.npy"))
        event_emo_query = np.load(os.path.join(self.model_dir, "event_emo.npy"))

        # textnorm language event_emo speech
        input_query = np.concatenate((textnorm_query, language_query, event_emo_query), axis=1)

        dataset = "dataset"
        os.makedirs(dataset, exist_ok=True)
        speech_dir = os.path.join(dataset, "speech", language, "withitn" if withitn else "woitn")
        mask_dir = os.path.join(dataset, "masks", language, "withitn" if withitn else "woitn")
        pe_dir = os.path.join(dataset, "position_encoding", language, "withitn" if withitn else "woitn")
        os.makedirs(speech_dir, exist_ok=True)
        os.makedirs(mask_dir, exist_ok=True)
        os.makedirs(pe_dir, exist_ok=True)

        # tf_speech = tarfile.open(f"{speech_dir}/speech.tar.gz", "a:gz")
        # tf_masks = tarfile.open(f"{dataset}/masks.tar.gz", "w:gz")
        # tf_pe = tarfile.open(f"{dataset}/position_encoding.tar.gz", "w:gz")

        slice_len = self.seq_len - 4
        for beg_idx in range(0, waveform_nums, self.batch_size):
            end_idx = min(waveform_nums, beg_idx + self.batch_size)
            feats, feats_len = self.extract_feat(waveform_list[beg_idx:end_idx])

            for i in range(int(np.ceil(feats.shape[1] / slice_len))):
                sub_feats = np.concatenate([input_query, feats[:, i*slice_len : (i+1)*slice_len, :]], axis=1)
                feats_len[0] = sub_feats.shape[1]

                if feats_len[0] < self.seq_len:
                    sub_feats = np.concatenate([sub_feats, np.zeros((1, self.seq_len - feats_len[0], 560), dtype=np.float32)], axis=1)

                masks = sequence_mask(torch.IntTensor([self.seq_len]), maxlen=self.seq_len, dtype=torch.float32)[:, None, :]

                masks = masks.numpy()  
                ctc_logits, encoder_out_lens = self.infer(sub_feats, 
                                                        masks,
                                                        position_encoding
                                    )
                
                # save dataset
                np.save(f"{speech_dir}/{wav_name}_{i}.npy", sub_feats)
                np.save(f"{mask_dir}/{wav_name}_{i}.npy", masks)
                np.save(f"{pe_dir}/{wav_name}_{i}.npy", position_encoding)
  
                # tf_speech.add(f"{dataset}/speech/{i}.npy")
                # tf_masks.add(f"{dataset}/masks/{i}.npy")
                # tf_pe.add(f"{dataset}/position_encoding/{i}.npy")

                # back to torch.Tensor
                ctc_logits = torch.from_numpy(ctc_logits).float()
                # support batch_size=1 only currently
                x = ctc_logits[0, : encoder_out_lens[0].item(), :]
                yseq = x.argmax(dim=-1)
                yseq = torch.unique_consecutive(yseq, dim=-1)

                mask = yseq != self.blank_id
                token_int = yseq[mask].tolist()
                if tokenizer is not None:
                    asr_res.append(tokenizer.tokens2text(token_int))
                else:
                    asr_res.append(token_int)
        
        # tf_speech.close()
        # tf_masks.close()
        # tf_pe.close()

        return asr_res

    def load_data(self, wav_content: Union[str, np.ndarray, List[str]], fs: int = None) -> List:
        def load_wav(path: str) -> np.ndarray:
            waveform, _ = librosa.load(path, sr=fs)
            return waveform

        if isinstance(wav_content, np.ndarray):
            return [wav_content]

        if isinstance(wav_content, str):
            return [load_wav(wav_content)]

        if isinstance(wav_content, list):
            return [load_wav(path) for path in wav_content]

        raise TypeError(f"The type of {wav_content} is not in [str, np.ndarray, list]")

    def extract_feat(self, waveform_list: List[np.ndarray]) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)

            feat, feat_len = self.frontend.lfr_cmvn(speech)

            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def infer(self, 
              feats: np.ndarray, 
              masks: np.ndarray,
              position_encoding: np.ndarray,
              ) -> Tuple[np.ndarray, np.ndarray]:
        outputs = self.ort_infer([feats, masks, position_encoding])
        return outputs
