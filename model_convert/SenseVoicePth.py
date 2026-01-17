import torch
from model import SenseVoiceSmall, SinusoidalPositionEncoder
from utils.frontend import WavFrontend
import librosa
from typing import List, Union, Dict, Tuple
import numpy as np


class SenseVoicePth(torch.nn.Module):
    def __init__(
        self,
        orig_model: SenseVoiceSmall,
        max_seq_len: int = 256,
        cmvn_file: str = "output_dir/am.mvn",
        token_file: str = "output_dir/tokens.txt",
    ):
        super().__init__()

        self.orig_model = orig_model
        self.max_seq_len = max_seq_len

        self.frontend = WavFrontend(
            cmvn_file=cmvn_file,
            fs=16000,
            window="hamming",
            n_mels=80,
            frame_length=25,
            frame_shift=10,
            lfr_m=7,
            lfr_n=6,
        )

        self.embed = SinusoidalPositionEncoder()
        position_encoding = self.embed.get_position_encoding(
            torch.zeros(1, max_seq_len + 4, 560)
        )

        wo_itn = self.orig_model.textnorm_dict["woitn"]
        textnorm_query = self.orig_model.embed(torch.LongTensor([wo_itn])).unsqueeze(1)

        event_emo_query = self.orig_model.embed(torch.LongTensor([[1, 2]])).repeat(
            1, 1, 1
        )

        self.register_buffer("position_encoding", position_encoding.detach())
        self.register_buffer("textnorm_query", textnorm_query.detach())
        self.register_buffer("event_emo_query", event_emo_query.detach())

        self.tokens = []
        with open(token_file, "r") as f:
            for line in f:
                self.tokens.append(line[:-1])

        self.blank_id = 0
        self.padding = 16

    def get_language_token(self, language: str):
        return self.orig_model.lid_dict[language]

    def load_data(self, filepath: str, sr=16000) -> np.ndarray:
        waveform, _ = librosa.load(filepath, sr=sr)
        return [waveform.flatten()]

    def extract_feat(
        self, waveform_list: List[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        feats, feats_len = [], []
        for waveform in waveform_list:
            speech, _ = self.frontend.fbank(waveform)

            feat, feat_len = self.frontend.lfr_cmvn(speech)

            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int64)
        return feats, feats_len

    def load_embeddings(self, emb_path, language, use_itn):
        embeddings = np.load(emb_path, allow_pickle=True).item()
        language_query = embeddings[language]
        textnorm_query = embeddings["withitn"] if use_itn else embeddings["woitn"]
        event_emo_query = embeddings["event_emo"]
        return textnorm_query, language_query, event_emo_query

    @torch.no_grad()
    def forward(
        self, speech: torch.Tensor, mask: torch.IntTensor, language: torch.IntTensor
    ):

        language_query = self.orig_model.embed(language.long()).unsqueeze(1)

        input_query = torch.concat(
            (self.textnorm_query, language_query, self.event_emo_query), dim=1
        )

        speech = torch.cat((input_query, speech), dim=1)
        # speech_len = speech_len.long()
        # speech_len += 4

        # masks = torch.zeros(1, 1, self.max_seq_len + 4, dtype=torch.int32)
        # masks[:, :, :speech_len[0]] = 1

        encoder_out, encoder_out_lens = self.orig_model.encoder.forward_export(
            speech, mask, self.position_encoding
        )
        if isinstance(encoder_out, tuple):
            encoder_out = encoder_out[0]

        ctc_logits = self.orig_model.ctc.ctc_lo(encoder_out)

        return ctc_logits, encoder_out_lens

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def unique_consecutive(self, arr):
        """
        找出数组中连续的唯一值，模拟 torch.unique_consecutive(yseq, dim=-1)

        参数:
        arr: 一维numpy数组

        返回:
        unique_values: 去除连续重复值后的数组
        """
        if len(arr) == 0:
            return np.array([])

        if len(arr) == 1:
            return arr.copy()

        # 找出变化的位置
        diff = np.diff(arr)
        change_positions = np.where(diff != 0)[0] + 1

        # 添加起始位置
        start_positions = np.concatenate(([0], change_positions))

        # 获取唯一值（每个连续段的第一个值）
        unique_values = arr[start_positions]

        return unique_values

    def postprocess(self, ctc_logits, encoder_out_lens):
        # 提取数据
        x = ctc_logits[0, 4 : encoder_out_lens[0], :]

        # 获取最大值索引
        yseq = np.argmax(x, axis=-1)

        # 去除连续重复元素
        yseq = self.unique_consecutive(yseq)

        # 创建掩码并过滤 blank_id
        mask = yseq != self.blank_id
        token_int = yseq[mask].tolist()

        return token_int

    def preprocess(self, audio_path: str):
        speech, speech_len = self.extract_feat(self.load_data(audio_path))
        return speech, speech_len

    def infer(self, audio_path: str, language: str):
        speech, speech_len = self.preprocess(audio_path)
        speech = torch.from_numpy(speech)
        speech_len = torch.from_numpy(speech_len)

        speech_len = speech_len.long()
        speech_len += 4

        self.max_seq_len = speech_len.item()

        mask = torch.zeros(1, 1, self.max_seq_len, dtype=torch.int32)
        mask[:, :, : speech_len[0]] = 1

        self.position_encoding = self.embed.get_position_encoding(
            torch.randn(1, speech_len.item(), 560)
        )

        language_token = torch.LongTensor([self.orig_model.lid_dict[language]])

        ctc_logits, encoder_out_lens = self.forward(speech, mask, language_token)

        ctc_logits = ctc_logits.numpy()
        encoder_out_lens = encoder_out_lens.numpy()
        tokens = self.postprocess(ctc_logits, encoder_out_lens)

        text = "".join([self.tokens[i] for i in tokens])
        return text

    def infer_origin(self, audio_path: str, language: str, kwargs):
        speech, speech_len = self.preprocess(audio_path)
        speech = torch.from_numpy(speech)
        speech_len = torch.from_numpy(speech_len)

        kwargs["data_type"] = "fbank"
        res = self.orig_model.inference(
            data_in=speech,
            data_lengths=speech_len,
            language=language,  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=False,
            ban_emo_unk=False,
            **kwargs
        )

        text = res[0][0]["text"]
        return text
