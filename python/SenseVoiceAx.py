import axengine as axe
import numpy as np
import librosa
from frontend import WavFrontend
import time
from typing import List, Union, Optional, Tuple
import torch


def unique_consecutive(arr):
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


class SenseVoiceAx:
    """SenseVoice axmodel runner"""

    def __init__(
        self,
        model_path: str,
        cmvn_file: str,
        token_file: str,
        bpe_model: str = None,
        max_seq_len: int = 256,
        beam_size: int = 3,
        hot_words: Optional[List[str]] = None,
        streaming: bool = False
    ):
        """
        Initialize SenseVoiceAx

        Args:
            model_path: Path of axmodel
            max_len:    Fixed shape of input of axmodel
            beam_size:  Max number of hypos to hold after each decode step
            language:   Support auto, zh(Chinese), en(English), yue(Cantonese), ja(Japanese), ko(Korean)
            hot_words:  Words that may fail to recognize,
                        special words/phrases (aka hotwords) like rare words, personalized information etc.
            use_itn:    Allow Invert Text Normalization if True,
                        ITN converts ASR model output into its written form to improve text readability,
                        For example, the ITN module replaces “one hundred and twenty-three dollars” transcribed by an ASR model with “$123.”
            streaming:  Processes audio in small segments or "chunks" sequentially and outputs text on the fly.
                        Use stream_infer method if streaming is true otherwise infer.

        """

        self.streaming = streaming

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

        self.model = axe.InferenceSession(model_path)
        self.sample_rate = 16000
        self.blank_id = 0
        self.max_seq_len = max_seq_len
        self.padding = 16
        self.input_size = 560
        self.query_num = 4
        self.tokens = self.load_tokens(token_file)

        self.lid_dict = {
            "auto": 0,
            "zh": 3,
            "en": 4,
            "yue": 7,
            "ja": 11,
            "ko": 12,
            "nospeech": 13,
        }

        if streaming:
            from asr_decoder import CTCDecoder
            from online_fbank import OnlineFbank

            # decoder
            if beam_size > 1 and hot_words is not None:
                self.beam_size = beam_size
                symbol_table = {}
                for i in range(len(self.tokens)):
                    symbol_table[self.tokens[i]] = i
                self.decoder = CTCDecoder(hot_words, symbol_table, bpe_model)
            else:
                self.beam_size = 1
                self.decoder = CTCDecoder()

            self.cur_idx = -1
            self.chunk_size = max_seq_len - self.padding
            self.caches_shape = (max_seq_len, self.input_size)
            self.caches = np.zeros(self.caches_shape, dtype=np.float32)
            self.zeros = np.zeros((1, self.input_size), dtype=np.float32)
            self.neg_mean, self.inv_stddev = (
                self.frontend.cmvn[0, :],
                self.frontend.cmvn[1, :],
            )

            self.fbank = OnlineFbank(window_type="hamming")
            self.stream_mask = self.sequence_mask(
                max_seq_len + self.query_num, max_seq_len + self.query_num
            )

    def load_tokens(self, token_file):
        tokens = []
        with open(token_file, "r") as f:
            for line in f:
                tokens.append(line[:-1])
        return tokens

    @property
    def language_options(self):
        return list(self.lid_dict.keys())

    def sequence_mask(self, max_seq_len, actual_seq_len):
        mask = np.zeros((1, 1, max_seq_len), dtype=np.int32)
        mask[:, :, :actual_seq_len] = 1
        return mask

    def load_data(self, filepath: str) -> np.ndarray:
        waveform, _ = librosa.load(filepath, sr=self.sample_rate)
        return waveform.flatten()

    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def preprocess(self, waveform):
        feats, feats_len = [], []
        for wf in [waveform]:
            speech, _ = self.frontend.fbank(wf)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len

    def postprocess(self, ctc_logits, encoder_out_lens):
        # 提取数据
        x = ctc_logits[0, 4 : encoder_out_lens[0], :]

        # 获取最大值索引
        yseq = np.argmax(x, axis=-1)

        # 去除连续重复元素
        yseq = unique_consecutive(yseq)

        # 创建掩码并过滤 blank_id
        mask = yseq != self.blank_id
        token_int = yseq[mask].tolist()

        return token_int

    def infer_waveform(self, waveform: np.ndarray, language="auto"):
        feat, feat_len = self.preprocess(waveform)

        slice_len = self.max_seq_len
        slice_num = int(np.ceil(feat.shape[1] / slice_len))

        language_token = self.lid_dict[language]
        language_token = np.array([language_token], dtype=np.int32)

        asr_res = []
        for i in range(slice_num):
            if i == 0:
                sub_feat = feat[:, i * slice_len : (i + 1) * slice_len, :]
            else:
                sub_feat = feat[
                    :,
                    i * slice_len - self.padding : (i + 1) * slice_len - self.padding,
                    :,
                ]

            real_len = sub_feat.shape[1]
            if real_len < self.max_seq_len:
                sub_feat = np.concatenate(
                    [
                        sub_feat,
                        np.zeros(
                            (1, self.max_seq_len - real_len, sub_feat.shape[-1]),
                            dtype=np.float32,
                        ),
                    ],
                    axis=1,
                )

            mask = self.sequence_mask(self.max_seq_len + self.query_num, real_len)

            # start = time.time()
            outputs = self.model.run(
                None,
                {
                    "speech": sub_feat,
                    "mask": mask,
                    "language": language_token,
                },
            )
            ctc_logits, encoder_out_lens = outputs

            token_int = self.postprocess(ctc_logits, encoder_out_lens)

            asr_res.extend(token_int)

        text = "".join([self.tokens[i] for i in asr_res])
        return text

    def infer(
        self, filepath_or_data: Union[Tuple[np.ndarray, int], str], language="auto", print_rtf=False
    ):
        assert not self.streaming, "This method is for non-streaming model"

        if isinstance(filepath_or_data, str):
            waveform = self.load_data(filepath_or_data)
        else:
            waveform, sr = filepath_or_data
            if sr != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate, res_type="soxr_hq")

        total_time = waveform.shape[-1] / self.sample_rate

        start = time.time()
        asr_res = self.infer_waveform(waveform, language)
        latency = time.time() - start

        if print_rtf:
            rtf = latency / total_time
            print(f"RTF: {rtf}    Latency: {latency}s  Total length: {total_time}s")
        return asr_res

    def decode(self, times, tokens):
        times_ms = []
        for step, token in zip(times, tokens):
            if len(self.tokens[token].strip()) == 0:
                continue
            times_ms.append(step * 60)
        return times_ms, "".join([self.tokens[i] for i in tokens])

    def reset(self):
        from online_fbank import OnlineFbank
        self.cur_idx = -1
        self.decoder.reset()
        self.fbank = OnlineFbank(window_type="hamming")
        self.caches = np.zeros(self.caches_shape)

    def get_size(self):
        effective_size = self.cur_idx + 1 - self.padding
        if effective_size <= 0:
            return 0
        return effective_size % self.chunk_size or self.chunk_size

    def stream_infer(self, audio, is_last, language="auto"):
        assert self.streaming, "This method is for streaming model"

        language_token = self.lid_dict[language]
        language_token = np.array([language_token], dtype=np.int32)

        self.fbank.accept_waveform(audio, is_last)
        features = self.fbank.get_lfr_frames(
            neg_mean=self.neg_mean, inv_stddev=self.inv_stddev
        )

        if is_last and len(features) == 0:
            features = self.zeros

        for idx, feature in enumerate(features):
            is_last = is_last and idx == features.shape[0] - 1
            self.caches = np.roll(self.caches, -1, axis=0)
            self.caches[-1, :] = feature
            self.cur_idx += 1
            cur_size = self.get_size()
            if cur_size != self.chunk_size and not is_last:
                continue

            speech = self.caches[None, ...]
            outputs = self.model.run(
                None,
                {
                    "speech": speech,
                    "mask": self.stream_mask,
                    "language": language_token,
                },
            )
            ctc_logits, encoder_out_lens = outputs
            probs = ctc_logits[0, 4 + self.padding: encoder_out_lens[0]]
            probs = torch.from_numpy(probs)

            if cur_size != self.chunk_size:
                probs = probs[self.chunk_size - cur_size :]
            if not is_last:
                probs = probs[: self.chunk_size]
            if self.beam_size > 1:
                res = self.decoder.ctc_prefix_beam_search(
                    probs, beam_size=self.beam_size, is_last=is_last
                )
                times_ms, text = self.decode(res["times"][0], res["tokens"][0])
            else:
                res = self.decoder.ctc_greedy_search(probs, is_last=is_last)
                times_ms, text = self.decode(res["times"], res["tokens"])
            yield {"timestamps": times_ms, "text": text}
