import onnxruntime as ort
import numpy as np
from utils.frontend import WavFrontend
import librosa
from typing import List, Union, Dict, Tuple, Callable


class SenseVoiceOnnx:
    def __init__(self, onnx_filename: str, cmvn_file: str, token_file: str):
        session_opts = ort.SessionOptions()
        session_opts.inter_op_num_threads = 1
        session_opts.intra_op_num_threads = 4

        self.session_opts = session_opts

        self.session = ort.InferenceSession(
            onnx_filename, sess_options=session_opts, providers=["CPUExecutionProvider"]
        )

        meta = self.session.get_modelmeta().custom_metadata_map
        self.max_seq_len = int(meta["max_seq_len"])

        self.lid_dict = {
            "auto": 0,
            "zh": 3,
            "en": 4,
            "yue": 7,
            "ja": 11,
            "ko": 12,
            "nospeech": 13,
        }

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

        self.tokens = []
        with open(token_file, "r") as f:
            for line in f:
                self.tokens.append(line[:-1])

        self.padding = 16
        self.blank_id = 0

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

    def forward(self, speech, mask, langauge_token):
        feed = {"speech": speech, "mask": mask, "language": langauge_token}

        ctc_logits, encoder_out_lens = self.session.run(None, feed)
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

    def sequence_mask(self, max_seq_len, actual_seq_len):
        mask = np.zeros((1, 1, max_seq_len), dtype=np.int32)
        mask[:, :, :actual_seq_len] = 1
        return mask

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

    def infer(
        self,
        audio_path: str,
        language: str,
        callback: Callable[[int, np.ndarray, np.ndarray, np.ndarray], None] = None,
    ):
        speech, speech_len = self.preprocess(audio_path)

        language_token = self.lid_dict[language]
        language_token = np.array([language_token], dtype=np.int32)

        query_num = 4
        slice_len = self.max_seq_len - query_num
        slice_num = int(np.ceil(speech.shape[1] / slice_len))

        tokens = []
        for i in range(slice_num):
            if i == 0:
                sub_feat = speech[:, i * slice_len : (i + 1) * slice_len, :]
            else:
                sub_feat = speech[
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

            mask = self.sequence_mask(self.max_seq_len + 4, real_len)

            if callback:
                callback(i, sub_feat, mask, language_token)

            ctc_logits, encoder_out_lens = self.forward(sub_feat, mask, language_token)

            tokens.extend(self.postprocess(ctc_logits, encoder_out_lens))

        text = "".join([self.tokens[i] for i in tokens])
        return text
