import axengine as axe
import numpy as np
import librosa
from frontend import WavFrontend
import os
import time
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union

def sequence_mask(lengths, maxlen=None, dtype=np.float32):
    # 如果 maxlen 未指定，则取 lengths 中的最大值
    if maxlen is None:
        maxlen = np.max(lengths)
    
    # 创建一个从 0 到 maxlen-1 的行向量
    row_vector = np.arange(0, maxlen, 1)
    
    # 将 lengths 转换为列向量
    matrix = np.expand_dims(lengths, axis=-1)
    
    # 比较生成掩码
    mask = row_vector < matrix
    
    # 返回指定数据类型的掩码
    return mask.astype(dtype)[None, ...]
    
def unique_consecutive_np(x, dim=None, return_inverse=False, return_counts=False):
    if dim is None:
        # 默认情况，展平后去重
        x_flat = x.ravel()
        mask = np.concatenate(([True], x_flat[1:] != x_flat[:-1]))
        unique_data = x_flat[mask]
    else:
        # 沿着指定维度去重
        axis = dim if dim >= 0 else x.ndim + dim
        if axis >= x.ndim:
            raise ValueError(f"dim {dim} is out of range for array of dimension {x.ndim}")
        
        # 使用 np.diff 检查相邻元素是否相同
        mask = np.ones(x.shape[axis], dtype=bool)
        if x.shape[axis] > 1:
            # 比较当前元素和前一个元素是否不同
            diff = np.diff(x, axis=axis)
            mask[1:] = np.any(diff != 0, axis=tuple(range(diff.ndim))[axis:])
        
        # 使用 mask 索引提取唯一元素
        unique_data = np.take(x, np.where(mask)[0], axis=axis)
    
    # 处理 return_inverse 和 return_counts
    results = (unique_data,)
    
    if return_inverse:
        if dim is None:
            inv_idx = np.cumsum(mask) - 1
        else:
            inv_idx = np.cumsum(mask) - 1
            # 需要调整形状以匹配输入
            inv_idx = np.expand_dims(inv_idx, axis=axis)
            inv_idx = np.broadcast_to(inv_idx, x.shape)
        results += (inv_idx,)
    
    if return_counts:
        if dim is None:
            counts = np.diff(np.where(np.concatenate((mask, [True])))[0])
        else:
            counts = np.diff(np.where(np.concatenate((mask, [True])))[0])
        results += (counts,)
    
    return results[0] if len(results) == 1 else results

class SenseVoiceAx:
    def __init__(self, model_path, language="auto", use_itn=True, tokenizer=None):
        model_path_root = os.path.join(os.path.dirname(model_path), "..")
        embedding_root = os.path.join(model_path_root, "embeddings")
        self.frontend = WavFrontend(cmvn_file=f"{model_path_root}/am.mvn",
                                    fs=16000, 
                                    window="hamming", 
                                    n_mels=80, 
                                    frame_length=25, 
                                    frame_shift=10,
                                    lfr_m=7,
                                    lfr_n=6,)
        self.model = axe.InferenceSession(model_path)
        self.sample_rate = 16000
        self.tokenizer = tokenizer
        self.blank_id = 0
        self.max_len = 34

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}

        self.position_encoding = np.load(f"{embedding_root}/position_encoding.npy")
        language_query = np.load(f"{embedding_root}/{language}.npy")
        textnorm_query = np.load(f"{embedding_root}/withitn.npy") if use_itn else np.load(f"{embedding_root}/woitn.npy")
        event_emo_query = np.load(f"{embedding_root}/event_emo.npy")
        self.input_query = np.concatenate((textnorm_query, language_query, event_emo_query), axis=1)
        self.query_num = self.input_query.shape[1]
        self.masks = sequence_mask(np.array([self.max_len], dtype=np.int32), dtype=np.float32)

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
        x = ctc_logits[0, :encoder_out_lens[0], :]

        # 获取最大值索引
        yseq = np.argmax(x, axis=-1)

        # 去除连续重复元素
        yseq = unique_consecutive_np(yseq, dim=-1)

        # 创建掩码并过滤 blank_id
        mask = yseq != self.blank_id
        token_int = yseq[mask].tolist()

        return token_int
    
    def infer_waveform(self, waveform: np.ndarray):
        feat, feat_len = self.preprocess(waveform)

        slice_len = self.max_len - self.query_num
        slice_num = int(np.ceil(feat.shape[1] / slice_len))

        asr_res = []
        for i in range(slice_num):
            sub_feat = feat[:, i*slice_len:(i+1)*slice_len, :]
            # concat query
            sub_feat = np.concatenate([self.input_query, sub_feat], axis=1)

            if sub_feat.shape[1] < self.max_len:
                sub_feat = np.concatenate([
                        sub_feat, 
                        np.zeros((1, self.max_len - sub_feat.shape[1], sub_feat.shape[-1]), dtype=np.float32)
                    ],
                    axis=1)
                
            outputs = self.model.run(None, {"speech": sub_feat,
                                            "masks": self.masks,
                                            "position_encoding": self.position_encoding})
            ctc_logits, encoder_out_lens = outputs

            token_int = self.postprocess(ctc_logits, encoder_out_lens)
            if self.tokenizer is not None:
                asr_res.append(self.tokenizer.tokens2text(token_int))
            else:
                asr_res.append(token_int)

        return asr_res
    
    def infer(self, filepath_or_data: Union[np.ndarray, str], print_rtf=True):
        if isinstance(filepath_or_data, str):
            waveform = self.load_data(filepath_or_data)
        else:
            waveform = filepath_or_data

        total_time = waveform.shape[-1] / self.sample_rate

        start = time.time()
        asr_res = self.infer_waveform(waveform)
        latency = time.time() - start

        if print_rtf:
            rtf = latency / total_time
            print(f"RTF: {rtf}    Latency: {latency}s  Total length: {total_time}s")
        return asr_res

    