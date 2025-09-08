from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


class SenseVoicePth:
    def __init__(self, model_path, max_len=68, language="auto", use_itn=True, tokenizer=None):
        self.model = AutoModel(
            model=model_path,
            trust_remote_code=True,
            remote_code="./model.py",    
            device="cuda:0",
            ban_emo_unk=True,
        )
        self.language = language
        self.use_itn = use_itn

    def infer(self, filepath_or_data: str, language="auto", print_rtf=True):
        res = self.model.generate(
            input=filepath_or_data,
            cache={},
            language=language,  # "zh", "en", "yue", "ja", "ko", "nospeech"
            use_itn=self.use_itn,
            batch_size_s=60,
            disable_pbar=True
        )
        text = rich_transcription_postprocess(res[0]["text"])

        return text