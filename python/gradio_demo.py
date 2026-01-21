import gradio as gr
import os
from SenseVoiceAx import SenseVoiceAx
from download_utils import download_model
import numpy as np

model_root = download_model("SenseVoice")
model_root = os.path.join(model_root, "sensevoice_ax650")
max_seq_len = 256
model_path = os.path.join(model_root, "sensevoice.axmodel")

assert os.path.exists(model_path), f"model {model_path} not exist"

cmvn_file = os.path.join(model_root, "am.mvn")
bpe_model = os.path.join(model_root, "chn_jpn_yue_eng_ko_spectok.bpe.model")
token_file = os.path.join(model_root, "tokens.txt")

model = SenseVoiceAx(
    model_path,
    cmvn_file,
    token_file,
    bpe_model,
    max_seq_len=max_seq_len,
    beam_size=3,
    hot_words=None,
    streaming=False,
)

# 你实现的语言转文本函数
def speech_to_text(audio_input, lang):
    """
    audio_input: 音频文件路径
    lang: 语言类型 "auto", "zh", "en", "yue", "ja", "ko"
    """
    if not audio_input:
        return "无音频"

    sr, audio_data = audio_input
    if audio_data.dtype == np.int16:
        audio_data = audio_data.astype(np.float32) / np.iinfo(np.int16).max

    asr_res = model.infer((audio_data, sr), lang, print_rtf=False)
    return asr_res


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            output_text = gr.Textbox(label="识别结果", lines=5)

        with gr.Row():
            audio_input = gr.Audio(
                sources=["microphone"], type="numpy", label="录制或上传音频", format="wav"
            )
            lang_dropdown = gr.Dropdown(
                choices=["auto", "zh", "en", "yue", "ja", "ko"],
                value="auto",
                label="选择音频语言",
            )

        audio_input.change(
            fn=speech_to_text, inputs=[audio_input, lang_dropdown], outputs=output_text
        )

    demo.launch(
        server_name="0.0.0.0",
        ssl_certfile="./cert.pem",
        ssl_keyfile="./key.pem",
        ssl_verify=False,
    )


if __name__ == "__main__":
    main()
