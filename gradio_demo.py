import gradio as gr
import os
from SenseVoiceAx import SenseVoiceAx
from tokenizer import SentencepiecesTokenizer
from print_utils import rich_transcription_postprocess
from download_utils import download_model

use_itn = True # 标点符号预测
max_len = 68

model_path_root = download_model("SenseVoice")
model_path = os.path.join(model_path_root, "sensevoice_ax650", "sensevoice.axmodel")
bpemodel = os.path.join(model_path_root, "chn_jpn_yue_eng_ko_spectok.bpe.model")

assert os.path.exists(model_path), f"model {model_path} not exist"

tokenizer = SentencepiecesTokenizer(bpemodel=bpemodel)
pipeline = SenseVoiceAx(model_path, 
                        max_len=max_len,
                        language="auto", 
                        use_itn=use_itn, 
                        tokenizer=tokenizer)
# 你实现的语言转文本函数
def speech_to_text(audio_path, lang):
    """
    audio_path: 音频文件路径
    lang: 语言类型 "auto", "zh", "en", "yue", "ja", "ko"
    """
    if not audio_path:
        return "无音频"
    
    pipeline.choose_language(language=lang)
    asr_res = pipeline.infer(audio_path, print_rtf=True)
    res = " ".join([rich_transcription_postprocess(i) for i in asr_res])
    # TODO: 这里写你的语音识别逻辑
    # 返回一个示例文本
    return res


def main():
    with gr.Blocks() as demo:
        with gr.Row():
            output_text = gr.Textbox(
                label="识别结果",
                lines=5
            )
            

        with gr.Row():
            audio_input = gr.Audio(
                sources=["microphone"],
                type="filepath",
                label="录制或上传音频",
                format="mp3"
            )
            lang_dropdown = gr.Dropdown(
                choices=["auto", "zh", "en", "yue", "ja", "ko"],
                value="auto",
                label="选择音频语言"
            )

        


        audio_input.change(
            fn=speech_to_text,
            inputs=[audio_input, lang_dropdown],
            outputs=output_text
        )

    demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            ssl_certfile="./cert.pem", ssl_keyfile="./key.pem", ssl_verify=False
        )

if __name__ == "__main__":
    main()