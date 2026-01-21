import numpy as np
from fastapi import FastAPI, HTTPException, Body
from fastapi.responses import JSONResponse
from typing import List, Optional
import logging
import json
from SenseVoiceAx import SenseVoiceAx
from download_utils import download_model
import os
import librosa

# 初始化日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="ASR Server", description="Automatic Speech Recognition API")

# 全局变量存储模型
asr_model = None


@app.on_event("startup")
async def load_model():
    """
    服务启动时加载ASR模型
    """
    global asr_model
    logger.info("Loading ASR model...")

    try:
        # 模型加载
        language = "auto"
        use_itn = True  # 标点符号预测
        max_len = 68

        model_root = download_model("SenseVoice")
        model_root = os.path.join(model_root, "sensevoice_ax650")
        max_seq_len = 256
        model_path = os.path.join(model_root, "sensevoice.axmodel")

        assert os.path.exists(model_path), f"model {model_path} not exist"

        cmvn_file = os.path.join(model_root, "am.mvn")
        bpe_model = os.path.join(model_root, "chn_jpn_yue_eng_ko_spectok.bpe.model")
        token_file = os.path.join(model_root, "tokens.txt")

        asr_model = SenseVoiceAx(
            model_path,
            cmvn_file,
            token_file,
            bpe_model,
            max_seq_len=max_seq_len,
            beam_size=3,
            hot_words=None,
            streaming=False,
        )

        print(f"language: {language}")
        print(f"use_itn: {use_itn}")
        print(f"model_path: {model_path}")

        logger.info("ASR model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load ASR model: {str(e)}")
        raise


def validate_audio_data(audio_data: List[float]) -> np.ndarray:
    """
    验证并转换音频数据为numpy数组

    参数:
    - audio_data: 浮点数列表表示的音频数据

    返回:
    - 验证后的numpy数组
    """
    try:
        # 转换为numpy数组
        np_array = np.array(audio_data, dtype=np.float32)

        # 验证数据有效性
        if np_array.ndim != 1:
            raise ValueError("Audio data must be 1-dimensional")

        if len(np_array) == 0:
            raise ValueError("Audio data cannot be empty")

        return np_array
    except Exception as e:
        raise ValueError(f"Invalid audio data: {str(e)}")


@app.get("/get_language", summary="Get current language")
async def get_language():
    return JSONResponse(content={"language": asr_model.language})


@app.get(
    "/get_language_options",
    summary="Get possible language options, possible options include [auto, zh, en, yue, ja, ko]",
)
async def get_language_options():
    return JSONResponse(content={"language_options": asr_model.language_options})


@app.post("/asr", summary="Recognize speech from numpy audio data")
async def recognize_speech(
    audio_data: List[float] = Body(
        ..., embed=True, description="Audio data as list of floats"
    ),
    sample_rate: Optional[int] = Body(16000, description="Audio sample rate in Hz"),
    language: Optional[str] = Body("auto", description="Language"),
):
    """
    接收numpy数组格式的音频数据并返回识别结果

    参数:
    - audio_data: 浮点数列表表示的音频数据
    - sample_rate: 音频采样率(默认16000Hz)

    返回:
    - JSON包含识别文本
    """
    try:
        # 检查模型是否已加载
        if asr_model is None:
            raise HTTPException(status_code=503, detail="ASR model not loaded")

        logger.info(f"Received audio data with length: {len(audio_data)}")

        # 验证并转换数据
        np_audio = validate_audio_data(audio_data)
        # 调用模型进行识别
        result = asr_model.infer_waveform((np_audio, sample_rate), language)

        return JSONResponse(content={"text": result})

    except ValueError as e:
        logger.error(f"Validation error: {str(e)}")
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Recognition error: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
