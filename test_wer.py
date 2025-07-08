import os, sys
import argparse
from SenseVoiceAx import SenseVoiceAx
from tokenizer import SentencepiecesTokenizer
from print_utils import rich_transcription_postprocess, rich_print_asr_res
from download_utils import download_model
import logging
import re


def setup_logging():
    """配置日志系统，同时输出到控制台和文件"""
    # 获取脚本所在目录
    script_dir = os.path.dirname(os.path.abspath(__file__))
    log_file = os.path.join(script_dir, "test_wer.log")
    
    # 配置日志格式
    log_format = '%(asctime)s - %(levelname)s - %(message)s'
    date_format = '%Y-%m-%d %H:%M:%S'
    
    # 创建logger
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    
    # 清除现有的handler
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # 创建文件handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(log_format, date_format)
    file_handler.setFormatter(file_formatter)
    
    # 创建控制台handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(log_format, date_format)
    console_handler.setFormatter(console_formatter)
    
    # 添加handler到logger
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str, help="Input dataset")
    parser.add_argument("--language", "-l", required=False, type=str, default="auto", choices=["auto", "zh", "en", "yue", "ja", "ko"])
    parser.add_argument("--max_num", type=int, default=-1, required=False, help="Maximum test data num")
    return parser.parse_args()


def min_distance(word1: str, word2: str) -> int:
 
    row = len(word1) + 1
    column = len(word2) + 1
 
    cache = [ [0]*column for i in range(row) ]
 
    for i in range(row):
        for j in range(column):
 
            if i ==0 and j ==0:
                cache[i][j] = 0
            elif i == 0 and j!=0:
                cache[i][j] = j
            elif j == 0 and i!=0:
                cache[i][j] = i
            else:
                if word1[i-1] == word2[j-1]:
                    cache[i][j] = cache[i-1][j-1]
                else:
                    replace = cache[i-1][j-1] + 1
                    insert = cache[i][j-1] + 1
                    remove = cache[i-1][j] + 1
 
                    cache[i][j] = min(replace, insert, remove)
 
    return cache[row-1][column-1]


def remove_punctuation(text):
    # 定义正则表达式模式，匹配所有标点符号
    # 这个模式包括常见的标点符号和中文标点
    pattern = r'[^\w\s]|_'
    
    # 使用sub方法将所有匹配的标点符号替换为空字符串
    cleaned_text = re.sub(pattern, '', text)
    
    return cleaned_text


def main():
    logger = setup_logging()
    args = get_args()

    dataset = args.dataset
    language = args.language
    use_itn = False # 标点符号预测
    max_num = args.max_num

    model_path_root = download_model("SenseVoice")
    model_path = os.path.join(model_path_root, "sensevoice_ax650", "sensevoice.axmodel")
    bpemodel = os.path.join(model_path_root, "chn_jpn_yue_eng_ko_spectok.bpe.model")

    assert os.path.exists(model_path), f"model {model_path} not exist"

    print(f"dataset: {dataset}")
    print(f"language: {language}")
    print(f"use_itn: {use_itn}")
    print(f"model_path: {model_path}")

    tokenizer = SentencepiecesTokenizer(bpemodel=bpemodel)
    pipeline = SenseVoiceAx(model_path, language, use_itn, tokenizer=tokenizer)

    # Load dataset
    wav_names = []
    references = []
    with open(os.path.join(dataset, "ground_truth.txt"), "r") as f:
        for line in f:
            line = line.strip()
            w, r = line.split(" ")
            wav_names.append(w)
            references.append(r)

    # Iterate over dataset
    hyp = []
    all_character_error_num = 0
    all_character_num = 0
    wer_file = open("wer.txt", "w")
    max_data_num = max_num if max_num > 0 else len(wav_names)
    for n, (wav_name, reference) in enumerate(zip(wav_names, references)):
        wav_path = os.path.join(dataset, "aishell_S0764", wav_name + ".wav")
        reference = remove_punctuation(reference)

        asr_res = pipeline.infer(wav_path, print_rtf=False)
        hypothesis = rich_print_asr_res(asr_res, will_print=False, remove_punc=True)
        hyp.append(hypothesis)

        character_error_num = min_distance(reference, hypothesis)
        character_num = len(reference)
        character_error_rate = character_error_num / character_num * 100

        all_character_error_num += character_error_num
        all_character_num += character_num

        hyp.append(hypothesis)
        references.append(reference)
        
        line_content = f"({n+1}/{max_data_num}) {os.path.basename(wav_path)}  gt: {reference}  predict: {hypothesis}  WER: {character_error_rate}%"
        wer_file.write(line_content + "\n")
        logger.info(line_content)

    total_character_error_rate = all_character_error_num / all_character_num * 100

    logger.info(f"Total WER: {total_character_error_rate}%")
    wer_file.write(f"Total WER: {total_character_error_rate}%")
    wer_file.close()

if __name__ == "__main__":
    main()