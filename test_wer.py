import os, sys
import argparse
from SenseVoiceAx import SenseVoiceAx
from tokenizer import SentencepiecesTokenizer
from print_utils import rich_transcription_postprocess, rich_print_asr_res
from download_utils import download_model
import jiwer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", "-d", required=True, type=str, help="Input dataset")
    parser.add_argument("--language", "-l", required=False, type=str, default="auto", choices=["auto", "zh", "en", "yue", "ja", "ko"])
    return parser.parse_args()


def main():
    args = get_args()

    dataset = args.dataset
    language = args.language
    use_itn = False # 标点符号预测

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
    wer_file = open("wer.txt", "w")
    for wav_name, reference in zip(wav_names, references):
        wav_path = os.path.join(dataset, "aishell_S0764", wav_name + ".wav")

        asr_res = pipeline.infer(wav_path, print_rtf=False)
        hypothesis = rich_print_asr_res(asr_res, will_print=False, remove_punc=True)
        hyp.append(hypothesis)

        wer = jiwer.cer(
                    reference,
                    hypothesis
                )
        
        line_content = f"{wav_name}  reference: {reference}  hypothesis: {hypothesis}  WER: {wer}"
        wer_file.write(line_content + "\n")
        print(line_content)

    total_wer = jiwer.cer(
                    references,
                    hyp
                )
    print(f"Total WER: {total_wer}")
    wer_file.write(f"Total WER: {total_wer}")
    wer_file.close()

if __name__ == "__main__":
    main()