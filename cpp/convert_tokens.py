try:
    import sentencepiece as spm
except ImportError:
    print("please run `pip install sentencepiece`")

import os

sp = spm.SentencePieceProcessor()
sp.load("../models/SenseVoice/chn_jpn_yue_eng_ko_spectok.bpe.model")

tokens = [sp.id_to_piece(i).replace("▁", " ") for i in range(sp.vocab_size())]

# self.gguf_writer.add_string("tokenizer.unk_symbol", "<unk>")

print(f"token num: {len(tokens)}")
print(f"tokens[:20]: {tokens[:20]}")

with open("tokens.txt", 'w') as f:
    for s in tokens:
        f.write(s + '\n')

print('Save to tokens.txt')