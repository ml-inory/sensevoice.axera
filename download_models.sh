#!/bin/bash
# 下载三芯预编译模型到 models/（AX650N / AX630C / AX620Q）
# 模型托管在 HF: https://huggingface.co/AXERA-TECH/SenseVoice
set -e
HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
mkdir -p models
for chip in sensevoice_ax650 sensevoice_ax630c sensevoice_ax620q; do
  if [ ! -d "models/$chip" ]; then
    echo "Downloading $chip ..."
    HF_ENDPOINT="$HF_ENDPOINT" python3 -c "
from huggingface_hub import snapshot_download
import sys
snapshot_download(repo_id='AXERA-TECH/SenseVoice', allow_patterns=['$chip/**'], local_dir='models')
"
  else
    echo "$chip already exists, skip"
  fi
done
echo "Done. models/:"
ls models
