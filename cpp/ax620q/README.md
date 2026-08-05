# AX620Q (AX620E, NPU1) C++ SDK

预编译产物（ARM 32-bit uclibc，AXera Pi Zero / AX620Q）：

- `lib/libax_asr_api.a` + `include/ax_asr_api.h`：静态库与头文件
- `test_sensevoice`：端到端示例（音频 → NPU 推理 → 文本）
- `asr_server`：FastAPI 风格 ASR 服务

模型目录结构（`-p` 指向的目录，来自根目录 `download_models.sh` 的 `models/sensevoice_ax620q/`）：

```
models/sensevoice/
├── sensevoice.axmodel   # AX620E NPU1 INT8 (292MB)
├── am.mvn
├── tokens.txt
└── chn_jpn_yue_eng_ko_spectok.bpe.model
```

运行：

```bash
export LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH
./test_sensevoice -a example/zh.mp3 -t sensevoice -p models/sensevoice_ax620q -l zh
```

源码与构建脚本见 [ax_asr_api](https://github.com/AXERA-TECH/ax_asr_api)（`build_ax620q.sh`）。
