# sensevoice.axera
FunASR SenseVoice on Axera, official repo: https://github.com/FunAudioLLM/SenseVoice

## TODO

- [x] 支持AX630C
- [x] 支持C++
- [x] 支持FastAPI
- [x] 支持AX620Q（AX620E NPU1）

## 功能
 - 语音识别
 - 自动识别语言(支持中文、英文、粤语、日语、韩语)
 - 情感识别
 - 自动标点
 
## 支持平台

- [x] AX650N
- [x] AX630C
- [x] AX620Q（AXera Pi Zero / AX620E NPU1）

## 环境安装

推荐在板上安装Miniconda管理虚拟环境，安装方法如下:
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-aarch64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

source ~/miniconda3/bin/activate

conda init --all
```

```
sudo apt-get install libsndfile-dev

conda create -n sensevoice python=3.12
conda activate sensevoice
pip install -r requirements.txt
```
如果空间不足可以使用 --prefix 指定别的安装路径

####  安装pyaxenigne

参考 https://github.com/AXERA-TECH/pyaxengine 安装 NPU Python API

在0.1.3rc2上测试通过，可通过
```
pip install https://github.com/AXERA-TECH/pyaxengine/releases/download/0.1.3.rc2/axengine-0.1.3-py3-none-any.whl
```
安装，或把版本号更改为你想使用的版本


## 使用
```
# 首次运行会自动从 huggingface 下载模型, 保存到 models/ 中
# 指定芯片: --chip ax650 (默认) / ax630c / ax620q
python3 main.py -i 输入音频文件 --chip ax650
python3 main.py -i 输入音频文件 --chip ax630c
python3 main.py -i 输入音频文件 --chip ax620q
```
运行参数说明:  
| 参数名称 | 说明 | 默认值 |
| --- | --- | --- |
| --input/-i | 输入音频文件 | |
| --language/-l | 识别语言，支持auto, zh, en, yue, ja, ko | auto |
| --chip/-c | 目标芯片：ax650 / ax630c / ax620q | ax650 |
| --streaming | 流式识别 | |


也可以直接下载三芯预编译模型到 models/：
```
./download_models.sh
```

### 示例:  
example下有测试音频  

如 中文测试
```
python main.py -i example/zh.mp3
```
输出
```
RTF: 0.04386647134764582    Latency: 0.2463541030883789s  Total length: 5.616s
ASR result: 开饭时间早上九点至下午五点

```

## C++ (AX650N / AX630C / AX620Q)

预编译产物在各芯片目录下，模型目录需为 `models/sensevoice/` 结构：

```bash
# AX650N
./cpp/ax650/test_sensevoice -a example/zh.mp3 -p models/sensevoice_ax650
# AX630C
./cpp/ax630c/test_sensevoice -a example/zh.mp3 -p models/sensevoice_ax630c
# AX620Q (AXera Pi Zero, uclibc 静态库 + 可执行文件)
./cpp/ax620q/test_sensevoice -a example/zh.mp3 -p models/sensevoice_ax620q
```

> 说明：`-p` 指向的目录内需为 `sensevoice/sensevoice.axmodel` 结构（`download_models.sh` 下载的
> `models/sensevoice_ax*` 目录已含嵌套 `sensevoice/` 子目录）；在板子上运行时先
> `export LD_LIBRARY_PATH=/opt/lib:$LD_LIBRARY_PATH`。

### C++ 源码（git submodule）

C++ 源码以 git submodule 方式内置，指向官方 [AXERA-TECH/ax_asr_api](https://github.com/AXERA-TECH/ax_asr_api)
（含已合入上游的 SenseVoice LFR 前端对齐修复与 AX620Q kaldi 库目录修复，见 [PR #10](https://github.com/AXERA-TECH/ax_asr_api/pull/10)）：

```bash
# 克隆仓库时带上 submodule
git clone --recursive https://github.com/ml-inory/sensevoice.axera.git
# 已有 clone 时初始化
git submodule update --init cpp/ax_asr_api
```

从源码重新构建三芯：

```bash
cd cpp/ax_asr_api
./build_ax650.sh      # AX650N
./build_ax630c.sh     # AX630C
./build_ax620q.sh     # AX620Q (需要 arm-AX620E-linux-uclibcgnueabihf 工具链 + ax620e_bsp_sdk)
```

预编译产物（`cpp/ax650`、`cpp/ax630c`、`cpp/ax620q`）可直接使用；需要修改或重新编译时用 submodule 源码。

### ASR 服务（asr_server）源码与编译

`asr_server`（FastAPI 风格 ASR HTTP 服务）预编译产物在三芯目录下：

```bash
./cpp/ax650/asr_server   # AX650N
./cpp/ax630c/asr_server  # AX630C
./cpp/ax620q/asr_server  # AX620Q（arm uclibc）
```

源码在 submodule 的 `cpp/ax_asr_api/cpp/server/`（`asr_server.cpp` / `asr_server.hpp` / `main.cpp` / `httplib.h`）。
三芯的构建脚本（`build_ax650.sh` / `build_ax630c.sh` / `build_ax620q.sh`）默认同时编译 `main` 和 `asr_server`：

```bash
cd cpp/ax_asr_api
./build_ax650.sh    # 产物: install/ax650/main + install/ax650/asr_server
./build_ax630c.sh   # 产物: install/ax630c/main + install/ax630c/asr_server
./build_ax620q.sh   # 产物: install/ax620q/main + install/ax620q/asr_server
```

### 流式识别（C++）

预编译产物已支持**专用流式模型**（`streaming_sensevoice.axmodel`，26 帧滚动窗口、每 10 帧一推，与 Python 流式对齐）。
模型目录需同时包含 `sensevoice.axmodel` 与 `streaming_sensevoice.axmodel`。

```bash
# 命令行演示（-s 流式模式，100ms 步长喂音频并打印增量结果）
./cpp/ax650/test_sensevoice -a example/zh.mp3 -t sensevoice -p models/sensevoice_ax650 -s

# 库 API
AX_ASR_StreamInit(handle);                       // 初始化流式状态
AX_ASR_StreamFeed(handle, pcm, n, 16000);        // 喂音频（float [-1,1]）
AX_ASR_StreamResult(handle, &text);              // 获取当前增量结果
AX_ASR_StreamFinish(handle);                     // 喂完后收尾刷新
```

源码改动见 `cpp/ax_asr_api_streaming.patch`（应用到 submodule 后可从源码复现构建）。

## 准确率

使用WER(Word-Error-Rate)作为评价标准  

**WER = 2.0%**  

### 复现测试结果

```
./download_datasets.sh
python test_wer.py -d aishell -g datasets/ground_truth.txt --language zh
```

## 模型转换

参考[model_convert](model_convert/README.md)

## 技术讨论

- Github issues
- QQ 群: 139953715
