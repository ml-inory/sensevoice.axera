# sensevoice.axera
FunASR SenseVoice on Axera, official repo: https://github.com/FunAudioLLM/SenseVoice

## TODO

- [ ] 支持AX630C
- [ ] 支持C++
- [ ] 支持FastAPI

## 功能
 - 语音识别
 - 自动识别语言(支持中文、英文、粤语、日语、韩语)
 - 情感识别
 - 自动标点
 
## 支持平台

- [x] AX650N
- [x] AX630C

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
# 首次运行会自动从huggingface上下载模型, 保存到models中
python3 main.py -i 输入音频文件
```
运行参数说明:  
| 参数名称 | 说明 | 默认值 |
| --- | --- | --- |
| --input/-i | 输入音频文件 | |
| --language/-l | 识别语言，支持auto, zh, en, yue, ja, ko | auto |
| --streaming | 流式识别 | |


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