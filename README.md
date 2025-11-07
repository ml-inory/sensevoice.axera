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
- [ ] AX630C

## 环境安装
```
sudo apt-get install libsndfile-dev
pip3 install -r requirements.txt
```
如果空间不足可以使用 --prefix 指定别的安装路径


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


### 示例:  
example下有测试音频  

如 粤语测试
```
python3 main.py -i example/yue.mp3
```
输出
```
RTF: 0.03026517820946964    Latency: 0.15689468383789062s  Total length: 5.184s
['呢几个字。', '都表达唔到，我想讲嘅意。', '思。']
```

## 准确率

使用WER(Word-Error-Rate)作为评价标准  

**WER = 3.74%**  

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