# 模型转换

支持 AX650N / AX630C / AX620Q 三芯的 SenseVoice 模型转换。

## 环境准备

Ubuntu Python3.10+（620Q 导出验证在 Python3.12 上通过）：

```bash
conda create -n sensevoice python=3.10
conda activate sensevoice
pip install -r requirements.txt
```

## 导出 ONNX

```bash
# AX650N/AX630C（原脚本）
python export_onnx.py

# AX620Q（适配新版 funasr 的静态导出，输入 speech/mask/language）
python export_onnx_v2.py --max_seq_len 256 --output_dir output_dir --onnx_name model.onnx
```

`max_seq_len` 默认 256（对应 2.56s 音频/次推理）。导出成功后生成 `output_dir/model.onnx`（静态 shape）。

## 导出量化数据集

```bash
python generate_data.py
```

## 导出 axmodel（Pulsar2）

```bash
# AX650N
pulsar2 build --input output_dir/model.onnx --config sensevoice.json --output_dir axmodel/ax650 --output_name sensevoice.axmodel

# AX620E NPU1（AX620Q / AXera Pi Zero）
pulsar2 build --input output_dir/model.onnx --config pulsar2_ax620e_npu1_int8.json --output_dir axmodel/ax620q --output_name sensevoice.axmodel

# AX620E NPU2（AX630C full-core 对比用）
pulsar2 build --input output_dir/model.onnx --config pulsar2_ax620e_npu2_int8.json --output_dir axmodel/ax620e_npu2 --output_name sensevoice.axmodel
```

运行成功后生成对应目录下的 `sensevoice.axmodel`。

## 备注

- 620Q（AX620E NPU1）为 INT8 + smooth quant 方案；U16 全层配置在该目标上编译异常慢，不推荐。
- 模型目录结构需为 `sensevoice_<chip>/sensevoice.axmodel`（650/630C/620Q），见根目录 `download_models.sh`。
