# 模型转换

## 环境准备

已在Ubuntu Python3.10上验证  

```
conda create -n sensevoice python=3.10
conda activate sensevoice
pip install -r requirements.txt
```

## 导出ONNX

```
python export_onnx.py
```

可传入export_onnx.py中的max_seq_len以输入模型的特征长度，目前为256    
导出成功后生成output_dir/model.onnx


## 导出量化数据集

```
python generate_data.py
```

## 导出axmodel

```
pulsar2 build --input output_dir/model.onnx --config sensevoice.json --output_dir axmodel --output_name sensevoice.axmodel
```
运行成功后生成 axmodel/sensevoice.axmodel