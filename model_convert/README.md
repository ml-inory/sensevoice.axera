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
python export.py
```

可修改export.py中的seq_len以输入模型的特征长度，目前为68  
导出成功后生成output_dir/model.onnx

简化ONNX  
```
onnxslim output_dir/model.onnx output_dir/model_sim.onnx
```


## 导出axmodel

```
pulsar2 build --input output_dir/model_sim.onnx --config sensevoice.json --output_dir axmodel --output_name sensevoice.axmodel
```
运行成功后生成 axmodel/sensevoice.axmodel