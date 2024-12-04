# HGLS-PE

## 使用方法

### 硬件要求
```
24 g 显存，64g 内存
```
### 环境配置 (linux)
```
conda create -n tkg python==3.10
conda activate tkg
conda install pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.1 -c pytorch -c nvidia
conda install -c dglteam/label/th24_cu121 dgl
pip install -r requirements.txt
```
### 生成数据
```
python generate_data.py --data ICEWS14s (or ICEWS18)

python save_data.py --data ICEWS18 (14s 不需要)
```
### 训练和测试
```
# 其他调整见 main.py long_config.yaml short_config.yaml
14.sh
18.sh
```

# 说明
```
该项目的大部分源码源于 https://github.com/CRIPAC-DIG/HGLS
```
