# Bayesian CSRNet
We provide a PyTorch implementation our improved [CSRNet](https://github.com/leeyeehoo/CSRNet-pytorch) which is based on the [Bayesian crowd counting](https://github.com/ZhihengCV/Bayesian-Crowd-Counting), called CSRNet.

For dataset and pretrained models refer to our [paper](https://github.com/tersekmatija/crowd-counting-cnns).

## Instructions

### 1. Preprocessing data
```
python preprocess_dataset.py --origin_dir <directory of original data> --data_dir <directory of processed data>
```
### 2. Training
```
python train.py --data_dir <directory of processed data> --save_dir <directory of log and model>
```
### 3. Evaluation
```
python test.py --data_dir <directory of processed data> --save_dir <directory of log and model>
```

## License
This project is modified from [Bayesian Crowd Counting project](https://github.com/ZhihengCV/Bayesian-Crowd-Counting) and is originally licensed under MIT | Copyright (c) 2020 Zhiheng_Ma.