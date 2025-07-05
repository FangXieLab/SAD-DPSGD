# Paper Title: Steps Adaptive Decay DPSGD: Enhancing Performance on Imbalanced Datasets with Differential Privacy with HAM10000 

## Description
This repository contains the implementation and supplementary materials for the paper titled "Steps Adaptive Decay DPSGD: Enhancing Performance on Imbalanced Datasets with Differential Privacy with HAM10000" by Xiaobo Huang.


You can install the required dependencies using the following command:
```bash
pip install -r requirements.txt
```


To run DP-SGD on HAM10000 dataset:
```bash
python main.py --eps=3 --delta=1e-3 --Method='DP-SGD'
python main.py --eps=8 --delta=1e-3 --Method='DP-SGD'
python main.py --eps=16 --delta=1e-3 --Method='DP-SGD'
```
To implement Auto-DP-SGD-L on HAM10000 dataset:
```bash
python main.py --eps=3 --delta=1e-3 --Method='Auto-DP-SGD-L'
python main.py --eps=8 --delta=1e-3 --Method='Auto-DP-SGD-L'
python main.py --eps=16 --delta=1e-3 --Method='Auto-DP-SGD-L'
```
To implement Auto-DP-SGD-S on HAM10000 dataset with noise multipliers decay parameter beta=0.8
```bash
python main.py --eps=3 --delta=1e-3 --q=0.8 --Method='Auto-DP-SGD-S'
python main.py --eps=8 --delta=1e-3 --q=0.8 --Method='Auto-DP-SGD-S'
python main.py --eps=16 --delta=1e-3 --q=0.8 --Method='Auto-DP-SGD-S'
```

To implement SAD-DP-SGD on HAM10000 dataset with noise multipliers decay parameter beta=0.8 and steps decay parameter gamma=0.9 and number of step n = 3
```bash
python main.py --eps=3 --delta=1e-3 --q=0.8 --qR=0.9 --numOfstep=3 --Method='SAD-DPSGD'
python main.py --eps=8 --delta=1e-3 --q=0.8 --qR=0.9 --numOfstep=3 --Method='SAD-DPSGD'
python main.py --eps=16 --delta=1e-3 --q=0.8 --qR=0.9 --numOfstep=3 --Method='SAD-DPSGD'
```