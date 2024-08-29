# DMixNet
DMixNet: A dendritic multi-layered perceptron architecture for image recognition

## Environment
- Python 3.9
- torch==2.0.1
- numpy==1.22.1
- einops, pandas, matplotlib

## How to run this code?
1. Create configure for the model.
2. Train a model by using the following command:
```commandline
python main.py --mode train --device cuda:0 --config ./configs/config.json
