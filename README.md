# AI4Life-MDC24-submission

This algorithm use Self2SelfPlus code from https://github.com/JK-the-Ko/Self2SelfPlus/tree/main
A single image is trained and denoised using self-supervised learning task with Image Quality Assessment Loss.

## Installation

pip install -r requirements.txt


## Training

python train.py --dataType Hagen --p 0.9 --numIters 10000

## Evaluaton

After the training, the result will be save in test/output/images/image-stack-structured-noise folder