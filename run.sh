#!/bin/sh

# python2.7, mxnet-cu90, NVIDIA TITAN Xp, cuda9 cudnn7
# create ./data folder, put base,web,rank,z_rank in ./data folder
# create ./output folder

rm -rf train.log

python data.py >>train.log 2>&1
python finetune.py --task collar_design_labels --model resnet152_v2 --epochs 20 >>train.log 2>&1
python finetune.py --task neckline_design_labels --model resnet152_v2 --epochs 20 >>train.log 2>&1
python finetune.py --task skirt_length_labels --model resnet152_v2 --epochs 20 >>train.log 2>&1
python finetune.py --task sleeve_length_labels --model resnet152_v2 --epochs 20 >>train.log 2>&1
python finetune.py --task neck_design_labels --model resnet152_v2 --epochs 20 >>train.log 2>&1
python finetune.py --task coat_length_labels --model resnet152_v2 --epochs 20 >>train.log 2>&1
python finetune.py --task lapel_design_labels --model resnet152_v2 --epochs 20 >>train.log 2>&1
python finetune.py --task pant_length_labels --model resnet152_v2 --epochs 20 >>train.log 2>&1

cd output
cat collar_design_labels.csv neckline_design_labels.csv skirt_length_labels.csv sleeve_length_labels.csv neck_design_labels.csv coat_length_labels.csv lapel_design_labels.csv pant_length_labels.csv >submit.csv

