#!/bin/bash
#PBS -l select=1:ncpus=2:gpu_id=0
#PBS -o out.txt				
#PBS -e err.txt				
#PBS -N yolov7-seg

cd /Data/home/chriswang/project/MBF-UNet

source ~/.bashrc											
conda activate cv2				

module load cuda-11.7										
python3 trainer.py
