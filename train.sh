#!/bin/bash

# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate pytorch_train

# Run the training script with nohup so it continues after terminal closes
nohup python -u train.py > train.log 2>&1 &
echo "Training started in background. Check train.log for output."

# To follow the log print
# tail -f train.log

# To stop the training process, you can use the following command:
# pkill -f "python -u train.py"