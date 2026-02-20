#!/bin/sh
### General options
### -- specify queue --
#BSUB -q gpul40s
### -- set the job Name --
#BSUB -J colony_train
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 1 gpu in exclusive process mode --
#BSUB -gpu "num=1:mode=exclusive_process"
### -- set walltime limit: hh:mm -- maximum 24 hours for GPU-queues --
#BSUB -W 24:00
### -- request 32GB of system-memory --
#BSUB -R "rusage[mem=32GB]"
### -- send notification at start --
#BSUB -B
### -- send notification at completion --
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
#BSUB -o logs/train_%J.out
#BSUB -e logs/train_%J.err
# -- end of LSF options --

set -e

# Load necessary modules
module load cuda/11.6

# Change to the correct working directory
cd /work3/s203512/colonyCounter/src


# Create directories
mkdir -p logs

# Add uv to PATH
export PATH="$HOME/.cargo/bin:$HOME/.local/bin:$PATH"

# Show GPU info
nvidia-smi

# Run training
echo "Starting colony training..."
/zhome/fa/f/155129/.local/bin/uv run --with typer python train/train_colony.py \
    --dataset-dir /work3/s203512/colonyCounter/dataset \
    --output-dir output/colony \
    --epochs 100 \
    --batch-size 4 \
    --num-workers 4

echo "Training complete!"
