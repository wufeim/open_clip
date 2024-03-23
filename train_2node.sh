#!/bin/bash

# Parameters
#SBATCH --error=/home/wma27/open_clip/exp/open_clip_dediffusion/%j_0_log.err
#SBATCH --gpus-per-node=8
#SBATCH --job-name=open_clip_dediffusion
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --open-mode=append
#SBATCH --output=/home/wma27/open_clip/exp/open_clip_dediffusion/%j_0_log.out
#SBATCH --partition=main
#SBATCH --time=2880
#SBATCH --mail-type=BEGIN,END
#SBATCH --mail-user=wma27@jh.edu
#SBATCH --cpus-per-task=4
#SBATCH --mem=256G
#SBATCH --exclude=ccvl[14,33-38]

module purge
module load conda

conda activate open_clip

which python

export MASTER_PORT=12802
master_addr=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_ADDR=$master_addr

cd src
torchrun --nproc_per_node 8 -m training.main \
    --train-data /datasets/wma27/imagenet1k/train \
    --val-data /datasets/wma27/imagenet1k/small_small_val \
    --batch-size 24 \
    --precision amp \
    --workers 4 \
    --model "dediffusion_ViT-B-32" \
    --report-to "wandb" \
    --accum-freq 8 \
    --epochs 10 \
    --logs /home/wma27/open_clip/logs
