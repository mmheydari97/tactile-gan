#!/bin/bash
#SBATCH --account=rrg-majidk
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:00:00

source $HOME/dlenv/bin/activate
SOURCEDIR=$HOME/Tactile/Pix2Pix
mkdir $SLURM_TMPDIR/data
mkdir $SLURM_TMPDIR/temp

tar -xzf ./tactile_int.tar.gz -C $SLURM_TMPDIR/data

tar -xzf saved_models_new.tar.gz -C $SLURM_TMPDIR/temp
mv $SLURM_TMPDIR/temp/*/*/models $SLURM_TMPDIR

module load cuda cudnn
python3 $SOURCEDIR/Pix2Pix.py --dir $SLURM_TMPDIR --total_iters 130 --iter_decay 150 --batch_size 1 --loss ls --lambda_A 5.0 --lambda_per 0.1
tar -czf saved_models_latest.tar.gz $SLURM_TMPDIR/models/*

