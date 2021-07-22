#!/bin/bash
#SBATCH --account=rrg-majidk
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=08:00:00


source $HOME/dlenv/bin/activate
SOURCEDIR=$HOME/Tactile/Pix2Pix
mkdir $SLURM_TMPDIR/data
tar xf ./tactile1000.tar -C $SLURM_TMPDIR/data
module load cuda cudnn
python3 $SOURCEDIR/Pix2Pix.py $SLURM_TMPDIR
tar cf saved_models.tar $SLURM_TMPDIR/models/*

