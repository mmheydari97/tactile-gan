#!/bin/bash
#SBATCH --account=rrg-majidk
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=23:00:00

source $HOME/dlenv/bin/activate
SOURCEDIR=$HOME/Tactile/Pix2Pix
mkdir $SLURM_TMPDIR/data
# mkdir $SLURM_TMPDIR/temp

unzip -qq $HOME/Tactile/data.zip -d $SLURM_TMPDIR
# tar -xzf saved_model.tar.gz -C $SLURM_TMPDIR/temp
# mv $SLURM_TMPDIR/temp/*/*/models $SLURM_TMPDIR

module load cuda cudnn
python3 $SOURCEDIR/train.py --dir $SLURM_TMPDIR --total_iters 135 --lambda_per 0.1 --batch_size 4 --w_per 1 0 0 0 --no_aug --label_smoothing
tar -czf saved_m_b4_per.tar.gz $SLURM_TMPDIR/models/*
# tar -czf saved_ch_unet_reg.tar.gz $SLURM_TMPDIR/checkpoints/*


