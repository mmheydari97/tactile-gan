#!/bin/bash
#SBATCH --account=rrg-majidk
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=7:00:00

source $HOME/dlenv/bin/activate
SOURCEDIR=$HOME/Tactile/Pix2Pix
mkdir $SLURM_TMPDIR/data
# mkdir $SLURM_TMPDIR/temp

unzip -qq $HOME/Tactile/dataset/data_bar.zip -d $SLURM_TMPDIR
# tar -xzf saved_m_bar.tar.gz -C $SLURM_TMPDIR/temp
# mv $SLURM_TMPDIR/temp/*/*/models $SLURM_TMPDIR

module load cuda cudnn
# ${w_param[*]}

python3 $SOURCEDIR/train.py --dir $SLURM_TMPDIR --folder_save "bar_ce" --folder_load "bar_ce"
tar -czf "saved_m_bar_ce.tar.gz" $SLURM_TMPDIR/models/*
# tar -czf saved_ch_b24.tar.gz $SLURM_TMPDIR/checkpoints/*


