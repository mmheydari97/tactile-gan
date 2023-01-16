#!/bin/bash
#SBATCH --account=rrg-majidk
#SBATCH --gres=gpu:v100l:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=13:00:00

source $HOME/dlenv/bin/activate
SOURCEDIR=$HOME/Tactile/tactile-gan
mkdir $SLURM_TMPDIR/data
# mkdir $SLURM_TMPDIR/temp

# unzip -qq $HOME/Tactile/dataset/data_bar.zip -d $SLURM_TMPDIR
tar -xzf $HOME/Tactile/dataset/plots_Jan23.tar.gz -C $SLURM_TMPDIR

# tar -xzf saved_m_bar.tar.gz -C $SLURM_TMPDIR/temp
# mv $SLURM_TMPDIR/temp/*/*/models $SLURM_TMPDIR

module load cuda cudnn
# ${w_param[*]}

python3 $SOURCEDIR/train.py --data $SLURM_TMPDIR/data --version 2 --folder_save "v2plot_rgb" --folder_load "v2plot_rgb"
tar -czf "saved_v2_plot_rgb.tar.gz" $SLURM_TMPDIR/models/*
# tar -czf saved_ch_v2_plot_rgb.tar.gz $SLURM_TMPDIR/checkpoints/*


