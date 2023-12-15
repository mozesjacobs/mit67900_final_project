#!/bin/bash
#SBATCH -c 2
#SBATCH -t 0-02:00
#SBATCH -p kempner_requeue
#SBATCH --mem=2g 
#SBATCH --gres=gpu:1 
#SBATCH --open-mode=append
#SBATCH -o hostname_%j.out 
#SBATCH -e hostname_%j.err
#SBATCH --mail-type=FAIL 
#SBATCH --account=kempner_ba_lab

# COPY FILES + MOVE DIRECTORY
rsync -avx ../ /n/holyscratch01/ba_lab/Users/mjacobs/mit67900_final_project/
cd /n/holyscratch01/ba_lab/Users/mjacobs/mit67900_final_project/experiments/exp1_15/

# SETUP ENVIRONMENT
module load gcc/13.2.0-fasrc01
source ~/.bashrc
mamba activate meow

# RUN EXPERIMENT
python ../train_ppo.py

# DEACTIVATE ENVIRONMENT
mamba deactivate

# SAVE RESULTS
rsync -avx results/ppo/ ~/mit67900_final_project/experiments/exp1_15/results/ppo/