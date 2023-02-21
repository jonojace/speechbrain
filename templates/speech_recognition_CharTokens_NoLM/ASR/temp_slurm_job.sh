#!/bin/bash
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --time=4-00:00:00
#SBATCH --partition=ILCC_GPU,CDT_GPU,ILCC_GPU_UBUNTU
#SBATCH --mail-user=s1785140@sms.ed.ac.uk
#SBATCH --mail-type=BEGIN,END,FAIL,INVALID_DEPEND,REQUEUE,STAGE_OUT
#SBATCH --output=./logs_slurm/%j

#### Initialise wandb ####
export WANDB_API_KEY=afb11ca4f70d9fc0dfab01f24f1dece4c707cd36
wandb login
wandb online

#### create the command to be run on the cluster ####
cmd_to_run_on_cluster="python train.py train_subsampling1x_nowhitespace.yaml"


#### the command that will be run ####
echo Running the following command on slurm: ${cmd_to_run_on_cluster}
${cmd_to_run_on_cluster}
