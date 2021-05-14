#!/bin/bash
#
#SBATCH --job-name=test_albert
#SBATCH --partition=metros1x
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=1
#SBATCH --mem=1000
#SBATCH --priority=None
#SBATCH -D /home/abou/meta_gradient_experiment
#SBATCH --output=slurm.%N.%j.out
#SBATCH --error=slurm.%N.%j.err
#SBATCH --export=ACEMD_HOME,HTMD_LICENSE_FILE

trap "touch /home/abou/meta_gradient_experiment/jobqueues.done" EXIT SIGTERM


cd /home/abou/meta_gradient_experiment
/home/abou/meta_gradient_experiment/run.sh