#!/bin/bash
#SBATCH -J smoke-tuning-slurm
#SBATCH -p gpu-mid
#SBATCH --account=grp_granite_dot_build
#SBATCH --nodes=1 --ntasks-per-node=1 --cpus-per-task=64 --mem=256G
#SBATCH --gres=gpu:8 --constraint=h200
#SBATCH --exclude=p1-r18-n1,p1-r10-n1
#SBATCH -t 1:30:00
#SBATCH -o /proj/granite-build/g4os/defense/smoke-tuning-slurm/slurm-%j.out
srun --container-image=/proj/granite-build/g4os/enroot/us.icr.io-cil15-shared-registry-fms-hf-tuning-6mar2026-a66092d5c01.sqsh   --container-mounts=/proj/granite-build/g4os:/proj/granite-build/g4os --container-remap-root --container-writable --no-container-mount-home   bash -c 'export HF_HOME=/tmp/hf_home NCCL_TIMEOUT=10800000; cd /proj/granite-build/g4os/defense/smoke-tuning-slurm && bash setup.sh && bash launch.sh'
