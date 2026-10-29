#!/bin/bash
# Submit from this directory: sbatch odm_reward_fsdp.sbatch
#SBATCH -p gpu-mid
#SBATCH --gres=gpu:h200:2
#SBATCH -c 16
#SBATCH --mem=400G
#SBATCH -t 2:00:00
#SBATCH --exclude=p1-r18-n1,p1-r10-n1,p3-r10-n2,p1-r30-n3,p1-r13-n3
#SBATCH -o odm_reward_fsdp-%j.out
B=/proj/granite-build/g4os/pylibs
D=/proj/granite-build/g4os/defense/defense_xmldata_v2/20260914T194134
M=/proj/granite-build/g4os/models/granite-5.1-20b-sft-preview-r261020a-eager
srun --container-image=/proj/granite-build/g4os/enroot/us.icr.io-cil15-shared-registry-fms-hf-tuning-6mar2026-a66092d5c01.sqsh \
  --container-mounts=/proj:/proj --container-writable bash -c "
  SP=\$(python3 -c \"import tuning,os;print(os.path.dirname(os.path.dirname(tuning.__file__)))\")
  R7=\$(ls -d \$SP/torch-2.7*.dist-info 2>/dev/null | head -1); R8=\$(ls -d \$SP/torch-2.8*.dist-info 2>/dev/null | head -1)
  if [ -n \"\$R7\" ] && [ -n \"\$R8\" ]; then comm -23 <(cut -d, -f1 \$R7/RECORD | grep \"^torch/.*\.py$\" | sort) <(cut -d, -f1 \$R8/RECORD | sort) | (cd \$SP && xargs -r rm -f); fi; test ! -e \$SP/torch/_inductor/kernel/mm_scaled.py && echo STALE_TORCH27_REMOVED
  export PYTHONPATH=$B/swash-fms-tuning-overlay-6mar:$B/swash-transformers-5.18-6mar
  export HF_HOME=/tmp/hf TOKENIZERS_PARALLELISM=false
  cd "$SLURM_SUBMIT_DIR" && torchrun --nproc_per_node=2 odm_reward_fsdp.py $M $D 120 8"
