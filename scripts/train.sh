#!/bin/bash

# set up the Data and checkpoint locations
DATABIN=/lustre/data/wikitext-103
OUTDIR=/lustre/data/out && mkdir -p $OUTDIR
SAVEDIR=/lustre/checkpoints

# set up environment variables for Torch DistributedDataParallel
WORLD_SIZE_JOB=$SLURM_NTASKS
RANK_NODE=$SLURM_NODEID
PROC_PER_NODE=8
MASTER_ADDR_JOB=(`scontrol show hostnames $SLURM_JOB_NODELIST`)
MASTER_PORT_JOB="12234"
DDP_BACKEND=c10d

# setup NCCL to use EFA
export FI_PROVIDER=efa
export FI_EFA_TX_MIN_CREDITS=64
export NCCL_DEBUG=INFO
export NCCL_TREE_THRESHOLD=0
export NCCL_SOCKET_IFNAME=eth0
export LD_LIBRARY_PATH=/home/ec2-user/nccl/build/lib:/usr/local/cuda/lib64:/opt/amazon/efa/lib64:/opt/amazon/openmpi/lib64:$LD_LIBRARY_PATH

# set up fairseq-train additional arguments
BUCKET_CAP_MB=200
TOTAL_UPDATE=500
MAX_SENTENCES=8
UPDATE_FREQ=1

# calling fairseq-train
torchrun \
    --nproc_per_node=$PROC_PER_NODE \
    --nnodes=$WORLD_SIZE_JOB \
    --node_rank=$RANK_NODE \
    --master_addr=${MASTER_ADDR_JOB} \
    --master_port=${MASTER_PORT_JOB} \
    $(which fairseq-train) \
    $DATABIN \
    --log-format json \
    --log-interval 25 \
    --seed 1 \
    --fp16 \
    --memory-efficient-fp16 \
    --criterion masked_lm \
    --optimizer adam \
    --lr-scheduler polynomial_decay \
    --task masked_lm \
    --num-workers 2 \
    --max-sentences $MAX_SENTENCES \
    --ddp-backend $DDP_BACKEND \
    --bucket-cap-mb $BUCKET_CAP_MB \
    --fast-stat-sync \
    --arch roberta_large \
    --max-epoch 2 \
    --max-update $TOTAL_UPDATE \
    --clip-norm 1.0 \
    --update-freq $UPDATE_FREQ \
    --lr 0.0006 \
    --save-dir $SAVEDIR \
    --sample-break-mode complete \
    --tokens-per-sample 512 \
    --adam-betas '(0.9, 0.98)' \
    --adam-eps 1e-06 \
    --warmup-updates 24000 \
    --total-num-update $TOTAL_UPDATE \
    --dropout 0.1 \
    --attention-dropout 0.1 \
    --weight-decay 0.01 | tee $OUTDIR/train.$RANK.$WORLD_SIZE.log
