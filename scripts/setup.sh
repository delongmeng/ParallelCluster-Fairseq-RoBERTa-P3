#!/bin/bash

export HOME=/home/ec2-user

# start configuration of NCCL and EFA only if CUDA and EFA present
CUDA_DIRECTORY=/usr/local/cuda
EFA_DIRECTORY=/opt/amazon/efa
OPENMPI_DIRECTORY=/opt/amazon/openmpi
if [ -d "$CUDA_DIRECTORY" ] && [ -d "$EFA_DIRECTORY" ]; then

    # installing NCCL
    NCCL_DIRECTORY=/home/ec2-user/nccl
    if [ ! -d "$NCCL_DIRECTORY" ]; then
        echo "[Node Setup] Installing NVIDIA nccl"
        cd /home/ec2-user
        git clone https://github.com/NVIDIA/nccl.git
        cd /home/ec2-user/nccl
        make -j src.build
        echo "[Node Setup] Finished Installing NVIDIA nccl"
    fi

    # installing aws-ofi-nccl
    AWS_OFI_DIRECTORY=/home/ec2-user/aws-ofi-nccl

    if [ ! -d "$AWS_OFI_DIRECTORY" ]; then
        echo "[Node Setup] Installing aws-ofi-nccl"
        cd /home/ec2-user
        git clone https://github.com/aws/aws-ofi-nccl.git -b aws
    fi
    cd $AWS_OFI_DIRECTORY

    ./autogen.sh
    ./configure --with-mpi=$OPENMPI_DIRECTORY --with-libfabric=$EFA_DIRECTORY --with-nccl=$NCCL_DIRECTORY/build --with-cuda=$CUDA_DIRECTORY
    export PATH=$OPENMPI_DIRECTORY/bin:$PATH
    make
    sudo make install
    echo "[Node Setup] Finished configure OFI with OPENMPI"
fi

# configuring the conda environment
cd /shared
CONDA_DIRECTORY=/shared/.conda/bin

if [ ! -d "$CONDA_DIRECTORY" ]; then
    # control will enter here if  doesn't exist.
    echo "[Node Setup] Conda installation not found. Installing..."
    wget -O miniconda.sh "https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh"
    PYTHON_VERSION=$(python --version 2>&1)
    echo "[Node Setup] Python version: $PYTHON_VERSION" 
    bash miniconda.sh -b -p /shared/.conda
    /shared/.conda/bin/conda init bash
    eval "$(/shared/.conda/bin/conda shell.bash hook)"
    CONDA_ENV=$(conda info --envs 2>&1)
    conda activate base
    CONDA_ENV=$(conda info --envs 2>&1)
    rm -rf miniconda.sh
    conda install python=3.6 -y
    echo "[Node Setup] Finished install Python 3.6"
fi

FAIRSEQ_DIRECTORY=/shared/fairseq

if [ ! -d "$FAIRSEQ_DIRECTORY" ]; then
    # control will enter here if  doesn't exist.
    echo "[Node Setup] Fairseq repository not found. Installing..."
    git clone https://github.com/pytorch/fairseq.git $FAIRSEQ_DIRECTORY
    PYTHON_VERSION=$(python --version 2>&1)
    pip install -e $FAIRSEQ_DIRECTORY -U
    echo "[Node Setup] Finished installing Fairseq"
    pip install boto3 torch tqdm
fi

chown -R ec2-user:ec2-user /lustre
chown -R ec2-user:ec2-user /shared

sudo -u ec2-user /shared/.conda/bin/conda init bash


PYTHON_VERSION=$(python --version 2>&1)
echo "[Node Setup] Python version: $PYTHON_VERSION"
CONDA_ENV=$(conda info --envs 2>&1)
echo "[Node Setup] Current conda environment: $CONDA_ENV"
echo "[Node Setup] All finished"
