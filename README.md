## Distributed Machine Learning with ParallelCluster

### Introduction

In this repo we show how to train the RoBERTa large language model from scratch in a distributed environment using AWS ParallelCluster, Slurm scheduler, Fairseq, AWS EFA, Nvidia NCCL, and PyTorch Distributed Data Parallel (DDP), with AWS p3dn instances. This repo largely follows this [workshop](https://www.hpcworkshops.com/09-ml-on-parallelcluster.html).


- [AWS ParallelCluster](https://aws.amazon.com/hpc/parallelcluster/): an AWS supported open source cluster management tool that helps deploy and manage high performance computing (HPC) clusters in the AWS Cloud.

- [Slurm](https://slurm.schedmd.com/documentation.html): a scheduler supported by ParallelCluster that is used to schedule jobs in the distributed environment.  

- [AWS Elastic Fabric Adapter (EFA)](https://aws.amazon.com/hpc/efa/): a network interface for Amazon EC2 instances that enables customers to run applications requiring high levels of inter-node communications at scale on AWS. With EFA, High Performance Computing (HPC) applications using the Message Passing Interface (MPI) and Machine Learning (ML) applications using NVIDIA Collective Communications Library (NCCL) can scale to thousands of CPUs or GPUs.  


- [Nvidia Collective Communications Library (NCCL)](https://developer.nvidia.com/nccl): a library of standard collective communication routines for multiple GPUs across a single node or multiple nodes. NCCL can be used together with EFA, Libfabric, and MPI to support various machine learning workloads. Note that as of now, NCCL with EFA is supported with `p3dn.24xlarge`, `p4d.24xlarge`, and `p5.48xlarge` instances only ([source](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start-nccl.html)).


- [PyTorch DistributedDataParallel (DDP)](https://pytorch.org/tutorials/intermediate/ddp_tutorial.html): implements data parallelism at the module level which can run across multiple machines.

- [Fairseq](https://ai.facebook.com/tools/fairseq/): a sequence modeling toolkit for training custom models for translation, summarization, and other text generation tasks.


### Model and Dataset

We use the `RoBERTa large` model in this repo. The details about this model can be found [here](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.md). Briefly, RoBERTa is a replication study of BERT with better design and improved performance, and the architecture is the same with BERT. We can train the model using our own data ([reference](https://github.com/facebookresearch/fairseq/blob/main/examples/roberta/README.pretraining.md)) and load the model that we trained. 


Here we use the wikitext-103 dataset as an example, which can be downloaded from [here](https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip) (181 MB). The WikiText language modeling dataset is a collection of over 100 million tokens extracted from the set of verified "Good" and "Featured" articles on Wikipedia. The version we use here is at word level (instead of the character level). The dataset contains `wiki.train.tokens`, `wiki.valid.tokens`, and `wiki.test.tokens`. No processing is needed other than replacing newlines with tokens. More details of this dataset can be found [here](https://blog.salesforceairesearch.com/the-wikitext-long-term-dependency-language-modeling-dataset/). 

```
# downloading data:
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-v1.zip -O wikitext-103-v1.zip
unzip wikitext-103-v1.zip

# upload to S3 bucket
aws s3 cp wikitext-103 s3://my-ml-bucket/wikitext-103 --recursive
```


### Create a Cluster Using ParallelCluster

- ParallelCluster can be installed following the instruction [here](https://docs.aws.amazon.com/parallelcluster/latest/ug/install-v3-virtual-environment.html).  

- A cluster can be configured and created following this [instruction](https://docs.aws.amazon.com/parallelcluster/latest/ug/install-v3-configuring.html). Here the configuration file `config/config_ml_fsxlustre_p3dn.yaml` was used to create a cluster using the following CLI command:

```
pcluster create-cluster --cluster-name myCluster --cluster-configuration config/config_ml_fsxlustre_p3dn.yaml
```

  - Instances: This cluster has one head node with the instance type of `c5n.2xlarge` and a compute queue with 2 so-called "dynamic" nodes of `p3dn.24xlarge` instance type (a dynamic node only gets turned on when needed and automatically turned off when the job is finished). 

  - Storage: Amazon FSx for Lustre file system is used as the storage solution, linked to a S3 bucket, and mounted to the `/lustre` path in the instance.

  - Bootstrap configuration: A OnNodeConfigured script (see `scripts/setup.sh`), stored in a S3 bucket, is used as a "custom action" in the bootstrap process. This script will be executed at the end of the instance bootstrap process to set up the environment needed for the model training, including:

    - Setting up conda virtual environment and install Fairseq
    - Setting up NCCL and EFA in compute nodes  

- SSM (or ssh) into the head node of the cluster. You can run NCCL test to make sure it's correctly set up.
```
sudo su - ubuntu

cd ~
git clone https://github.com/NVIDIA/nccl-tests.git
cd  nccl-tests/
make NCCL_HOME=$CUDA_DIRECTORY
NCCL_DEBUG=INFO build/all_reduce_perf -b 8 -f 2 -e 32M -c 1
```   

- In the head node, you can try simple Slurm commands such as:

```
sinfo
squeue
srun -N 1 hostname
srun -N 2 hostname
sbatch -N 1 --wrap "sleep 10"
sbatch -N 2 --wrap "sleep 10"
scontrol show job --details
```


### Train the RoBERTa Model

- Prepare scripts. Note that since this `/lustre` directory actually points to the S3 bucket that is linked to the FSx for Lustre file system, you have access to the files stored in this bucket, which can be a convenient way to prepare scripts, dataset and store training artifacts. Here we need to prepare these scripts: `scripts/preprocess.sh`, `scripts/job.slurm`, and `scripts/train.sh`.   
  - The `preprocess.sh` script uses `fairseq-preprocess` to preprocess the data.  
  - The `job.slurm` file sets up some Slurm environment variables, and uses `srun` to execute the `train.sh` script. For example, `--nodes=2` specifies that 2 nodes will be requested for the job.    
  - The `train.sh` script uses DDP `torchrun` (see reference [here](https://pytorch.org/docs/stable/elastic/run.html)) to execute the `fairseq` model training command with settings related to the distributed environment.  

- Activate the virtual environment

```
eval "$(/shared/.conda/bin/conda shell.bash hook)"
```

-  Preprocess the data on a single node

```
cd /lustre
srun --exclusive -n 1 preprocess.sh
```

- Train the model on a multi-node distributed environment

```
cd /lustre
sbatch job.slurm
```

- Note that you can monitor the job progress by executing the `tail -f slurm-<job-id>.out` command.  


- If you are interested in how to use a pre-trained LLM model for inference with or without fine-tuning, please take a look at this repo: https://github.com/delongmeng/Machine-Translation-LLM-Finetuning.


### Troubleshooting

- If you get "Insufficient Capacity Errors" (ICEs) when creating the cluster, you may consider EC2 [capacity reservation](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/capacity-reservations-using.html).

- The output of the OnNodeConfigured script for head node can be found here: `/var/log/cfn-init-cmd.log` and `/var/log/cfn-init.log`.

- The output of the OnNodeConfigured script for compute node can be found here: `/var/log/cloud-init-output.log`.

