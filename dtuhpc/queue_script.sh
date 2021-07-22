#!/bin/sh

# -- start of LSF options --
### General options
### –- specify queue --
#BSUB -q gpujdha
### -- set the job Name --
#BSUB -J $(uuidgen)
### -- ask for number of cores (default: 1) --
#BSUB -n 4
### -- Select the resources: 4 gpu in exclusive process mode --
#BSUB -gpu "num=4:mode=exclusive_process"
### -- set walltime limit: hh:mm --  maximum 24 hours for GPU-queues right now
#BSUB -W 672:00
# request 10GB of system-memory
#BSUB -R "span[hosts=1] rusage[mem=10GB]"
### -- set the email address --
# if you want to receive e-mail notifications on a non-default address
#BSUB -u jdh@corti.ai
### -- send notification at start --
#BSUB -B
### -- send notification at completion--
#BSUB -N
### -- Specify the output and error file. %J is the job-id --
### -- -o and -e mean append, -oo and -eo mean overwrite --
#BSUB -o dtuhpc/logs/%J.out
#BSUB -e dtuhpc/logs/%J.out
# -- end of LSF options --

bash dtuhpc/run_hpc_job.sh "$@"  # Pass all arguments along to run script
