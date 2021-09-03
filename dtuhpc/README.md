# HPC Cluster at DTU

## Accessing the cluster
```bash 
ssh user@login1.hpc.dtu.dk
ssh user@login2.hpc.dtu.dk
```

Home dir
`/zhome/c2/b/86488`

1. `git clone https://github.com/JakobHavtorn/vseq.git`
2. Create `VSEQ.env` file containing the `VSEQ_DATA_ROOT_DIRECTORY`.
3. Prepare any datasets to be used by running `scripts/data/prepare_<name>.py`


## Submit a job
```bash
bsub -J "CW-VAE" -env WANDB_NOTES="V100 HPC test run" -oo "dtuhpc/logs/%J.out" -eo "dtuhpc/logs/%J.out" -q gpujdha -u "jdh@corti.ai" -B -N -W 672:00 -n 4 -gpu "num=4:mode=exclusive_process" -R "span[hosts=1] rusage[mem=6GB]" "bash dtuhpc/hpc_run_job.sh 'python3 experiments/experiment_cwvae_audio_ddp.py --gpus 4 --num_workers 4 --epochs 1000 --free_nats_start_value 4 --free_nats_steps 80000 --hidden_size 256 --latent_size 256 --time_factors 64 --input_coding mu_law --num_bits 16 --save_checkpoints True'"
```


## Monitor a job
```bash
nodestat -G gpujdha  # Check status of hardware for queue
qstat gpujdha  # Check queue status

bstat jobid  # Show job status
bkill jobid  # Kill job
bstat -C  # Check efficiency
bstat -M  # Check memory
```


## References
https://www.hpc.dtu.dk/?page_id=2759
https://www.hpc.dtu.dk/?page_id=1519
