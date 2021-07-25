# HPC Cluster at DTU

## Access
Login nodes
`ssh jdha@login1.hpc.dtu.dk`
`ssh jdha@login2.hpc.dtu.dk`

Transfer nodes
`ssh jdha@transfer.gbar.dtu.dk`

Home directory
`/zhome/c2/b/86488`

1. `git clone https://github.com/JakobHavtorn/vseq.git`
2. Create `VSEQ.env` file containing the `VSEQ_DATA_ROOT_DIRECTORY`.
3. Prepare any datasets to be used by running `scripts/data/prepare_<name>.py`

## Submit a job
```bash
bsub -J "CW-VAE" -env "WANDB_NOTES='1L DLM 10 mix TasNetCoder (160) (MuLaw 16bit) (DDP) (V100) (dim=256)'" -oo "dtuhpc/logs/%J.out" -eo "dtuhpc/logs/%J.err" -q gpujdha -u "jdh@corti.ai" -B -N -W 672:00 -n 4 -gpu "num=4:mode=exclusive_process" -R "span[hosts=1] rusage[mem=6GB]" "bash dtuhpc/hpc_run_job.sh 'python3 experiments/experiment_cwvae_audio_ddp.py --gpus 4 --num_workers 4 --epochs 1000 --free_nats_start_value 4 --free_nats_steps 120000 --hidden_size 256 --latent_size 128 --time_factors 160 --input_coding mu_law --num_bits 16 --save_checkpoints True --wandb_tags TasNet'"
```

```bash
bsub < dtuhpc/hpc_queue_script.sh
```

## Monitor a job
```bash
qstat gpujdha

bstat jobid
bkill jobid
bstat -C  # Efficiency
bstat -M  # Memory
```

## References
https://www.hpc.dtu.dk/?page_id=2759
https://www.hpc.dtu.dk/?page_id=1519
