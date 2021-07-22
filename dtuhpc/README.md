# HPC Cluster at DTU

`ssh jdha@login1.hpc.dtu.dk`
`ssh jdha@login2.hpc.dtu.dk`

Home dir
`/zhome/c2/b/86488`

1. `git clone https://github.com/JakobHavtorn/vseq.git`
2. Create `VSEQ.env` file containing the `VSEQ_DATA_ROOT_DIRECTORY`.
3. Prepare any datasets to be used by running `scripts/data/prepare_<name>.py`


`nodestat -G gpujdha`
`qstat gpujdha`

`bsub -q gpujdha -u "jdh@corti.ai" -B -N -o "test.out" -e "test.err" -J testjob -W 1:00 -n 1 -gpu "num=1:mode=exclusive_process" -R "rusage[mem=6GB]" "echo 'Hello world' > test.log"`
The `test.log`, `test.err` and `test.out` files are created at the current working directory.


Submit a job from `root/path/Documents/vseq`
```bash
UUID=$(uuidgen)
bsub -J $UUID -oo "dtuhpc/logs/$UUID.out" -eo "dtuhpc/logs/$UUID.out" -q gpujdha -u "jdh@corti.ai" -B -N -W 1:00 -n 4 -gpu "num=4:mode=exclusive_process" -R "span[hosts=1] rusage[mem=6GB]" "bash dtuhpc/hpc_run_job.sh 'python3 experiments/experiment_cwvae_audio_ddp.py --gpus 4 --num_workers 4 --epochs 1000 --free_nats_start_value 4 --free_nats_steps 80000 --hidden_size 96 --latent_size 96 --time_factors 64 --input_coding mu_law --num_bits 16 --save_checkpoints True'"
```

```
UUID=$(uuidgen)
bsub -J $UUID -oo "dtuhpc/logs/$UUID.out" -eo "dtuhpc/logs/$UUID.out" -q gpujdha -u "jdh@corti.ai" -B -N -W 1:00 -n 4 -gpu "num=4:mode=exclusive_process" -R "span[hosts=1] rusage[mem=6GB]" "bash dtuhpc/hpc_run_job.sh 'python3 test.py'"
```

## Submit a job
```bash
bsub < dtuhpc/hpc_queue_script.sh "command to run"
```
For instance

```bash
export CMD="WANDB_NOTES='V100 test run' python3 experiments/experiment_cwvae_audio_ddp.py --gpus 4 --num_workers 4 --epochs 3000 --free_nats_start_value 4 --free_nats_steps 1500000 --hidden_size 256 --latent_size 256 --time_factors 64 --input_coding mu_law --num_bits 16 --save_checkpoints True"
bsub < dtuhpc/hpc_queue_script.sh $CMD
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
