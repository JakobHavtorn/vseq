# ICLR 2022 experiments

CWVAE
- L=1 d=96      s=64
- L=2 d=96      s=64,512
- L=3 d=96      s=64,512,4096
- L=1 d=96 g=96 s=64
- L=2 d=96 g=96 s=64,512
- L=3 d=96 g=96 s=64,512,4096

- L=2 d=256 (V100): sweet-sunset-129 1xrnjn5y
- L=1 d=256 (V100): fresh-galaxy-128 2oi2n3t4



```
env CUDA_VISIBLE_DEVICES="9" WANDB_NOTES="1L DLM 1 mix TasNetCoder (z=96, s=64) (MuLaw 16bit)" python experiments/experiment_cwvae_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 --latent_size 96 --time_factors 64 --input_coding mu_law --num_bits 16 --num_mix
 1 --save_checkpoints True

 env CUDA_VISIBLE_DEVICES="8" WANDB_NOTES="2L DLM 1 mix TasNetCoder (z=96, s=64) (MuLaw 16bit)" python experiments/experiment_cwvae_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 96 --latent_size 96 96 --time_factors 64 512 --input_coding mu_law --num_bits 16 --num_mix
 1 --save_checkpoints True

  env CUDA_VISIBLE_DEVICES="7" WANDB_NOTES="2L DLM 1 mix TasNetCoder (z=96, s=64) (MuLaw 16bit)" python experiments/experiment_cwvae_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 96 96 --latent_size 96 96 96 --time_factors 64 512 4096 --input_coding mu_law --num_bits 16 --num_mix
 1 --save_checkpoints True
```