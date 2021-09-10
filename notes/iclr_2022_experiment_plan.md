# ICLR 2022 experiments

TODO
- [ ] Fix error with global latent variable for more than 1 latent layer [We would like to run these experiments]
- [ ] Fix `terminated by signal SIGFPE (Floating point exception)` / `terminated by signal SIGSEGV (Address boundary error)` error for CW-VAE on hyperion, iapetus [We would like to run CW-VAE experiments on these GPUs]
- [ ] Fix issue with DDP trainig with global latent variable (expected len(lengths) to be ... batch size 1 and 4). Or is this the above?

CWVAE
- L=1 d=96      s=64
- L=2 d=96      s=64,512
- L=3 d=96      s=64,512,4096
- L=1 d=96 g=96 s=64
- L=2 d=96 g=96 s=64,512
- L=3 d=96 g=96 s=64,512,4096

- L=2 d=256 (V100): sweet-sunset-129 1xrnjn5y
- L=1 d=256 (V100): fresh-galaxy-128 2oi2n3t4

SRNN
- h=96 z=96 stack=200
- h=512 z=128 stack=200
- h=512 z=256 stack=200

VRNN
- h=96 z=96 stack=200
- h=512 z=128 stack=200
- h=512 z=256 stack=200


CWVAE
```
env CUDA_VISIBLE_DEVICES="9" WANDB_NOTES="L=1 DMoL=10 z=96 s=64 MuLaw=16bit TasNetCoder" python experiments/experiment_cwvae_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 --latent_size 96 --time_factors 64 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True

env CUDA_VISIBLE_DEVICES="8" WANDB_NOTES="L=2 DMoL=10 z=96 s=64 MuLaw=16bit TasNetCoder" python experiments/experiment_cwvae_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 96 --latent_size 96 96 --time_factors 64 512 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True

env CUDA_VISIBLE_DEVICES="7" WANDB_NOTES="L=3 DMoL=10 z=96 s=64 MuLaw=16bit TasNetCoder" python experiments/experiment_cwvae_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 96 96 --latent_size 96 96 96 --time_factors 64 512 4096 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True

# Above but DDP
env CUDA_VISIBLE_DEVICES="4,5" WANDB_NOTES="L=3 DMoL=10 z=96 s=64 MuLaw=16bit TasNetCoder" python experiments/experiment_cwvae_audio_ddp.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 96 96 --latent_size 96 96 96 --time_factors 64 512 4096 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True --gpus 2

```

CWVAE with global
```
env CUDA_VISIBLE_DEVICES="9" WANDB_NOTES="L=1 DMoL=10 z=96 g=96 s=64 MuLaw=16bit TasNetCoder" python experiments/experiment_cwvae_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 --latent_size 96 --global_size 96 --time_factors 64 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True

env CUDA_VISIBLE_DEVICES="8" WANDB_NOTES="L=2 DMoL=10 z=96 g=96 s=64 MuLaw=16bit TasNetCoder" python experiments/experiment_cwvae_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 96 --latent_size 96 96 --global_size 96 --time_factors 64 512 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True

env CUDA_VISIBLE_DEVICES="7" WANDB_NOTES="L=3 DMoL=10 z=96 g=96 s=64 MuLaw=16bit TasNetCoder" python experiments/experiment_cwvae_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 96 96 --latent_size 96 96 96 --global_size 96 --time_factors 64 512 4096 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True

# Above but DDP
env CUDA_VISIBLE_DEVICES="0,1,2,3" WANDB_NOTES="L=3 DMoL=10 z=96 g=96 s=64 MuLaw=16bit TasNetCoder" python experiments/experiment_cwvae_audio_ddp.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 96 96 --latent_size 96 96 96 --global_size 96 --time_factors 64 512 4096 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True
```



SRNN
```
# X Like small CW-VAEs
env WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="9" WANDB_NOTES="DMoL=10 MuLaw=16bit h=96 z=96" python experiments/experiment_srnn_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 --latent_size 96 --stack_frames 64 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True

# Like original paper
env WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="9" WANDB_NOTES="DMoL=10 MuLaw=16bit h=256 z=256" python experiments/experiment_srnn_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 256 --latent_size 256 --stack_frames 64 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True

```


VRNN
```
# X Like small CW-VAEs
env WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="4" WANDB_NOTES="DMoL=10 MuLaw=16bit h=96 z=96" python experiments/experiment_vrnn_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 96 --latent_size 96 --stack_frames 64 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True

# Like original paper
env WANDB_MODE=disabled CUDA_VISIBLE_DEVICES="9" WANDB_NOTES="DMoL=10 MuLaw=16bit h=256 z=256" python experiments/experiment_vrnn_audio.py --num_workers 4 --epochs 3000 --free_nats_start_value 2 --free_nats_steps 100000 --hidden_size 256 --latent_size 256 --stack_frames 64 --input_coding mu_law --num_bits 16 --num_mix 10 --save_checkpoints True


```