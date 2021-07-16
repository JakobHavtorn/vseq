**Language modelling**

| Dataset   | PTB word |         | PTB char |         | MIDI   |         |     |     | Notes                            |
| --------- | -------- | ------- | -------- | ------- | ------ | ------- | --- | --- | -------------------------------- |
| **Model** | **LL**   | **BPD** | **LL**   | **BPD** | **LL** | **BPD** |     |     |                                  |
| LSTM LM   | -113     | 7.4     | -113     | 1.43    | -1146  | 6.33    |     |     |                                  |
| HMLSTM    | -111     | 7.3     | -107     | 1.39    |        |         |     |     |                                  |
| CWRNN     | -108.4   | 7.14    | -158     | 1.99    |        |         |     |     | Clock periods have little effect |
| Bowman    | -105.23  | 6.92    | -141     | 1.78    |        |         |     |     |                                  |
| VRNN      | -109.2   | 6.87    | -112.7   | 1,.41   |        |         |     |     |                                  |
|           |          |         |          |         |        |         |     |     |                                  |
|           |          |         |          |         |        |         |     |     |                                  |
|           |          |         |          |         |        |         |     |     |                                  |
|           |          |         |          |         |        |         |     |     |                                  |
|           |          |         |          |         |        |         |     |     |                                  |
|           |          |         |          |         |        |         |     |     |                                  |



**Stacked audio waveform encoding**
(200 if nothing else stated)

Observations:
- Gradient clipping on VRNN caused faster initial training but also a worse minimum and then overfitting (ehtereal-oath-169 VS northern-glade-143)
- Samples are poor for stack size 20 run, probably due to temporal inconsistency caused by non-overlapping input transform (more pronounced for smaller stacks)
- Reconstructions are much less noisy for stack size 20 runs.
- Effects of residual posterior (SRNN) are not completely clear (seems detrimental?)


| Dataset   | TIMIT (waveform)                 |     | Notes                                                    |
| --------- | -------------------------------- | --- | -------------------------------------------------------- |
| **Model** | **BPD**                          |     |                                                          |
| VRNN      | 0.75                             |     | DLM 10 mix (no kl cost, autoencoding)                    |
| VRNN      | 1.10                             |     | DLM 10 mix                                               |
| VRNN      | 0.954 (e600)                     |     | DLM 10 mix (1 free nats)                                 |
| VRNN      | 0.974 (e800)                     |     | DLM 10 mix (8 free nats. 30000)                          |
| VRNN      | 0.971 (e600)                     |     | DLM 10 mix (1 free nats, grad clip)                      |
| VRNN      | 1.03                             |     | DLM 1 mix                                                |
| VRNN      | 0.998                            |     | DLM 1 mix (1 free nats)                                  |
| VRNN      | 1.699                            |     | DLM 1 mix (1 free nats) (no h->x, gen, no x->h, inf+gen) |
| VRNN      | 1.699                            |     | DLM 1 mix (1 free nats) (no h->x, gen)                   |
| VRNN      | 1.17 (e600)                      |     | DLM 1 mix (1 free nats) (CW-VAE, no x->h, inf+gen)       |
| VRNN      | 1.07                             |     | DL 1 mix (1 free nats)                                   |
| VRNN      | 0.5394 (e176) posterior collapse |     | DLM 1 mix 20 frames (1 free nats, 2000)                  |
| VRNN      | 0.3896 (e189)                    |     | DLM 1 mix 20 frames (1 free nats, 20000)                 | fresh-energy                      |
|           |                                  |     |                                                          |
| SRNN      | 0.97 (e800)                      |     | DLM 10 mix (1 free nats. 30000)                          |
| SRNN      | 0.935 (e800)                     |     | DLM 10 mix (8 free nats. 30000)                          |
| SRNN      | 0.975 (e1000)                    |     | DL 1 mix (8 free nats. 30000)                            |
| SRNN      | 1.01 (e1000)                     |     | DL 1 mix (8 free nats, 50000)                            |
| SRNN      | 1.055 (e600)                     |     | DL 1 mix (8 free nats, 10000)                            |
|           |                                  |     |                                                          |
| CWVAE     | 0.87 (e1000)                     |     | 3L k=6 DLM 10 mix (no kl cost, autoencoding)             |
| CWVAE     | 0.43 (e750)                      |     | 1L DLM 1 mix (no kl cost, autoencoding)                  |
| CWVAE     | 1.14 e(750)                      |     | 1L DLM 1 mix (1 free nats) (VRNN with no x->h inf+gen)   | Similar to VRNN model as expected |
| CWVAE     | 1.03 (e250)                      |     | 3L k=2 DLM 10 mix                                        |
| CWVAE     | 1.035 (e250)                     |     | 3L k=2 DLM 10 mix (8 free nats)                          |
| CWVAE     | 1.125 (e400) (overfit slightly)  |     | 3L k=6 DLM 10 mix (1 free nats)                          |
| CWVAE     | **NaNs with clip? running**      |     | 3L k=2 DLM 10 mix (1 free nats, grad clip)               |
| CWVAE     | 1.08 (e635)                      |     | 3L DLM 10 mix (NoTmpAbs (200))                           |
| CWVAE     | 4.54 (e191, overfit)             |     | 3L k=4 DLM 10 mix (MuLaw encoded 16bit in 8bit out)      | whole-river Interesting samples   |
|           |                                  |     |                                                          |
|           |                                  |     |                                                          |
|           |                                  |     |                                                          |
| Wavenet   | (ms)                             |     | 200 stacked frames                                       |
| Wavenet   | (ms)                             |     | 20 stacked frames                                        |
| Wavenet   | (ms)                             |     | no stacked frames (pure waveform)                        |



**CPC conv encoder on raw PCM (or MuLaw PCM)**
- Unfreezing pretrained encoder parameters improves likelihood
- Using a convolutional decoder instead of dense improves likelihood
- MuLaw encoding makes training more stable and sample quality higher
- 16bit VS 8bit output: 
- 16bit VS 8bit input:
  - 8bit quantized input (instead of 16bit) makes no difference when output is 8bit

| Dataset | TIMIT (waveform) |     | Notes                                                                                                           | Name                            |
| ------- | ---------------- | --- | --------------------------------------------------------------------------------------------------------------- | ------------------------------- |
| CWVAE   | 1.308 (e750)     |     | 1L DLM 10 mix PretrainedCPCEncoder, frozen (160) (frames)                                                       | zany-durian                     |
| CWVAE   | 1.144 (e450)     |     | 1L DLM 10 mix PretrainedCPCEncoder, frozen, conv decoder (160) (frames)                                         | easy-meadow                     |
| CWVAE   | 0.831 (589)      |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (frames)                                     | likely-firefly                  |
| CWVAE   | 4.175 (531)      |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (MuLaw 16bit in 8bit out)                    | warm-breeze                     |
| CWVAE   | 4.131 (539)      |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (MuLaw 16bit in 8bit out) clip-grad-value 1  | rosy-deluge                     | ~same loss jumps                            |
| CWVAE   | 4.16 (e750)      |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (MuLaw 16bit in 8bit out) clip-grad-norm 1   | restful-leaf-40                 |
| CWVAE   | 4.16 (e750)      |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (MuLaw 16bit in 8bit out) clip-grad-norm 0.1 | glowing-disco-43                |
| CWVAE   | 4.365 (e750)     |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, linear-upsample-conv decoder (160) (MuLaw 16bit in 8bit out)    | genial-cosmos-39 + lucky-fog-49 |
| CWVAE   |                  |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, nearest-upsample-conv decoder (160) (MuLaw 16bit in 8bit out)   | crimson-silence-50              |
| CWVAE   | 8.56 (e750)      |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (frames 16bit in 16bit out)                  | clean-morning-45                |
| CWVAE   | 12.2 (e750)      |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (MuLaw 16bit in 16bit out)                   | wise-sunset-46                  | still converging.. very interesting samples |
| CWVAE   | 13.02 (e750)     |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (MuLaw 16bit divisor in 16bit out)           | silver-smoke-59                 | MuLaw with 16bit divisor                    |
| CWVAE   | 12.08 (e750)     |     | 1L DLaplaceM 10 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (MuLaw 16bit in 16bit out)             | sage-deluge-58                  | ...                                         |
| CWVAE   | 12.239 (e535)    |     | 1L DLaplaceM 1 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (MuLaw 16bit in 16bit out)              | legendary-cloud-62              | ...                                         |
| CWVAE   | 4.172 (e750)     |     | 1L DLM 10 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (MuLaw 8bit in 8bit out)                     | genial-snow + dark-grass-48     | 8bit quantized input (instead of 16bit)     |
| CWVAE   | 4.328 (e750)     |     | 1L DLM 1 mix PretrainedCPCEncoder, not frozen, conv decoder (160) (MuLaw 8bit in 8bit out)                      | apricot-monkey-47               | more noisy than 10 mixture components       |



**Convolution encoder/decoder**

| Dataset | TIMIT (waveform) |     | Notes                                                                                                                 | Name                    |
| ------- | ---------------- | --- | --------------------------------------------------------------------------------------------------------------------- | ----------------------- |
| CWVAE   | crashed          |     | 1L DLM 10 mix Conv1dCoder (64) (MuLaw 16bit) (zero kernel overlap)                                                    | royal-fog-73            |
| CWVAE   | 11.973 (e750)    |     | 1L DLM 10 mix Conv1dCoder (64) (MuLaw 16bit)                                                                          | misunderstood-breeze-70 |
| CWVAE   | 12.177 (e750)    |     | 2L DLM 10 mix Conv1dCoder (64, 512) (MuLaw 16bit)                                                                     | wise-blaze-71           |
| CWVAE   | 12.141 (e750)    |     | 3L DLM 10 mix Conv1dCoder (64, 512, 4096) (MuLaw 16bit)                                                               | volcanic-microwave-72   | z3 dead |
| CWVAE   | 12.031 (e750)    |     | 2L DLM 10 mix Conv1dCoder (64, 512) (MuLaw 16bit) (ContextDecoder)                                                    | major-violet-7          |
| CWVAE   | 12.179 (e750)    |     | 3L DLM 10 mix Conv1dCoder (64, 512, 4096) (MuLaw 16bit) (ContextDecoder)                                              | exalted-night-7         | z3 dead |
| CWVAE   | **running**      |     | 1L DLM 10 mix Conv1dCoder (64) (MuLaw 16bit) (e5000, LengthSampler)                                                   | lucky-leaf-96           |
| CWVAE   | **running**      |     | 2L DLM 10 mix Conv1dCoder (64, 512) (MuLaw 16bit) (ContextDecoder) (e5000, LengthSampler)                             | grateful-serenity-95    |
| CWVAE   | **running**      |     | 3L DLM 10 mix Conv1dCoder (64, 512, 4096) (MuLaw 16bit) (ContextDecoder) (e5000, length LengthSampler exalted-night-7 | rich-elevator-97        |
| CWVAE   | **running**      |     | 1L DLM 10 mix Conv1dCoder (64) (MuLaw 16bit divisor) (e5000, LengthSampler)                                           | pious-snow-103          |
 


```
env WANDB_MODE=disabled WANDB_NOTES="1L DLM 10 mix Conv1dCoder (64) (MuLaw 16bit) (zero kernel overlap)" CUDA_VISIBLE_DEVICES='8' python experiments/experiment_cwvae_audio.py --num_workers 4 --batch_size 16 --epochs 750 --free_nats_start_value 4 --free_nats_steps 80000 --hidden_size 64 --latent_size 64 --time_factors 64 --input_coding mu_law --num_bits 16

env WANDB_MODE=disabled WANDB_NOTES="1L DLM 10 mix Conv1dCoder (64) (MuLaw 16bit)" CUDA_VISIBLE_DEVICES='8' python experiments/experiment_cwvae_audio.py --num_workers 4 --batch_size 16 --epochs 750 --free_nats_start_value 4 --free_nats_steps 80000 --hidden_size 64 --latent_size 64 --time_factors 64 --input_coding mu_law --num_bits 16

env WANDB_MODE=disabled WANDB_NOTES="2L DLM 10 mix Conv1dCoder (64, 512) (MuLaw 16bit)" CUDA_VISIBLE_DEVICES='9' python experiments/experiment_cwvae_audio.py --num_workers 4 --batch_size 16 --epochs 750 --free_nats_start_value 4 --free_nats_steps 80000 --hidden_size 64 192 --latent_size 64 128 --time_factors 64 512 --input_coding mu_law --num_bits 16

env WANDB_MODE=disabled WANDB_NOTES="3L DLM 10 mix Conv1dCoder (64, 512, 4096) (MuLaw 16bit)" CUDA_VISIBLE_DEVICES='1' python experiments/experiment_cwvae_audio.py --num_workers 4 --batch_size 16 --epochs 750 --free_nats_start_value 4 --free_nats_steps 80000 --hidden_size 64 192 448 --latent_size 64 128 192 --time_factors 64 512 4096 --input_coding mu_law --num_bits 16
```



**TasNet Encoder**

| Dataset | TIMIT (waveform) |     | Notes                                                               | Name               |
| ------- | ---------------- | --- | ------------------------------------------------------------------- | ------------------ |
| CWVAE   | **running**      |     | 1L DML 10 mix TasNet (64) (MuLaw 16bit in 16bit out) 32h 64z        | generous-forest-54 |
| CWVAE   | **running**      |     | 2L DML 10 mix TasNet (64, 4096) (MuLaw 16bit in 16bit out) 32h 64z  | laced-planet-57    |
| CWVAE   | **running**      |     | 1L DML 10 mix TasNet (64) (MuLaw 16bit in 16bit out) 128h 128z      | trim-glade-102     | DDP |
| CWVAE   | **running**      |     | 2L DML 10 mix TasNet (64, 512) (MuLaw 16bit in 16bit out) 128h 128z | solar-paper-101    | DDP |

```
 env WANDB_MODE=disabled WANDB_NOTES="1L DML 10 mix TasNet (64, 4096, 65535) (MuLaw 16bit in 16bit out)" CUDA_VISIBLE_DEVICES='6' python experiments/experiment_cwvae_audio.py --num_workers 4 --batch_size 2 --epochs 750 --free_nats_start_value 4 --free_nats_steps 80000 --hidden_size 32 --latent_size 64 --time_factors 64 --input_coding mu_law
 
env WANDB_MODE=disabled WANDB_NOTES="2L DML 10 mix TasNet (64, 4096, 65535) (MuLaw 16bit in 16bit out)" CUDA_VISIBLE_DEVICES='5' python experiments/experiment_cwvae_audio.py --num_workers 4 --batch_size 2 --epochs 750 --free_nats_start_value 4 --free_nats_steps 80000 --hidden_size 32 --latent_size 64 --time_factors 64 4096 --input_coding mu_law

env WANDB_MODE=disabled WANDB_NOTES="1L DLM 10 mix PretrainedCPCEncoder, not frozen, upsample-conv decoder (160) (MuLaw 16bit in 8bit out)" CUDA_VISIBLE_DEVICES='6' python experiments/experiment_cwvae_audio.py --num_workers 4 --batch_size 2 --epochs 750 --free_nats_start_value 4 --free_nats_steps 80000 --hidden_size 32 --latent_size 64 --time_factors 64 4096 65536 --input_coding mu_law
```
