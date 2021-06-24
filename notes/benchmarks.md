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



**Stacked audio waveform encoding (200 if nothing else stated)**

Observations:
- Gradient clipping on VRNN caused faster initial training but also a worse minimum and then overfitting (ehtereal-oath-169 VS northern-glade-143)
- Samples are poor for stack size 20 run, probably due to temporal inconsistency caused by non-overlapping input transform (more pronounced for smaller stacks)
- Effects of residual posterior (SRNN) are not completely clear (seems detrimental?)


| Dataset   | TIMIT (waveform)                           |     | Notes                                                    |
| --------- | ------------------------------------------ | --- | -------------------------------------------------------- |
| **Model** | **BPD**                                    |     |                                                          |
| VRNN      | 0.75                                       |     | DLM 10 mix (no kl cost, autoencoding)                    |
| VRNN      | 1.10                                       |     | DLM 10 mix                                               |
| VRNN      | 0.954 (e600)                               |     | DLM 10 mix (1 free nats)                                 |
| VRNN      | **running**                                |     | DLM 10 mix (8 free nats. 30000)                          |
| VRNN      | 0.971 (e600)                               |     | DLM 10 mix (1 free nats, grad clip)                      |
| VRNN      | 1.03                                       |     | DLM 1 mix                                                |
| VRNN      | 0.998                                      |     | DLM 1 mix (1 free nats)                                  |
| VRNN      | 1.699                                      |     | DLM 1 mix (1 free nats) (no h->x, gen, no x->h, inf+gen) |
| VRNN      | 1.699                                      |     | DLM 1 mix (1 free nats) (no h->x, gen)                   |
| VRNN      | 1.17 (e600)                                |     | DLM 1 mix (1 free nats) (CW-VAE, no x->h, inf+gen)       |
| VRNN      | 1.07                                       |     | DL 1 mix (1 free nats)                                   |
| VRNN      | 0.61 (e109) **running** posterior collapse |     | DLM 1 mix 20 frames (1 free nats, 2000)                  |
| VRNN      | 0.796 (e45) **running**                    |     | DLM 1 mix 20 frames (1 free nats, 20000)                 |
|           |                                            |     |                                                          |
| SRNN      | 0.97 (e800)                                |     | DLM 10 mix (1 free nats. 30000)                          |
| SRNN      | 0.935 (e800)                               |     | DLM 10 mix (8 free nats. 30000)                          |
| SRNN      | 0.975 (e1000)                              |     | DL 1 mix (8 free nats. 30000)                            |
| SRNN      | 1.01 (e1000)                               |     | DL 1 mix (8 free nats, 50000)                            |
| SRNN      | 1.055 (e600)                               |     | DL 1 mix (8 free nats, 10000)                            |
|           |                                            |     |                                                          |
| CWVAE     | 0.87 (e1000)                               |     | 3L k=6 DLM 10 mix (no kl cost, autoencoding)             |
| CWVAE     | 1.03 (e250)                                |     | 3L k=2 DLM 10 mix                                        |
| CWVAE     | 1.035 (e250)                               |     | 3L k=2 DLM 10 mix (8 free nats)                          |
| CWVAE     | **running** (e1000) (overfit)              |     | 3L k=6 DLM 10 mix (1 free nats)                          |
| CWVAE     | **NaNs with clip? running**                |     | 3L k=2 DLM 10 mix (1 free nats, grad clip)               |
| CWVAE     | **running**                                |     | 1L DLM 1 mix (1 free nats) (VRNN with no x->h inf+gen)   | Sweetums 8 |
| CWVAE     | **running** 0.695 (e300)                   |     | 1L DLM 1 mix (no kl cost, autoencoding)                  | Sweetums 2 |
| CWVAE     | **running**                                |     | 3L DLM 10 mix (NoTmpAbs (200))                           | Scooter 8
|           |                                            |     |                                                          |
|           |                                            |     |                                                          |
|           |                                            |     |                                                          |
| Wavenet   | (ms)                                       |     | 200 stacked frames                                       |
| Wavenet   | (ms)                                       |     | 20 stacked frames                                        |
| Wavenet   | (ms)                                       |     | no stacked frames (pure waveform)                        |



