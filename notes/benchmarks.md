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



| Dataset   | TIMIT (waveform)    |     | Notes                                                    |
| --------- | ------------------- | --- | -------------------------------------------------------- |
| **Model** | **BPD**             |     |                                                          |
| VRNN      | 0.75                |     | DLM 10 mix (no kl cost, autoencoding)                    |
| VRNN      | 1.10                |     | DLM 10 mix                                               |
| VRNN      | 0.96                |     | DLM 10 mix (1 free nats)                                 |
| VRNN      | 1.03                |     | DLM 1 mix                                                |
| VRNN      | 0.998               |     | DLM 1 mix (1 free nats)                                  |
| VRNN      | 1.699               |     | DLM 1 mix (1 free nats) (no h->x, gen, no x->h, inf+gen) |
| VRNN      | 1.699               |     | DLM 1 mix (1 free nats) (no h->x, gen)                   |
| VRNN      | 1.20 (e406) running |     | DLM 1 mix (1 free nats) (CW-VAE, no x->h, inf+gen)       |
| VRNN      | 1.07                |     | DL 1 mix (1 free nats)                                   |
| VRNN      | running             |     | DLM 1 mix 20 frames (1 free nats, 2000)                  |
| VRNN      | running             |     | DLM 1 mix 20 frames (1 free nats, 20000)                 |
|           |                     |     |                                                          |
| SRNN      | running             |     | DLM 10 mix (1 free nats. 30000)                          |
| SRNN      | running             |     | DLM 10 mix (8 free nats. 30000)                          |
| SRNN      | 0.975 (e1000)       |     | DL 1 mix (8 free nats. 30000)                            |
| SRNN      | 1.01 (e1000)        |     | DL 1 mix (8 free nats, 50000)                            |
| SRNN      | 1.055 (e600)        |     | DL 1 mix (8 free nats, 10000)                            |
|           |                     |     |                                                          |
| CWVAE     | running             |     | DLM 10 mix (no kl cost, autoencoding)                    |
| CWVAE     | running             |     | DLM 10 mix                                               |
| CWVAE     | running             |     | DLM 10 mix (1 free nats)                                 |
|           |                     |     |                                                          |
|           |                     |     |                                                          |
|           |                     |     |                                                          |
| Wavenet   | (ms)                |     | 200 stacked frames                                       |
| Wavenet   | (ms)                |     | 20 stacked frames                                        |
| Wavenet   | (ms)                |     | no stacked frames (pure waveform)                        |


