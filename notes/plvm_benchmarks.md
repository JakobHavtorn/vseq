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



| Dataset   | TIMIT (waveform) |     | Notes                                     |
| --------- | ---------------- | --- | ----------------------------------------- |
| **Model** | **BPD**          |     |                                           |
| VRNN      | 1.10             |     | DLM 10 mix                                |
| VRNN      | 0.96             |     | DLM 10 mix (1 free nats)                  |
| VRNN      | 1.03             |     | DLM 1 mix                                 |
| VRNN      | 0.998            |     | DLM 1 mix (1 free nats)                   |
| VRNN      | 1.699            |     | DLM 1 mix (1 free nats) (CW-VAE)          |
| VRNN      | 1.699            |     | DLM 1 mix (1 free nats) (no h->x)         |
| VRNN      | 1.07             |     | DL 1 mix (1 free nats)                    |
| VRNN      | running          |     | DLM 1 mix 20 frames (1 free nats)         |
|           |                  |     |                                           |
| SRNN      | running          |     | DL 1 mix (1 free nats)                    |
| SRNN      | running          |     | DL 1 mix (1 free nats) residual posterior |


