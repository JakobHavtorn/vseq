# VRNN Albations

Ablation experiments to determine the mechanism that makes CW-VAE (and VRNN) work.

**Baseline**
- LSTM
- Clockwork RNN
- HMRNN
- 

**Models:**
- VRNN
- SRNN
- VTA
- Clockwork VAE

**Ablation experiments:**
x VRNN vanilla (6.93 bpd)
x VRNN no x -> h skip (inference+generative) (8.4 bpd)
x VRNN no h -> x skip (generative) (7.5 bpd)
x VRNN no x -> h skip and no h -> x skip (8.4 bpd)
- VRNN prior independent over timesteps (i.e. always standard Gaussian) (also examined in SRNN paper)

- SRNN vanilla
- SRNN without stochastic transition z_{t-1} -> z_t (no dependency between z's, equiv. to VRNN without z_t -> d_t connection)
- SRNN filtering version (backward_recurrent is a per-timestep linear transform)

- VTA?

- Clockwork VAE single layer (VRNN with modifications 2. and 3.)
- Clockwork VAE single layer conditioned on "future" observation x_{t+1}
