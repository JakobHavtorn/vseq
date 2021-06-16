# VRNN Albations

Ablation experiments to determine the mechanism that makes CW-VAE (and VRNN) work.

**Models:**
1. VRNN
2. Clockwork VAE
3. VTA
4. SRNN

**Ablation experiments:**
1. VRNN vanilla
2. VRNN no x -> h skip (inference+generative)
3. VRNN no h -> x skip (generative)
4. VRNN prior independent over timesteps (i.e. always standard Gaussian) (also examined in SRNN paper)

4. Clockwork VAE single layer (VRNN with modifications 2. and 3.)
5. Clockwork VAE single layer conditioned on "future" observation x_{t+1}

6. SRNN vanilla
7. SRNN without stochastic transition z_{t-1} -> z_t (no dependency between z's, equiv. to VRNN without z_t -> d_t connection)
8. SRNN filtering version (backward_recurrent is a per-timestep linear transform)
8. 
