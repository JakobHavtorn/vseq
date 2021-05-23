## Model

Denote the latent state at timestep $t$ and level $l$ by $s_t^l$ and the input frames by $x_t$.

- Active steps:
  $$
  \mathcal{T}_l \equiv \{ t\in[1,T] \, | \, t \text{ mod } k^{l-1} = 1\}
  $$

  - with $k=2$, we have $k^{l-1} = 1$ for $l=1$,  $2$ for $l=2$ and $4$ for $l=3$; hence
    - $\mathcal{T}_1 = \{ 1,2,3,\ldots \}$
    - $\mathcal{T}_2 = \{ 1,3,5,\ldots \}$
    - $\mathcal{T}_3 = \{ 1,5,9,\ldots \}$

- Copied states:
  $$
  s_t^l \equiv s^l_{\max_\tau \{ \tau\in\mathcal{T}_l \, | \, \tau \leq t \}}
  $$

  - with $k=2$
    - $s^1_1 = s^1_1, \quad s^1_2 = s^1_2, \ldots$
    - $s^2_{1:2} = s^2_2,\quad s^2_{3:4} = s^2_4, \ldots$ 
    - $s^2_{1:4} = s^2_4,\quad s^2_{5:8} = s^2_8, \ldots$ 

- Joint distribution

  - Two terms:
    - Reconstruction terms of inputs given the lowest level latent.
    - State transitions at all levels that are conditioned on the previous latent and the latent above.
    $$
    p(x_{1:T}, s_{1:T}^{1:L} = \left( \prod_{t=1}^T p(x_t|s_t^1) \right) \left( \prod_{l=1}^L \prod_{t\in\mathcal{T}_l} p(s_t^l|s_{t-1}^l, s_t^{l+1}) \right)
    $$

  - Implementation
    - Encoder: $e_t^l = \text{enc}(x_{t:t+k^{l-1}-1}))$
    - Posterior transition: $q_t^l = q(s_t^l|s_{t-1}^l, s_t^{l+1},e_t^l)$
    - Prior transition: $p_t^l = p(s_t^l | s_{t-1}^l, s_t^{l+1})$
    - Decoder: $p(x_t | s_t^1)$

- Inference

  - CW-VAE embeds the observed frames using a CNN.
  - Each active latent state at a level $l$ receives the image embeddings of its corresponding $k^{l-1}$ observation frames.
  - The diagonal Gaussian belief $q_t^l$ is then computed as a function of the input features, the posterior sample at the previous step, and the posterior sample above.
  - We reuse all weights of the generative model for inference except for the output layer that predicts the mean and variance.

- Generation:

  - The diagonal Gaussian prior $p_t^l $is computed by applying the transition function from the latent state at the previous timestep in the current level, as well as the state belief at the level above.
  - Finally, the posterior samples at the lowest level are decoded into images using a transposed CNN.

- Loss
  $$
  \max_{e,q,p} \sum_{t=1}^T \mathbb{E} \left[ \ln p(x_t | s_t^1) \right] - \sum_{l=1}^L\sum_{t\in\mathcal{T}_l} \mathbb{E}_{q_{t-1}^l q_t^{l+1}} \left[ D_{\text{KL}}(q_t^l || p_t^l) \right]
  $$

- The state variable
  - We split the state $s_t^l$ into stochastic ($z_t^l$) and deterministic ($h_t^l$) parts.
  - The deterministic state is computed using the top-down and temporal context, which then conditions the stochastic state at that level.
  - The stochastic variables follow diagonal Gaussians with predicted means and variances.
  - We use one GRU (Cho et al., 2014) per level to update the deterministic variable at every active step.

## Architecture details

- We use convolutional frame encoders and decoders, with architectures very similar to the DCGAN (Radford et al., 2016) discriminator and generator, respectively.
- To obtain the input embeddings $e_t^l$ at a particular level, $k_{l-1}$ input embeddings are pre-processed using a feed-forward network and then summed to obtain a single embedding.
- We do not use any skip connections between the encoder and decoder, which would bypass the latent states.

## Model size

- We keep the output size of the encoder at each level of CW-VAE as $|e_t^l| = 1024$, that of the stochastic states as $|p_t^l| = |q_t^l| = 20$, and that of the deterministic states as $|h_t^l| = 200$.
- All hidden layers inside the cell, both for prior and posterior transition, are set to $200$.
- We increase state sizes to | p t l | = | q t l | = 100, | h t l | = 800, and hidden layers inside the cell to 800 for the MineRL Navigate dataset.
- Parameters:
  - 12M small
  - 34M large

## Optimization

Optimizer: Adam (Kingma & Ba, 2014) with a learning rate of $3e−4$ and $\epsilon=1e−4$

Batch size: $100$ sequences with $100$ frames each.

Training time: 20 hours for 100 epochs with three level model on NVidia Titan `XP`



## Baselines

### RSSM

Joint distribution

$$
p(x_{1:T}, s_{1:T} = \prod_{t=1}^T p(x_t|s_t) p(s_t|s_{t-1})
$$

- Implementation
  - Encoder: $e_t = \text{enc}(x_{t:t}))$
  - Posterior transition: $q_t = q(s_t|s_{t-1},e_t)$
  - Prior transition: $p_t = p(s_t | s_{t-1})$
  - Decoder: $p(x_t | s_t)$

$\rightarrow$ corresponds to CW-VAE with $l=1$.

### NoTmpAbs

Joint distribution is the same as for CW-VAE but with $k=1$ (ablation baseline).

$\rightarrow$ corresponds to CW-VAE with $k=1$ and all layers ticking at the same (fastest) rate.


