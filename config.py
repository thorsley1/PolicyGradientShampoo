import jax
from dataclasses import dataclass, field
from typing import List

from jax.numpy import inf

@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Config:
    seed: int = 1                           # Base seed for all PRNGs used during training

    n_cycles: int = 2048                    # Number of rollout - update cycles
    n_envs: int = 2048                      # Number of parallel environments used during the rollout phase
    n_rollout: int = 64                     # Number of transitions collected per environment in a single rollout phase

    epochs_per_collection: int = 1         # Number of update epochs per collection
    updates_per_epoch: int = 4              # Number of gradient steps taken in each epoch in the update phase
    bwd_batch_size: int = 32768             # Number of individual transitions used in a single gradient step
    bwd_seq_len: int = 1                    # Sequence length of a single training example
    optimistic_reset_ratio: int = 16        # 

    n_layers: int = 2                       # Number of intermediate layers in the policy
    d_emb: int = 8268                       # Embedding dimension. Equal to the dimensionality of the observation space
    d_unemb: int = 43                       # Unembedding dimensions. Equal to the dimensionality of the action space
    d_model: int = 128                      # The model's primary hidden dimension. Dimensionality of the residual stream
    d_mlp: int = 384                        # 
    norm_eps: float = 1e-8                  #

    disc: float = 0.99                      # The discount factor used in the calculation of discounted rewards to go for pg reward weighting
    ent_coeff: float = 1e-2                 # Scaling of the entropy term added to the pg
    trust_reg_eps: float = 1000             # 

    base_lr: float = 5e-1                   # The base temperature of a single sampled gradient. Scaled by scheduler and per-layer scales
    beta_g: float = 1. - 1e-1               # The momentum coefficient of the policy gradients 
    beta_p: float = 1. - 1e-1               # The momentum coefficient of the intermediate Fisher statistics
    beta_w: float = 1. #- 1e-5               # The momentum coefficient of the parameters. I.e. for weight decay
    inv_freq: int = 1                       # The number of updates between recalculation of the inverse Fisher
    precond_init_eps: float = 1e-2          # An epislon value used for the initialisation of the preconditioner buffer
    precond_eps: float = 1e-4               # An epislon value used during the Fisher inversion

    embed_lr_scale: float = 1.              # Scaling constant for the embedding layer learning rate. For hyperparameter transfer
    unembed_lr_scale: float = 1.            # Scaling constant for the unembedding layer learning rate. For hyperparameter transfer
    inter_lr_scale: float = 1.              # Scaling constant for the intermediate layers learning rate. For hyperparameter transfer

    max_pg_norm: float = 2e2                # 
    max_lp_norm: float = 1e2                # 
    max_cond_norm: float = 1e1              # 




@jax.tree_util.register_dataclass
@dataclass
class Metrics:
    episode_returns: List = field(default_factory=list)
    episode_length: List = field(default_factory=list)
    act_ent: List = field(default_factory=list)
    importance_mean: List = field(default_factory=list)
    kl_est: List = field(default_factory=list)
    pg_norm: List = field(default_factory=list)
    lp_norm: List = field(default_factory=list)
    uncond_norm: List = field(default_factory=list)
    cond_grad_norm: List = field(default_factory=list)

    def add_collection_metrics(self, new_col_metrics):
        new_ep_returns, new_ep_lengths = new_col_metrics['returned_episode_returns']  * 100 / 226., new_col_metrics['returned_episode_lengths']
        return Metrics(
            episode_returns=self.episode_returns + [float(new_ep_returns)],
            episode_length=self.episode_length + [float(new_ep_lengths)],
            act_ent=self.act_ent,
            importance_mean=self.importance_mean,
            kl_est=self.kl_est,
            pg_norm=self.pg_norm,
            lp_norm=self.lp_norm,
            uncond_norm=self.uncond_norm,
            cond_grad_norm=self.cond_grad_norm,
        )

    def add_grad_and_optim_metrics(self, new_grad_metrics, new_optim_metrics):
        def flatten_first_two_dims(arr):
            return list(arr.reshape(-1, *arr.shape[2:]))

        act_ent, importance_mean, kl_est = new_grad_metrics
        pg_norm, lp_norm, uncond_norm, cond_grad_norm, = new_optim_metrics

        return Metrics(
            episode_returns=self.episode_returns,
            episode_length=self.episode_length,
            act_ent=self.act_ent + flatten_first_two_dims(act_ent),
            importance_mean=self.importance_mean + flatten_first_two_dims(importance_mean),
            kl_est=self.kl_est + flatten_first_two_dims(kl_est),
            pg_norm=self.pg_norm + flatten_first_two_dims(pg_norm),
            lp_norm=self.lp_norm + flatten_first_two_dims(lp_norm),
            uncond_norm=self.uncond_norm + flatten_first_two_dims(uncond_norm),
            cond_grad_norm=self.cond_grad_norm + flatten_first_two_dims(cond_grad_norm),
        )