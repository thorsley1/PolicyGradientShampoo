import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from craftax.craftax_env import make_craftax_env_from_name
from dataclasses import dataclass
from functools import partial
from string import ascii_lowercase, ascii_uppercase
from tqdm import tqdm

from config import Config, Metrics
from model import Parameters, init_params, model
from wrappers import OptimisticResetVecEnvWrapper, LogWrapper


## Initialise the environment

@partial(jax.jit, static_argnums=(2,3,4,))
def collect_single_transition(carry, step, env_step, env_params, cfg):
    """Environment step to be scanned over."""

    os_pair, model_params = carry
    obs, state, key = os_pair
    log_probs = model(model_params, obs, cfg)

    next_key, act_key, step_key = jr.split(key, 3)

    acts = jr.categorical(act_key, log_probs, axis=-1)
    act_log_probs = jnp.take_along_axis(log_probs, acts[..., None], axis=-1).squeeze(-1)

    next_obs, next_state, rwds, dones, infos = env_step(
      step_key, state, acts, env_params
      )

    next_os_pair = (next_obs, next_state, next_key)
    return (next_os_pair, model_params), [obs.astype(jnp.bfloat16), acts, rwds.astype(jnp.bfloat16), dones, act_log_probs.astype(jnp.bfloat16), infos]


def init_cfg_and_envs():
    """Initialises the config and environment."""

    unwrapped_single_env = make_craftax_env_from_name("Craftax-Symbolic-v1", False)
    env_params = unwrapped_single_env.default_params

    cfg = Config(
        d_emb=unwrapped_single_env.observation_space(env_params).shape[0],
        d_unemb=unwrapped_single_env.action_space(env_params).n,
        )

    log_single_env = LogWrapper(unwrapped_single_env)
    single_env = OptimisticResetVecEnvWrapper(log_single_env, num_envs=cfg.n_envs, reset_ratio=min(cfg.optimistic_reset_ratio, cfg.n_envs),)
    
    key_init = jr.PRNGKey(cfg.seed)
    obs_init, state_init = single_env.reset(key_init, env_params)
    env_step = single_env.step

    single_trans_scan = jax.tree_util.Partial(collect_single_transition, env_step=env_step, env_params=env_params, cfg=cfg)
    return cfg, (single_trans_scan, (obs_init, state_init, key_init))


## Define Grads object

@jax.tree_util.register_dataclass
@dataclass
class Grads:
    pg_grads: Parameters
    lp_grads: Parameters


## Initialise the optimiser state

@jax.tree_util.register_dataclass
@dataclass
class State:
    step: int
    params: Parameters
    grad_buffer: Parameters
    lr_scales: Parameters
    precond_buffer: Parameters
    fisher_stats: Parameters


def init_pc_buf(params, cfg):
    """Initialises the optimiser state preconditioning matrix buffer."""

    pc_leaves = []
    for leaf in params.list_flatten():
        shape = leaf[0].shape
        layers, rest = shape[0], shape[1:]

        precond_buf = []
        for dim in rest:
            precond_buf.append(cfg.precond_init_eps * jnp.eye(dim)[None, :, :].repeat(layers, axis=0))

        pc_leaves.append(precond_buf)

    return Parameters.list_unflatten(pc_leaves)


def init_state(cfg: Config):
    """Initialises the optimiser state."""

    params = init_params(cfg)

    grad_buffer = jax.tree.map(lambda x: 0.*x, params)

    lr_scales = Parameters(
        [jnp.full((1), cfg.embed_lr_scale), ],
        [jnp.full((1), cfg.unembed_lr_scale), ],
        [jnp.full((1), cfg.inter_lr_scale), ],
        tuple([jnp.full((1), cfg.inter_lr_scale), ] for _ in params.layer_params)
    )

    precond_buffer = init_pc_buf(params, cfg)
    
    return State(
        step=0,
        params=params,
        grad_buffer=grad_buffer,
        lr_scales=lr_scales,
        precond_buffer=precond_buffer,
        fisher_stats=precond_buffer
    )


## Overall setup

def setup_training():
    """Sets up everything needed for training."""
    cfg, env = init_cfg_and_envs()
    state = init_state(cfg)
    metrics = Metrics()
    return state, metrics, env, cfg




## Environment rollout

@partial(jax.jit, static_argnames=('cfg',))
def process_transitions(col_transitions, cfg, step):
    """Coverts the collected transitions into the appropriate form for the backward pass and gets the training metrics."""

    def rtg_scan_fn(carry, x):
        r_t, done_t = x
        current_return = r_t + cfg.disc * carry * (1.0 - done_t.astype(jnp.bfloat16))
        return current_return, current_return

    obs, acts, rwds, dones, old_act_log_probs, infos = col_transitions
    _, rwd_to_go = jax.lax.scan(rtg_scan_fn, jnp.zeros(cfg.n_envs), (rwds, dones), reverse=True)
    rwd_weightings = rwd_to_go - ((cfg.n_envs * rwd_to_go.mean() - rwd_to_go.mean(axis=0)) / (cfg.n_envs - 1.))

    key = jr.PRNGKey(step)
    permutation = jr.permutation(key, cfg.updates_per_epoch * cfg.bwd_batch_size)

    ordered_transitions = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), [obs, acts, old_act_log_probs, rwd_weightings])
    flat_transitions = jax.tree.map(lambda x: x.reshape((cfg.updates_per_epoch * cfg.bwd_batch_size, cfg.bwd_seq_len, *x.shape[2:])), ordered_transitions)
    perm_transitions = jax.tree.map(lambda x: jnp.take(x, permutation, axis=0), flat_transitions)
    update_transitions = jax.tree.map(lambda x: x.reshape((cfg.updates_per_epoch, cfg.bwd_batch_size, cfg.bwd_seq_len, *x.shape[2:])), perm_transitions)

    new_metrics = jax.tree.map(lambda x: (x * infos["returned_episode"]).sum() / infos["returned_episode"].sum(), infos,)
    return update_transitions, new_metrics


def get_transitions(env, model_state, metrics, cfg):
    """Collects a batch of transitions."""

    step_scan, os_pair_init = env

    carry, collection_transitions = jax.lax.scan(step_scan, (os_pair_init, model_state.params), length=cfg.n_rollout)
    transitions, new_metrics = process_transitions(collection_transitions, cfg, model_state.step)

    os_pair_final, _ = carry
    env = (step_scan, os_pair_final)
    metrics = metrics.add_collection_metrics(new_metrics)
    return transitions, env, metrics


## Loss calculation and backward pass

def losses_and_metrics(params, transition, cfg):
    """Calculates the appriate losses to get policy / log prob gradients and optimisation metrics."""

    obs, acts, old_act_log_probs, rwd_weighting = transition
    log_probs = model(params, obs, cfg)
    new_act_log_probs = jnp.take_along_axis(log_probs, acts[..., None], axis=-1).squeeze(-1)

    act_log_prob_diff = new_act_log_probs - old_act_log_probs
    kl_measure = act_log_prob_diff - (1 - jnp.exp(-act_log_prob_diff))
    detach_cond = kl_measure > cfg.trust_reg_eps
    sqrt_importance = jax.lax.stop_gradient(jnp.exp(act_log_prob_diff / 2))
    importance = jax.lax.stop_gradient(jnp.exp(act_log_prob_diff))

    pg_weighting = jax.lax.stop_gradient((importance * rwd_weighting) - (cfg.ent_coeff * new_act_log_probs))
    tr_log_probs = jnp.where(detach_cond, jax.lax.stop_gradient(new_act_log_probs), new_act_log_probs)
    pg_loss, lp_loss = jnp.mean(pg_weighting * tr_log_probs), jnp.mean(importance * tr_log_probs)

    act_ent = -(jnp.exp(log_probs) * log_probs).sum(-1).mean() / jnp.log(log_probs.shape[-1])
    importance_mean = importance.mean()
    kl_est = kl_measure.mean()
    new_metrics = (act_ent, importance_mean, kl_est)

    return jnp.asarray([pg_loss, lp_loss]), (new_metrics)


def clip_grad(grad, max_grad_norm):
    """Clips the gradients before they enter the optimiser."""

    clip_coef = jnp.minimum(1.0, max_grad_norm / (grad.norm() + 1e-6))
    return jax.tree.map(lambda arr: clip_coef * arr, grad)


def separate_and_clip_grads(joint_grads, cfg):
    """Separates the gradients from the jacobian and clips them."""

    pg_grads = jax.tree.map(lambda arr: arr[0], joint_grads)
    lp_grads = jax.tree.map(lambda arr: arr[1], joint_grads)
    clipped_pg = clip_grad(pg_grads, cfg.max_pg_norm)
    clipped_lp = clip_grad(lp_grads, cfg.max_lp_norm)
    return clipped_pg, clipped_lp


def get_grads(state, transitions, cfg):
    """Takes in a set of transitions and returns the policy / log prob gradients for the optimiser."""

    joint_grads, new_metrics = jax.jacrev(losses_and_metrics, has_aux=True)(state.params, transitions, cfg)
    pg_grads, lp_grads = separate_and_clip_grads(joint_grads, cfg)
    return Grads(pg_grads, lp_grads), new_metrics


## Stepping the optimiser

def list_map(f, xs):
    """Similar to jax.tree.map but for list_unflatten. Needed for the preconditioner lists."""

    return Parameters.list_unflatten([f(*leaf) for leaf in zip(*[x.list_flatten() for x in xs])])


def scheduler(step, cfg):
    """Learning rate schedule. TODO: Add more optionality here."""

    total_steps = cfg.epochs_per_collection * cfg.updates_per_epoch * cfg.n_cycles
    return step / total_steps #jnp.minimum(1.0, step / (0.1 * total_steps))


def update_precond_buf(precond_buf, grad, beta_p, batch_size):
    """Updates the preconditioner buffers with the new log prob grads. Standard Shampoo optimiser method."""

    def update_single_layer(precond_buf, grad, beta_p=beta_p, batch_size=batch_size):
        grad = grad[0]
        for dim_idx in range(grad.ndim):
            grad = jnp.swapaxes(grad, 0, dim_idx)
            current_shape = grad.shape
            grad = jnp.reshape(grad, (current_shape[0], -1))
            precond_update = batch_size * grad @ grad.T
            grad = jnp.reshape(grad, current_shape)

            new_dim_pc = beta_p * precond_buf[dim_idx] + (1 - beta_p) * precond_update
            precond_buf = precond_buf[:dim_idx] + [new_dim_pc,] + precond_buf[dim_idx+1:]

        return precond_buf

    return jax.block_until_ready(list_map(jax.vmap(update_single_layer), (precond_buf, grad)))


def update_inv_fisher(fisher_stats, precond_buffer, step, inv_freq, eps):
    """Periodically updates the preconditioner itself by taking the (-1/rank)th power of the buffer."""

    def get_new_inv_fisher(fisher_stats, precond_buf, eps=eps):
        def get_partial_inv(fisher_stats, precond_buf, eps=eps):
            rank = len(precond_buf)
            for axis, single_axis_pc_buf in enumerate(precond_buf):
                n_l, dim = single_axis_pc_buf.shape[0], single_axis_pc_buf.shape[1]
                if dim <= 2048:
                    eigenvalues, eigenbasis = jax.vmap(jnp.linalg.eigh, in_axes=0)(single_axis_pc_buf)
                    safe_eigenvalues = jnp.maximum(eigenvalues, 0.)
                    inv_eigenvalues = jnp.reciprocal(jnp.power(safe_eigenvalues, 1/(rank)) + eps)
                    inv_fisher = jnp.einsum('nij,nj,nkj->nik', eigenbasis, inv_eigenvalues, eigenbasis)
                    safe_inv_fisher = jnp.where(jnp.isfinite(inv_fisher), inv_fisher, 0.)
                else:
                    safe_inv_fisher = jnp.eye(dim)[None, :, :].repeat(n_l, axis=0)
                fisher_stats = fisher_stats[:axis] + [safe_inv_fisher,] + fisher_stats[axis+1:]
            return fisher_stats

        return list_map(get_partial_inv, (fisher_stats, precond_buf))
    
    return jax.lax.cond((step-1) % inv_freq == 0, get_new_inv_fisher, lambda fs, _ : fs, fisher_stats, precond_buffer)


def update_gradient_buffer(grad_buffer, pg_grads, beta_g):
    """Updates the policy gradient buffer."""

    return jax.tree.map(lambda grad_buf, grad: beta_g * grad_buf + (1 - beta_g) * grad, grad_buffer, pg_grads)


def get_unconditioned_gradients(grad_buffer, step, beta_g):
    """Bias corrects the policy gradient buffer."""

    return jax.tree.map(lambda grad_buf: grad_buf / (1 - beta_g ** step), grad_buffer)


def condition_gradients(uncond_grads, fisher_stats):
    """Applies the preconditioner to the policy gradient."""

    def rot(tensor, cob, into=True):
        """Applies a list of change of basis matricies to each dimension of a particular tensor."""

        rank = tensor[0].ndim
        batch_ind = ascii_lowercase[0]
        old_inds, new_inds = ascii_lowercase[1:rank], ascii_uppercase[1:rank]
        contraction_terms = [batch_ind + new_idx + old_idx if into else batch_ind + old_idx + new_idx for new_idx, old_idx in zip(new_inds, old_inds)]
        einsum_string = f"{','.join(contraction_terms)},{batch_ind + old_inds}->{batch_ind + new_inds}"
        einsum_args = [*cob, *tensor]
        return [jnp.einsum(einsum_string, *einsum_args, optimize=True), ]

    return list_map(rot, (uncond_grads, fisher_stats))


def update_params(params, grads, lr, lr_scales, beta_w, max_grad_norm):
    """Clips conditioned grads and updates params with appropriate lr_scales and weight decay."""

    clip_coef = jnp.minimum(1.0, max_grad_norm / (grads.norm() + 1e-6))
    return jax.tree.map(lambda param, grad, scales: (beta_w * param + lr * scales * clip_coef * grad), params, grads, lr_scales)


def step_optim(state, grads, cfg):
    """Performs a single step of the optimiser given a set of gradients."""

    step = state.step + 1

    precond_buffer = update_precond_buf(state.precond_buffer, grads.lp_grads, cfg.beta_p, cfg.bwd_batch_size)
    fisher_stats = update_inv_fisher(state.fisher_stats, precond_buffer, step, cfg.inv_freq, cfg.precond_eps)

    gradient_buffer = update_gradient_buffer(state.grad_buffer, grads.pg_grads, cfg.beta_g)
    uncond_grads = get_unconditioned_gradients(gradient_buffer, step, cfg.beta_g)
    cond_grads = condition_gradients(uncond_grads, state.fisher_stats)
    params = update_params(state.params, cond_grads, cfg.base_lr * scheduler(step, cfg), state.lr_scales, cfg.beta_w, cfg.max_cond_norm)

    optim_metrics = (grads.pg_grads.norm(), grads.lp_grads.norm(), uncond_grads.norm(), cond_grads.norm())

    return State(
        step = step,
        params = params,
        lr_scales = state.lr_scales,
        grad_buffer = gradient_buffer,
        precond_buffer = precond_buffer,
        fisher_stats = fisher_stats
    ), optim_metrics


def get_single_state_update(in_state, transitions, cfg):
    """Takes gradients and steps the optimiser given a set of transitions."""

    grads, grad_metrics = get_grads(in_state, transitions, cfg)
    out_state, optim_metrics = step_optim(in_state, grads, cfg)
    return out_state, (grad_metrics, optim_metrics)


def get_single_epoch_update(in_st_pair, scan_step, cfg):
    """Gets updates across the set of transition batches."""

    in_state, transitions = in_st_pair
    scan_body = partial(get_single_state_update, cfg=cfg)
    out_state, new_metrics = jax.lax.scan(scan_body, in_state, transitions)
    return (out_state, transitions), new_metrics


@partial(jax.jit, static_argnames=('cfg',))
def get_all_epoch_updates(in_state, transitions, cfg):
    """Gets updates across the set of epochs."""

    in_st_pair = (in_state, transitions)
    scan_body = partial(get_single_epoch_update, cfg=cfg)
    out_st_pair, new_metrics = jax.lax.scan(scan_body, in_st_pair, length=cfg.epochs_per_collection)
    out_state, _ = out_st_pair
    return out_state, new_metrics


def update_state(in_state, transitions, metrics, cfg):
    """Gets the overall update per collection."""

    out_state, new_metrics = get_all_epoch_updates(in_state, transitions, cfg)
    metrics = metrics.add_grad_and_optim_metrics(*new_metrics)
    return out_state, metrics


## Training outer loop

def single_cycle(state, env, metrics, cfg):
    """Performs a single cycle of collections and updates."""

    transitions, env, metrics = get_transitions(env, state, metrics, cfg)
    state, metrics = update_state(state, transitions, metrics, cfg)
    return state, env, metrics


def train_policy():
    """Trains a model policy"""

    state, metrics, env, cfg = setup_training()
    for _ in tqdm(range(cfg.n_cycles)):
        state, env, metrics = single_cycle(state, env, metrics, cfg)

    return metrics


## Plotting functionality

def plot_metrics(metrics):
    metric_names = [
        ("episode_returns", "Average Return (% of max)", "Returns over Updates"),
        ("episode_length", "Average Episode Length", "Episode Lengths over Updates"),
        ("act_ent", "Action Entropy / Max Entropy", "Relative Action Entropy over Updates"),
        ("importance_mean", "Importance Mean", "Importance Mean over Updates"),
        ("kl_est", "KL Estimate", "KL Estimate over Updates"),
        ("pg_norm", "Policy Gradient Norm", "Policy Gradient Norm over Updates"),
        ("lp_norm", "Likelihood Gradient Norm", "Likelihood Gradient Norm over Updates"),
        ("uncond_norm", "Gradient Norm", "Gradient Norm over Updates"),
        ("cond_grad_norm", "Cond. Gradient Norm", "Post-Conditioning Gradient Norm over Updates"),
    ]

    n_metrics = len(metric_names)
    n_cols = 3
    n_rows = (n_metrics + n_cols - 1) // n_cols

    plt.figure(figsize=(6 * n_cols, 4 * n_rows))

    for idx, (attr, ylabel, title) in enumerate(metric_names, 1):
        plt.subplot(n_rows, n_cols, idx)
        data = getattr(metrics, attr)
        if len(data) > 0 and hasattr(data[0], "shape") and hasattr(data[0], "__len__"):
            # If each entry is an array, plot the mean and optionally std
            import numpy as np
            arr = np.array(data)[16:]
            if arr.ndim > 1:
                plt.plot(arr.mean(axis=1), label="mean")
                plt.fill_between(
                    range(len(arr)),
                    arr.mean(axis=1) - arr.std(axis=1),
                    arr.mean(axis=1) + arr.std(axis=1),
                    alpha=0.2,
                    label="std"
                )
                plt.legend()
            else:
                plt.plot(arr)
        else:
            plt.plot(data)
        plt.xlabel("Update")
        plt.ylabel(ylabel)
        plt.title(title)

    plt.tight_layout()
    plt.savefig("all_metrics.png")




if __name__ == "__main__":
    import os
    os.environ['TF_GPU_ALLOCATOR'] = 'cuda_malloc_async'
    metrics = train_policy()
    plot_metrics(metrics)