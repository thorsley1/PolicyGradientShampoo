import jax
import jax.numpy as jnp
import jax.random as jr
from dataclasses import dataclass
from functools import partial
from typing import Tuple, List

from config import Config


@jax.tree_util.register_dataclass
@dataclass(frozen=True)
class Parameters:
    embed: List[jax.Array]
    unembed: List[jax.Array]
    unembed_norm_scale: List[jax.Array]
    layer_params: Tuple[List[jax.Array]]

    def list_flatten(self):
        children = (self.embed, self.unembed, self.unembed_norm_scale) + self.layer_params
        return tuple(children)

    @classmethod
    def list_unflatten(cls, children):
        embed, unembed, unembed_norm_scale, *layer_params = children
        return cls(embed, unembed, unembed_norm_scale, tuple(layer_params))

    def norm(self):
        leaves = jax.tree.leaves(self)
        return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def init_params(cfg: Config):
    seed = cfg.seed
    d_emb = cfg.d_emb
    d_unemb = cfg.d_unemb
    d_model = cfg.d_model
    d_mlp = cfg.d_mlp
    n_layers = cfg.n_layers

    base_key = jr.PRNGKey(seed)
    emb_key, unemb_key, unembed_norm_scale_key, proj_in_key, proj_gate_key, proj_out_key, mlp_norm_scale_key = jr.split(base_key, 7)

    d_model_scale = jax.lax.rsqrt(float(d_model))
    d_mlp_scale = jax.lax.rsqrt(float(d_mlp))
    
    embed = [jr.normal(emb_key, (1, d_emb, d_model)), ]
    unemb = [jr.normal(unemb_key, (1, d_model, d_unemb)) * d_model_scale, ]
    unemb_norm_scale = [jr.normal(unembed_norm_scale_key, (1, d_model)) * d_model_scale, ]
    proj_in = [jr.normal(proj_in_key, (n_layers, d_model, d_mlp)) * d_model_scale, ]
    proj_gate = [jr.normal(proj_gate_key, (n_layers, d_model, d_mlp)) * d_model_scale, ]
    proj_out = [jr.normal(proj_out_key, (n_layers, d_mlp, d_model)) * d_mlp_scale, ]
    mlp_norm_scales = [jr.normal(mlp_norm_scale_key, (n_layers, d_model)) * d_model_scale, ]

    layer_params = (proj_in, proj_gate, proj_out, mlp_norm_scales)

    return Parameters(embed=embed, unembed=unemb, unembed_norm_scale=unemb_norm_scale, layer_params=layer_params)


@partial(jax.jit, static_argnums=(2,))
def model(params: Parameters, obs: jax.Array, cfg: Config):
    def apply_layer(res, layer_params, norm_eps=cfg.norm_eps):
        proj_in, proj_gate, proj_out, norm_scales = layer_params
        normed_res = res * jax.lax.rsqrt(jnp.mean(jnp.square(res), axis=-1, keepdims=True) + norm_eps) * norm_scales[0]
        pre = jnp.matmul(normed_res, proj_in[0])
        gate = jax.nn.gelu(jnp.matmul(res, proj_gate[0]))
        post = pre * gate
        res = res + jnp.matmul(post, proj_out[0])
        return res, None

    res = jnp.matmul(obs, params.embed[0][0])
    res, _ = jax.lax.scan(apply_layer, res, params.layer_params)
    res = res * jax.lax.rsqrt(jnp.mean(jnp.square(res), axis=-1, keepdims=True) + cfg.norm_eps) * params.unembed_norm_scale[0][0]
    return jax.nn.log_softmax(jnp.matmul(res, params.unembed[0][0]), axis=-1)