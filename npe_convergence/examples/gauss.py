# Just do an easy Gauss ... perhaps one can actually exactly compute the TV for
import jax.numpy as jnp
import jax.random as random


def gauss(key, loc, scale, batch_size=1):
    return random.multivariate_normal(key, loc, scale, shape=(batch_size, loc.shape[0]))

def get_summaries(x):
    # TODO:
    return jnp.mean(x, axis=0)#, jnp.var(x, axis=0)
