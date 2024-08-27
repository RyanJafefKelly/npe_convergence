import jax.numpy as jnp
import jax.random as random
from jax import lax

import numpy as np

import numpyro
import numpyro.contrib
import numpyro.contrib.control_flow
import numpyro.distributions as dist

PAIRS = ([0, 5], [3, 1], [4, 2])
# indices = ([0, 5], ...)  # TODO? neater or weird?


def svar(key, theta, n_obs=1_000):
    n_obs=100_000  # TODO! EXPERIMENT
    k = theta.shape[-1] - 1  # dimension of VAR, last element is sigma

    # Initialising transition matrix X
    X_base = -0.1 * jnp.eye(k)
    indices = []
    for i, j in PAIRS:
        indices.append((i, j))  # Forward pair
        indices.append((j, i))  # Reverse pair

    indices = jnp.array(indices)
    indices = (indices[:, 0], indices[:, 1])

    X = X_base.at[indices].set(theta[:-1])

    sigma = theta[-1]
    Y0 = sigma * random.normal(key, (k,))  # Initial state

    def transition(carry, _):
        key, Y_prev = carry
        key, subkey = random.split(key)  # Generate a new subkey for each time step
        Y_new = jnp.matmul(X, Y_prev) + sigma * random.normal(subkey, (k,))
        return (key, Y_new), Y_new

    # Using scan to simulate the time series
    _, Y = lax.scan(transition, (key, Y0), None, length=n_obs)
    return Y



def compute_summaries(Y):
    n = Y.shape[0]
    k = Y.shape[1]
    S = jnp.zeros(k+1)

    for ii, pair in enumerate(PAIRS):
        S = S.at[ii*2].set(jnp.mean(Y[pair[0], 1:] * Y[pair[1], 0:-1]))
        S = S.at[ii*2 + 1].set(jnp.mean(Y[pair[1], 1:] * Y[pair[0], 0:-1]))
        # S[ii*2 + 1] = np.mean(Y[pair[1], 1:]* Y[pair[0], 0:-1])

    S = S.at[-1].set(jnp.std(Y))

    return S


# def expected_autocovariance(X, theta, sigma):
#     autocovariance = sigma**2 * X  # TODO! WRONG I THINK
#     return autocovariance


def get_model(obs, n_obs):
    # TODO: more compact way to write this?
    numpyro_key = numpyro.prng_key()
    print('numpyro_key: ', numpyro_key)
    # key = random.PRNGKey(987)  # TODO: problematic same key each run... or good?

    key = numpyro_key if numpyro_key is not None else random.PRNGKey(987)

    theta_dims = 6  # TODO! MAGIC
    theta = numpyro.sample("theta", dist.Uniform(-.9, 0.9).expand([theta_dims]))

    sigma = numpyro.sample("sigma", dist.Uniform(0, 1))

    k = len(theta)
    X = -0.1 * jnp.eye(k)

    indices = []
    for i, j in PAIRS:
        indices.append((i, j))  # Forward pair
        indices.append((j, i))  # Reverse pair

    # Convert to a numpy array for easy indexing if necessary
    indices = jnp.array(indices)
    indices = tuple((indices[:, 0], indices[:, 1]))  # Convert to tuple for indexing

    X = X.at[indices].set(theta)
    expected_autocov = sigma**2 * (jnp.matmul(X, X.T) + jnp.eye(k))  # TODO! UNSURE ABOUT
    expected_autocov_vector = expected_autocov[indices]
    numpyro.deterministic("expected_autocov", expected_autocov_vector)

    # TODO: for loop
    # series = jnp.zeros((n_obs + 1, k))
    # Y0 = sigma * random.normal(key, (k,))  # Initial state
    # series = series.at[0, :].set(Y0)  # Set initial state

    # key, subkey = random.split(key)  # Split the key for each time step
    # noise = sigma * random.normal(subkey, (n_obs, k))

    # # Simulate the time series
    # for t in range(n_obs):
    #     m_t = jnp.dot(X, series[t, :])  # Matrix multiplication
    #     series = series.at[t + 1, :].set(m_t + noise[t, :])  # Update the series array

    # TODO: SCAN
    # def transition(carry, _):
    #     y_prev, key = carry
    #     key, subkey = random.split(key)

    #     m_t = jnp.matmul(X, y_prev) #+ sigma * random.normal(subkey, shape=(k,))
    #     y_t = m_t + sigma * random.normal(subkey, shape=(k,))
    #     return (y_t, key), m_t

    # timesteps = jnp.arange(n_obs)
    # # init = obs[0]
    # key, subkey = random.split(key)
    # Y0 = sigma * random.normal(subkey, (k,))
    # init = Y0, key  # Initial state with zero vector

    # # with numpyro.handlers.condition(data={"y": obs[1:]}):
    #     # _, mu = numpyro.contrib.control_flow.scan(transition, init, timesteps)
    # # numpyro.contrib.control_flow.scan(transition, init, timesteps)
    # _, series = lax.scan(transition, init, timesteps)


    # # numpyro.deterministic("mu", mu)
    # autocov = compute_summaries(series)[:-1]
    # autocov = jnp.zeros(6)
    # for ii, pair in enumerate(PAIRS):
    #     autocov = autocov.at[2*ii].set(jnp.mean(series[pair[0], 1:] * series[pair[1], :-1]))
    #     autocov = autocov.at[2*ii + 1].set(jnp.mean(series[pair[1], 1:] * series[pair[0], :-1]))

    # autocov = jnp.mean(series[1:, :] * series[:-1, :], axis=0)
    # numpyro.deterministic("autocov", autocov)
    # sample_std = jnp.std(series)



    # variance_scale = 1/jnp.sqrt(n_obs)
    variance_scale = 0.1  # TODO! FIX
    # variance_scale = 1
    obs_data = {f'obs_autocov{i+1}': obs[i] for i in range(k)}
    obs_data['obs_std'] = obs[6]

    with numpyro.handlers.condition(data=obs_data):
        for i in range(6):
            numpyro.sample(f'obs_autocov{i+1}', dist.Normal(expected_autocov_vector[i], variance_scale))

        numpyro.sample('obs_std', dist.Normal(sigma, variance_scale))

    # TODO: MAY WANT TO USE A DIFFERENT VARIANCE THAN TRANSFORM FOR STABILITY
    # for i in range(6):
    #     # Sample from a standard normal distribution
    #     # z = numpyro.sample(f'z_autocov{i+1}', dist.Normal(0, 1))
        
    #     # Transform z to have the desired mean and standard deviation
    #     # transformed_autocov = autocov[i] + z * variance_scale
        
    #     # Condition on the observed autocovariance
    #     numpyro.sample(f'obs_autocov{i+1}', dist.Normal(autocov[i], 0.1), obs=obs[i])
    # numpyro.sample('obs_std', dist.Normal(sample_std, variance_scale), obs=obs[6])

    return


def run_inference(obs, n_obs, key):
    model = get_model
    true_params = jnp.array([0.579, -0.143, 0.836, 0.745, -0.660, -0.254, 0.1]) # TODO: CHEAT TO SEE IF WORKS
    # TODO MCMC SAMPLING
    sampler = numpyro.infer.NUTS(model)
    mcmc = numpyro.infer.MCMC(sampler, num_warmup=2_000, num_samples=10_000, num_chains=1)
    key, subkey = random.split(key)
    mcmc.run(subkey, obs, n_obs, init_params={'theta': true_params[:-1], 'sigma': true_params[-1]})
    mcmc.print_summary()
    return mcmc.get_samples()
