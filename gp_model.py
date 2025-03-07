from celerite2.jax import GaussianProcess, terms
import jax.numpy as jnp
import numpyro
import numpyro.distributions as dist

def harmonic_SHO_model(t, y, yerr, yquarters, f0, nharmonics, f0_frac_uncert=0.1):
    uquarters, quarter_indices = jnp.unique(yquarters, return_inverse=True)
    nquarters = uquarters.shape[0]

    n_in_bins = jnp.bincount(quarter_indices)
    y_means = jnp.bincount(quarter_indices, weights=y) / n_in_bins
    y_vars = jnp.bincount(quarter_indices, weights=(y - y_means[quarter_indices])**2) / n_in_bins
    y_std = jnp.sqrt(y_vars)

    y_tot_var = jnp.sum(y_vars * n_in_bins) / jnp.sum(n_in_bins)
    y_tot_std = jnp.sqrt(y_tot_var)

    mus_scaled = numpyro.sample('mus_scaled', dist.Normal(0,1), sample_shape=(nquarters,))
    mus = numpyro.deterministic('mus', mus_scaled * y_std + y_means)

    y_centered = y - mus[quarter_indices]

    log_fs_scaled = numpyro.sample('log_fs_scaled', dist.Normal(0, 1), sample_shape=(nharmonics,))
    log_fs = numpyro.deterministic('log_fs', log_fs_scaled*f0_frac_uncert + jnp.log(f0) + jnp.log(jnp.arange(nharmonics)))
    fs = numpyro.deterministic('fs', jnp.exp(log_fs))

    Qs = numpyro.sample('Qs', dist.LogNormal(jnp.log(10.0), jnp.log(10.0)/2), sample_shape=(nharmonics,))
    sigmas = numpyro.sample('sigmas', dist.LogNormal(jnp.log(y_tot_std), jnp.log(10.0)/2), sample_shape=(nharmonics,))

    trms = [terms.SHOTerm(w0=2*jnp.pi*fs[i], Q=Qs[i], sigma=sigmas[i]) for i in range(nharmonics)]
    kernel = terms.TermSum(*trms)

    gp = GaussianProcess(kernel, mean=0.0)
    gp.compute(t, yerr=yerr)
    numpyro.factor('log_likelihood', gp.log_likelihood(y_centered))