from celerite2.pymc import terms, GaussianProcess
import pymc as pm
import pytensor.tensor as pt

def harmonic_SHO_model(t, y, yerr, yquarters, f0, nharmonics, mu_mu, mu_sigma, sho_sigma_prior, f0_frac_uncert=0.1):
    with pm.Model() as model:
        nquarters = mu_mu.shape[0]

        mus_scaled = pm.Normal('mus_scaled', 0, 1, shape=(nquarters,))
        mus = pm.Deterministic('mus', mus_scaled * mu_sigma + mu_mu)

        y_centered = y - mus[yquarters]

        log_f0_scaled = pm.Normal('log_f0_scaled', 0, 1)
        log_f0 = pm.Deterministic('log_f0', log_f0_scaled*f0_frac_uncert + pt.log(f0))
        fs = pm.Deterministic('fs', pt.exp(log_f0 + pt.log(np.arange(nharmonics) + 1)))

        log_Qs = pm.Uniform('log_Qs', pt.log(1), pt.log(100), shape=(nharmonics,))
        Qs = pm.Deterministic('Qs', pt.exp(log_Qs))

        sigmas = pm.LogNormal('sigmas', pt.log(sho_sigma_prior), pt.log(10.0)/2, shape=(nharmonics,))

        trms = [terms.SHOTerm(w0=2*np.pi*fs[i], Q=Qs[i], sigma=sigmas[i]) for i in range(nharmonics)]
        kernel = terms.TermSum(*trms)

        gp = GaussianProcess(kernel)
        gp.compute(t, yerr=yerr)
        pm.Potential('log_likelihood', gp.log_likelihood(y_centered))

        return model