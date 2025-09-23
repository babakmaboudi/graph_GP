import numpy as np
import torch
from torch.optim import LBFGS
import pickle
import matplotlib.pyplot as plt
from prior import Matern_graph_pytorch, heat_periodic_pytorch_new
from plot_tools import Graph_Plotter
from matplotlib import animation
from pyro.infer.mcmc import MCMC, NUTS

def sample_posterior():
    dtype = torch.float64
    with open('obs/torch/heat_1D/obs_non_smooth.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    v0 = obs_data['v0']
    T_max = obs_data['T_max']
    dt = obs_data['dt']
    N = obs_data['N']
    nu = obs_data['nu']
    MAX_ITER = int(T_max/dt)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma

    prior = heat_periodic_pytorch_new(N)

    def forward_operator(p,w):
        prior.update_L( torch.exp(p) )
        return prior.sample_heat( v0, nu=nu, T=T_max, dt=dt, w_noise=w )

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x, w: torch.sum( (forward_operator(x, w) - y_true)**2/sigma2 ) + torch.sum( (x)**2 ) + torch.sum( w**2 )

    def pyro_NLP(x):
        return negative_log_posterior(x['x'], x['w'])

    nuts_kernel = NUTS(potential_fn=pyro_NLP)

    MCMC_params = {
        'kernel': nuts_kernel,
        'num_chains': 1,
        'initial_params': {'x': torch.zeros(N-1, dtype=torch.float64), 'w': torch.randn(MAX_ITER, N, dtype=torch.float64) },
        'num_samples': 2000,
        'warmup_steps': 200
    }

    mcmc = MCMC(**MCMC_params)
    mcmc.run()

    samples = mcmc.get_samples()

    print(samples)

    stat_data = {'x': samples['x'], 'w': samples['w']}

    with open('./stat/heat_1D/non-smooth-samples.pickle', 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
    

if __name__ == "__main__":
    sample_posterior()
