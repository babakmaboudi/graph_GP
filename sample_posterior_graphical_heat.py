import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
from prior import Matern_graph_pytorch
from pyro.infer.mcmc import MCMC, NUTS
from plot_tools import Graph_Plotter
from matplotlib import animation
import os
from create_signal import make_1d_grid_graph
import networkx as nx

class forward_operator_heat():
    """
    This class creates a forward operator for the Graph non-stationary (heat) Gaussian process.
    To initiate it takes the initial conidition (source term) of the stationary 
    Gaussian process and a Graph Gaussian process.

    The forward operator takes the graph weights and updates the Laplacian operator
    of the Graph GP and then solves the stationary graph GP for the sources term 
    in the right-hand-side.
    """


    def __init__(self, v0, G, prior_map, nu, dt, T):
        """
        Parameters:
        -----------
        v0: initial value
        G: graph
        prior_map: mapping used for positivity of the prior
        nu: regularity of the prior
        dt: processed time step
        T: maximum process time

        """
        self.v0 = v0
        self.G = G
        self.prior_map = prior_map
        self.nu = nu
        self.dt = dt
        self.T = T



    def forward(self, p, w_noise):
        """
        This function takes graph weights p and solves the stationary graph
        GP for source term v0

        Parameters:
        -----------
        p: np array of graph weights
        w: process noise

        Returns:
        --------
        solution of IVP for stationary graph
        """

        self.G.compute_laplacian_from_tensor_autograd( self.prior_map(p), normalized=False ) # here we create a (non-linear) log-Gaussian prior
        return self.G.sample_heat(self.v0, nu=self.nu, dt=self.dt, T=self.T, w_noise=w_noise )




def sample_posterior_heat():
    with open('obs/graphical_heat/heat/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    #x_smooth = obs_data['x_smooth']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']
    dt_prior = obs_data['dt_prior']
    T_prior = obs_data['T_prior']
    MAX_ITER_prior = obs_data['MAX_ITER_prior']
    w_noise_true = obs_data['w_noise_true']
    N = obs_data['N']

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = make_1d_grid_graph(N, weighted = False)
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process
    problem = forward_operator_heat(v0, G, prior_map, nu_prior, dt_prior, T_prior) # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    neg_log_posterior = lambda x, w: torch.sum( (problem.forward(x, w) - y_obs)**2/sigma2 ) + torch.sum( (x)**2 ) + torch.sum( w**2 )

    def pyro_NLP(x):
        return neg_log_posterior(x['x'], x['w'])

    nuts_kernel = NUTS(potential_fn=pyro_NLP)

    MCMC_params = {
        'kernel': nuts_kernel,
        'num_chains': 1,
        'initial_params': {'x': torch.zeros(x_true.shape, dtype=torch.float64), 'w': torch.zeros(w_noise_true.shape, dtype=torch.float64) },  
        'num_samples': 2000,
        'warmup_steps': 1000
    }

    mcmc = MCMC(**MCMC_params)
    mcmc.run()

    samples = mcmc.get_samples()
    samples_x = samples['x']
    samples_w = samples['w']

    stat_data = {'x_samples': samples_x, 'w_samples': samples_w}

    dir = './stat/graphical_heat/heat/'
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(dir, 'samples.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    x_mean = torch.mean(samples_x, axis=0)
    x_std = torch.std(samples_x, axis=0)
    w_mean = torch.mean(samples_w, axis=0)
    y_mean = problem.forward(x_mean, w_mean)

    # plotting the true parameter
    f, axes = plt.subplots(1,2, figsize=[12,6])
    pos = nx.get_node_attributes(graph, "pos")

    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=pos) # initiating the graph plotter
    plotter.plot_stationary(y_true[-1].detach().numpy() ) # plotting the initial condition
    anim1 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)
    axes[0].set_title('noise-free measurements')
    #axes[0].set_aspect('equal')

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=pos)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[1])
    axes[1].set_title('true graph weights')
    #axes[1].set_aspect('equal')
    
    temp = prior_map(x_true).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    f, axes = plt.subplots(1,2, figsize=[12,6])

    # plotting the MAP graph weights
    plotter = Graph_Plotter(graph, y_mean.detach().numpy(), axes[0], pos=pos) # initiating the graph plotter
    plotter.plot_stationary(y_mean[-1].detach().numpy() ) # plotting the initial condition
    anim2 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_mean.shape[0], interval=100)
    axes[0].set_title('reconstructed measurements')
    #axes[0].set_aspect('equal')

    plotter = Graph_Plotter(graph, y_mean.detach().numpy(), axes[0], pos=pos)
    plotter.plot_graph_wieghts(prior_map(x_mean).detach().numpy(), axes[1])#, vmin=vmin, vmax=vmax)
    axes[0].set_title('estimated graph weights')
    #axes[0].set_aspect('equal')


    plt.show()
if __name__ == '__main__':
    sample_posterior_heat()
