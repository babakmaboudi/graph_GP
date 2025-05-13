import numpy as np
import torch
from torch.optim import LBFGS
import pickle
import matplotlib.pyplot as plt
from Matern_prior import Matern_graph_pytorch, Graph_Plotter
from matplotlib import animation
from pyro.infer.mcmc import MCMC, NUTS


class forward_operator():
    """
    This class creates a forward operator for the Graph stationary Gaussian process.
    To initiate it takes the initial conidition (source term) of the stationary 
    Gaussian process and a Graph Gaussian process.

    The forward operator takes the graph weights and updates the Laplacian operator
    of the Graph GP and then solves the stationary graph GP for the sources term 
    in the right-hand-side.
    """


    def __init__(self, v0, G):
        """
        Parameters:
        -----------
        v0: initial value
        G: graph
        """
        self.v0 = v0
        self.G = G


    def forward(self, p, w_noise, nu=1, dt=0.1, T=5.):
        """
        This function takes graph weights p and solves the stationary graph
        GP for source term v0

        Parameters:
        -----------
        p: np array of graph weights

        Returns:
        --------
        solution of IVP for stationary graph
        """

        self.G.update_L( 0.1*torch.exp(p), normalize=True ) # here we create a (non-linear) log-Gaussian prior
        return self.G.sample_heat(self.v0, nu=nu, dt=dt, T=T, w_noise=w_noise )

def inverse():
    """
    This function will sample from the posterior distribution constructed by a Grap Gaussian process.
    This function considers an observation file './obs/obs_stationary.pickle' with components:
    
    y_true: noise free observation
    noise_vec: a normalized noise vector follosing a Gaussian distribution
    x_true: (not needed for sampling) the true solution to the inverse problem
    """

    # loading the observation vector
    with open('./obs/heat/reg1/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    dtype = obs_data['dtype']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec'] 
    x_true = obs_data['x_true']
    w_noise_true = obs_data['w_noise_true']
    dt = obs_data['dt']
    T = obs_data['T']
    MAX_ITER = obs_data['MAX_ITER']
    nu = obs_data['nu']
    v0 = obs_data['v0']
    sigma = obs_data['sigma']

    # creating signal/observation with noise std defined in sigma with 1% noise level
    #sigma = 0.01*torch.linalg.norm(y_true)/torch.sqrt( torch.tensor( y_true.shape[0] ) )
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    #f,axes = plt.subplots(1,2)
    #axes[0].imshow(y_true.detach().numpy())
    #axes[1].imshow(y_obs.detach().numpy())

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph, normalize=True) # Initiating a graph Gaussian process
    problem = forward_operator(v0, G) # creating a forward operator  

    # defining the log-posterior with a standard normal Gaussian prior
    neg_log_posterior = lambda x, w: 0.5*torch.sum( (problem.forward(x, w, nu=nu, dt=dt, T=T) - y_obs)**2/sigma2 ) + 0.5*torch.sum( (x)**2 ) + 0.5*torch.sum( w**2 )

    def pyro_NLP(x):
        return neg_log_posterior(x['x'], x['w'])

    with open('./stat/heat/reg1/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)

    x_MAP = MAP_data['x_MAP']
    #x0.requires_grad_(True)
    w_MAP = MAP_data['w_MAP']
    #w0.requires_grad_(True)
    x0 = x_MAP.clone().detach().to(dtype=torch.float64)
    w0 = w_MAP.clone().detach().to(dtype=torch.float64)

    nuts_kernel = NUTS(potential_fn=pyro_NLP)

    MCMC_params = {
        'kernel': nuts_kernel,
        'num_chains': 1,
        'initial_params': {'x': x0, 'w': w0 },  
        'num_samples': 1000,
        'warmup_steps': 200
    }

    mcmc = MCMC(**MCMC_params)
    mcmc.run()

    samples = mcmc.get_samples()
    samples_x = samples['x']
    samples_w = samples['w']

    x_mean = torch.mean(samples_x, axis=0)
    x_std = torch.std(samples_x, axis=0)
    w_mean = torch.mean(samples_w, axis=0)
    y_mean = problem.forward(x_mean, w_mean, nu=nu, dt=dt, T=T)

    x_prior = torch.randn_like(x_mean)
    w_prior = torch.randn_like(w_mean) 
    y_prior = problem.forward(x_prior, w_prior, nu=nu, dt=dt, T=T)

    f,axes = plt.subplots(1,4)
    axes[0].imshow(y_true.detach().numpy())
    axes[0].set_title('true')
    axes[1].imshow(y_obs.detach().numpy())
    axes[1].set_title('noisey')
    axes[2].imshow(y_mean.detach().numpy())
    axes[2].set_title('mean')
    axes[3].imshow(y_prior.detach().numpy())
    axes[3].set_title('sample from prior')


    f,ax = plt.subplots(1)
    ax.imshow( torch.abs( (y_true - y_mean)/(y_true) ).detach().numpy() )

    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    f,ax = plt.subplots(1)
    plotter = Graph_Plotter(graph, y_mean.detach().numpy(), ax, pos=state_positions)
    plotter.plot_graph_wieghts(y_mean.detach().numpy(), x_std.detach().numpy(), ax)
    ax.set_title('std')

    f,ax = plt.subplots(1)
    plotter = Graph_Plotter(graph, y_mean.detach().numpy(), ax, pos=state_positions)
    plotter.plot_graph_wieghts(y_mean.detach().numpy(), torch.abs(x_std - x_true).detach().numpy(), ax)
    ax.set_title('error')

    plt.show()

if __name__ == '__main__':
    inverse()
