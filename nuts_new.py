import numpy as np
import torch
from torch.optim import LBFGS
import pickle
import matplotlib.pyplot as plt
from prior import Matern_graph_pytorch_new
from pyro.infer.mcmc import MCMC, NUTS
from plot_tools import Graph_Plotter

class forward_operator_stationary():
    """
    This class creates a forward operator for the Graph stationary Gaussian process.
    To initiate it takes the initial conidition (source term) of the stationary 
    Gaussian process and a Graph Gaussian process.

    The forward operator takes the graph weights and updates the Laplacian operator
    of the Graph GP and then solves the stationary graph GP for the sources term 
    in the right-hand-side.
    """


    def __init__(self, v0, G, nu, prior_map):
        """
        Parameters:
        -----------
        v0: initial value
        G: graph
        prior_map: mapping used for positivity of the prior
        nu: regularity of the prior
        """
        self.v0 = v0
        self.G = G
        self.prior_map = prior_map
        self.nu = nu


    def forward(self, p):
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

        self.G.compute_laplacian_from_tensor_autograd( self.prior_map(p), normalized=True ) # here we create a (non-linear) log-Gaussian prior
        return self.G.sample_stationary(self.v0, nu=self.nu)


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

        self.G.compute_laplacian_from_tensor_autograd( self.prior_map(p), normalized=True ) # here we create a (non-linear) log-Gaussian prior
        return self.G.sample_heat(self.v0, nu=self.nu, dt=self.dt, T=self.T, w_noise=w_noise )

def sample_posterior_stationary():
    with open('obs/torch/correlation/stationary/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch_new(graph) # Initiating a graph Gaussian process
    problem = forward_operator_stationary(v0, G, nu_prior, prior_map) # creating a forward operator

# defining the log-posterior with a standard normal Gaussian prior
    neg_log_posterior = lambda x: 0.5*torch.sum( (problem.forward(x) - y_obs)**2/sigma2 ) + 0.5*torch.sum( x**2 )

    def pyro_NLP(x):
        return neg_log_posterior(x['x'])

    nuts_kernel = NUTS(potential_fn=pyro_NLP)

    MCMC_params = {
        'kernel': nuts_kernel,
        'num_chains': 1,
        'initial_params': {'x': torch.zeros(x_true.shape, dtype=torch.float64) },  
        'num_samples': 2000,
        'warmup_steps': 1000
    }

    mcmc = MCMC(**MCMC_params)
    mcmc.run()

    samples = mcmc.get_samples()

    stat_data = {'samples': samples}

    with open('./stat/paper_experiments/stationary/samples.pickle', 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(samples['x'].shape)


    x_mean = torch.mean( samples['x'], dim=0 )
    y_mean = problem.forward(x_mean)

    # plotting the true parameter
    f, axes = plt.subplots(2,2, figsize=[12,12])
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0,0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true.detach().numpy() ) # plotting the initial condition
    axes[0,0].set_title('noise free measurement')

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0,0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[0,1])
    axes[0,1].set_title('true graph weights')

    temp = prior_map(x_true).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    # plotting the mean graph weights
    plotter = Graph_Plotter(graph, y_mean.detach().numpy(), axes[1,0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_mean.detach().numpy() ) # plotting the initial condition
    axes[1,0].set_title('reconstructed measurement')

    plotter = Graph_Plotter(graph, y_mean.detach().numpy(), axes[1,0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_mean).detach().numpy(), axes[1,1], vmin=vmin, vmax=vmax)
    axes[1,1].set_title('estimated graph weights')

    plt.show()

if __name__ == '__main__':
    sample_posterior_stationary()
