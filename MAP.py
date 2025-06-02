import numpy as np
import torch
from torch.optim import LBFGS
import pickle
import matplotlib.pyplot as plt
from prior import Matern_graph_pytorch
from plot_tools import Graph_Plotter
from matplotlib import animation

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

        self.G.update_L( self.prior_map(p), normalize=True ) # here we create a (non-linear) log-Gaussian prior
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

        self.G.update_L( self.prior_map(p), normalize=True ) # here we create a (non-linear) log-Gaussian prior
        return self.G.sample_heat(self.v0, nu=self.nu, dt=self.dt, T=self.T, w_noise=w_noise )



def MAP_stationary():
    with open('obs/torch/stationary/obs.pickle', 'rb') as handle:
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
    G = Matern_graph_pytorch(graph, normalize=True) # Initiating a graph Gaussian process
    problem = forward_operator_stationary(v0, G, nu_prior, prior_map) # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x: torch.sum( (problem.forward(x) - y_obs)**2/sigma2 ) + torch.sum( x**2 )

    x = torch.zeros(G.num_edges, dtype=torch.float64) # initial guess for the sampler
    x.requires_grad_(True)
    optimizer = LBFGS( [x], max_iter=1000, line_search_fn="strong_wolfe" ) # defining an optimization method

    def closure():
        optimizer.zero_grad()

        # negative log posterior
        loss = negative_log_posterior(x)

        # computing the gradient
        loss.backward()
        print(loss)
        return loss

    optimizer.step(closure)

    x_MAP = x
    y_MAP = problem.forward(x_MAP)

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

    # plotting the MAP graph weights
    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[1,0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_MAP.detach().numpy() ) # plotting the initial condition
    axes[1,0].set_title('reconstructed measurement')

    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[1,0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1,1], vmin=vmin, vmax=vmax)
    axes[1,1].set_title('estimated graph weights')

    plt.show()

def MAP_heat():
    with open('obs/torch/heat/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
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

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph, normalize=True) # Initiating a graph Gaussian process
    problem = forward_operator_heat(v0, G, prior_map, nu_prior, dt_prior, T_prior) # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x, w: torch.sum( (problem.forward(x, w) - y_obs)**2/sigma2 ) + torch.sum( (x)**2 ) + torch.sum( w**2 )

    x = torch.zeros(G.num_edges, dtype=torch.float64) # initial guess for the sampler
    x.requires_grad_(True)
    w_noise = torch.zeros_like( w_noise_true ).to(torch.float64)
    w_noise.requires_grad_(True)
    optimizer = LBFGS( [x, w_noise], max_iter=5000, line_search_fn="strong_wolfe" )

    def closure():
        optimizer.zero_grad()

        # negative log posterior
        loss = negative_log_posterior(x, w_noise)

        # computing the gradient
        loss.backward()
        print(loss)
        return loss

    optimizer.step(closure)

    # plotting the true parameter
    f, axes = plt.subplots(1,2, figsize=[12,6])
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    x_MAP = x
    w_noise_MAP = w_noise
    y_MAP = problem.forward(x_MAP, w_noise_MAP)

    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true[-1].detach().numpy() ) # plotting the initial condition
    anim1 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)
    axes[0].set_title('noise-free measurements')

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[1])
    axes[1].set_title('true graph weights')
    
    temp = prior_map(x_true).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    f, axes = plt.subplots(1,2, figsize=[12,6])

    # plotting the MAP graph weights
    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_MAP[-1].detach().numpy() ) # plotting the initial condition
    anim2 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_MAP.shape[0], interval=100)
    axes[0].set_title('reconstructed measurements')

    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1], vmin=vmin, vmax=vmax)
    axes[0].set_title('estimated graph weights')


    plt.show()

if __name__ == '__main__':
    #MAP_stationary()
    MAP_heat()

