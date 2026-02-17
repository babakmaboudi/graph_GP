import numpy as np
import torch
from torch.optim import LBFGS
import pickle
import matplotlib.pyplot as plt
from prior import Matern_graph_pytorch
from plot_tools import Graph_Plotter
from matplotlib import animation
import os

class forward_operator_stationary():
    """
    This class creates a forward operator for the Graph stationary Gaussian process.
    To initiate it takes the initial conidition (source term) of the stationary 
    Gaussian process and a Graph Gaussian process.

    The forward operator takes the graph weights and updates the Laplacian operator
    of the Graph GP and then solves the stationary graph GP for the sources term 
    in the right-hand-side.
    """


    def __init__(self, v0, G, nu, prior_map, covid=False):
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
        self.covid = covid

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

class forward_operator_stationary_linearreaction():
    def __init__(self, v0, G, nu, prior_map, covid=False):
        self.v0 = v0
        self.G = G
        self.prior_map = prior_map
        self.nu = nu
        self.covid = covid

    def forward(self, p):
        self.G.compute_laplacian_from_tensor_autograd( self.prior_map(p), normalized=True ) # here we create a (non-linear) log-Gaussian prior
        if self.covid==False:
            return self.G.sample_stationary_with_linearreaction(self.v0, nu=self.nu)
        else:
            return self.G.sample_stationary_with_linearreaction(self.v0, nu=self.nu, clamp01=True)


class forward_operator_stationary_nonlinearreaction():
    def __init__(self, v0, G, nu, prior_map, covid=False):
        self.v0 = v0
        self.G = G
        self.prior_map = prior_map
        self.nu = nu
        self.covid = covid

    def forward(self, p):
        self.G.compute_laplacian_from_tensor_autograd( self.prior_map(p), normalized=True ) # here we create a (non-linear) log-Gaussian prior
        if self.covid == False:
            return self.G.sample_stationary_with_nonlinearreaction(self.v0, nu=self.nu)
        else:
            return self.G.sample_stationary_with_nonlinearreaction(self.v0, nu=self.nu, clamp01=True)



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
        return self.G.sample_heat(self.v0, nu=self.nu, dt=self.dt, T=self.T, w_noise=w_noise)


class forward_operator_heat_linearreaction():
    """
    This class creates a forward operator for the Graph non-stationary (heat) Gaussian process.
    To initiate it takes the initial conidition (source term) of the stationary
    Gaussian process and a Graph Gaussian process.

    The forward operator takes the graph weights and updates the Laplacian operator
    of the Graph GP and then solves the stationary graph GP for the sources term
    in the right-hand-side.
    """


    def __init__(self, v0, G, prior_map, nu, dt, T, covid=False):
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
        self.covid = covid



    def forward(self, p, w_noise, implicit=True):
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
        if implicit and self.covid==False:
            return self.G.sample_heat_with_linearreaction_implicit(self.v0, nu=self.nu, dt=self.dt, T=self.T, w_noise=w_noise)
        elif implicit and self.covid:
            return self.G.sample_heat_with_linearreaction_implicit(self.v0, nu=self.nu, dt=self.dt, T=self.T,
                                                                   w_noise=w_noise, clamp01=True)
        elif implicit==False and self.covid:
            return self.G.sample_heat_with_linearreaction(self.v0, nu=self.nu, dt=self.dt, T=self.T,
                                                          w_noise=w_noise, clamp01=True)
        else:
            return self.G.sample_heat_with_linearreaction(self.v0, nu=self.nu, dt=self.dt, T=self.T,
                                                                   w_noise=w_noise)


class forward_operator_heat_nonlinearreaction():
    """
    This class creates a forward operator for the Graph non-stationary (heat) Gaussian process.
    To initiate it takes the initial conidition (source term) of the stationary
    Gaussian process and a Graph Gaussian process.

    The forward operator takes the graph weights and updates the Laplacian operator
    of the Graph GP and then solves the stationary graph GP for the sources term
    in the right-hand-side.
    """


    def __init__(self, v0, G, prior_map, nu, dt, T, covid=False):
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
        self.covid = covid



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
        if self.covid == False:
            return self.G.sample_heat_with_nonlinreaction(self.v0, nu=self.nu, dt=self.dt, T=self.T, w_noise=w_noise )
        else:
            return self.G.sample_heat_with_nonlinreaction(self.v0, nu=self.nu, dt=self.dt, T=self.T, w_noise=w_noise, clamp01=True)



def MAP_stationary():
    with open('obs/stationary/obs.pickle', 'rb') as handle:
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
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process
    problem = forward_operator_stationary(v0, G, nu_prior, prior_map) # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x: torch.sum( (problem.forward(x) - y_obs)**2/sigma2 ) + torch.sum( x**2 )

    x = torch.zeros(110, dtype=torch.float64) # initial guess for the sampler
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

    stat_data = {'x_MAP': x_MAP}
    dir = './stat/stationary/'
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(dir, 'MAP.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plotting the true parameter
    f, axes = plt.subplots(2,2, figsize=[12,12])
    with open('stat_positions_exact.pickle' , 'rb') as handle:
        state_positions = pickle.load(handle)


    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0,0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true.detach().numpy() ) # plotting the initial condition
    axes[0,0].set_title('noise free measurement')
    axes[0,0].set_aspect('equal')

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0,0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[0,1])
    axes[0,1].set_title('true graph weights')
    axes[0,1].set_aspect('equal')

    temp = prior_map(x_true).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    # plotting the MAP graph weights
    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[1,0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_MAP.detach().numpy() ) # plotting the initial condition
    axes[1,0].set_title('reconstructed measurement')
    axes[1,0].set_aspect('equal')

    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[1,0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1,1], vmin=vmin, vmax=vmax)
    axes[1,1].set_title('estimated graph weights')
    axes[1,1].set_aspect('equal')

    plt.show()


def MAP_stationary_linearreaction_real_covid(covid=True):
    # covid=True activates clampint to [0,1]
    with open('data/covid_data/y_cases_normalized.pkl', 'rb') as handle:
        obs_data = pickle.load(handle)


    #noise_vec = obs_data['noise_vec']
    nu_prior = 1  # regularity
    # The non-linear mapping that makes the prior positive
    prior_map_scale = .1  # output variance parameter
    prior_map_mean = .0
    prior_map_text = 'lambda x:  prior_map_mean + prior_map_scale*prior_map_parameter*torch.exp(x)'  # the copy of the mapping for future reference
    prior_map = lambda x: prior_map_mean + prior_map_scale * torch.exp(x)  # the actual mapping
    #nu_prior = obs_data['nu_prior']
    #prior_map_mean = obs_data['prior_map_mean']
    #prior_map_scale = obs_data['prior_map_scale']

    dtype = torch.float64
    graph = pickle.load(
        open('./data/covid_data/g.pkl', "rb"))  # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph)  # Initiating a graph Gaussian process
    v0 = torch.zeros(G.num_nodes, dtype=dtype)
    # v0 = 10*torch.ones(G.N, dtype=dtype)
    # v0 = 1+torch.rand(G.N, dtype=dtype)
    v0[25] = 1.
    #v0 = obs_data['v0']

    prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    y_obs = obs_data#['y_obs']
    y_obs = y_obs.reshape(51, 80)[:, 0]  # numpy array, shape (51,)
    #could take mean of 80 time points as well!
    #y_obs = y_obs.reshape(51, 80).mean(axis=1)  # (51,)
    #y_obs = y_true + sigma*noise_vec


    # creating a forward operator
    #graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    #G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process
    problem = forward_operator_stationary_linearreaction(v0, G, nu_prior, prior_map,covid=covid) # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x: torch.sum( (problem.forward(x) - y_obs)**2/sigma2 ) + torch.sum( x**2 )

    x = torch.zeros(110, dtype=torch.float64) # initial guess for the sampler
    x.requires_grad_(True)

    y_obs = torch.as_tensor(y_obs, dtype=torch.float32, device=x.device)
    x_obs = torch.normal(torch.zeros(G.num_edges), torch.ones(G.num_edges)).to(dtype)
    # creating signal/observation with noise std defined in sigma with 1% noise level
    #y_true_torch = torch.from_numpy(y_true)
    #sigma = 0.01 * torch.linalg.norm(y_true_torch)
    sigma = 0.01 * np.linalg.norm(y_obs)
    #sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma

    sigma2 = torch.as_tensor(sigma2, dtype=torch.float32, device=x.device)

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

    stat_data = {'x_MAP': x_MAP}

    dir = './stat/stationary_linearreaction_real_covid/'
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(dir, 'MAP.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plotting the true parameter
    f, axes = plt.subplots(1,2, figsize=[12,12])
    with open('stat_positions_exact.pickle' , 'rb') as handle:
        state_positions = pickle.load(handle)


    plotter = Graph_Plotter(graph, y_obs.detach().numpy(), axes[0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_obs.detach().numpy() ) # plotting the initial condition
    axes[0].set_title('measurement')
    axes[0].set_aspect('equal')


    temp = prior_map(x_obs).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    # plotting the MAP graph weights
    # plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[1,0], pos=state_positions) # initiating the graph plotter
    # plotter.plot_stationary(y_MAP.detach().numpy() ) # plotting the initial condition
    # axes[1,0].set_title('reconstructed measurement')
    # axes[1,0].set_aspect('equal')

    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1], vmin=vmin, vmax=vmax)
    axes[1].set_title('estimated graph weights')
    axes[1].set_aspect('equal')

    plt.show()


def MAP_stationary_linearreaction():
    with open('obs/stationary_linearreaction/obs.pickle', 'rb') as handle:
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
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process
    problem = forward_operator_stationary_linearreaction(v0, G, nu_prior, prior_map) # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x: torch.sum( (problem.forward(x) - y_obs)**2/sigma2 ) + torch.sum( x**2 )

    x = torch.zeros(110, dtype=torch.float64) # initial guess for the sampler
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

    stat_data = {'x_MAP': x_MAP}

    dir = './stat/stationary_linearreaction/'
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(dir, 'MAP.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plotting the true parameter
    f, axes = plt.subplots(2,2, figsize=[12,12])
    with open('stat_positions_exact.pickle' , 'rb') as handle:
        state_positions = pickle.load(handle)


    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0,0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true.detach().numpy() ) # plotting the initial condition
    axes[0,0].set_title('noise free measurement')
    axes[0,0].set_aspect('equal')

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0,0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[0,1])
    axes[0,1].set_title('true graph weights')
    axes[0,1].set_aspect('equal')

    temp = prior_map(x_true).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    # plotting the MAP graph weights
    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[1,0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_MAP.detach().numpy() ) # plotting the initial condition
    axes[1,0].set_title('reconstructed measurement')
    axes[1,0].set_aspect('equal')

    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[1,0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1,1], vmin=vmin, vmax=vmax)
    axes[1,1].set_title('estimated graph weights')
    axes[1,1].set_aspect('equal')

    plt.show()


def MAP_stationary_nonlinearreaction_real_covid(covid=True):
    # covid=True activates clampint to [0,1]
    with open('data/covid_data/y_cases_normalized.pkl', 'rb') as handle:
        obs_data = pickle.load(handle)


    #noise_vec = obs_data['noise_vec']
    nu_prior = 1  # regularity
    # The non-linear mapping that makes the prior positive
    prior_map_scale = .1  # output variance parameter
    prior_map_mean = .0
    prior_map_text = 'lambda x:  prior_map_mean + prior_map_scale*prior_map_parameter*torch.exp(x)'  # the copy of the mapping for future reference
    prior_map = lambda x: prior_map_mean + prior_map_scale * torch.exp(x)  # the actual mapping
    #nu_prior = obs_data['nu_prior']
    #prior_map_mean = obs_data['prior_map_mean']
    #prior_map_scale = obs_data['prior_map_scale']

    dtype = torch.float64
    graph = pickle.load(
        open('./data/covid_data/g.pkl', "rb"))  # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph)  # Initiating a graph Gaussian process
    v0 = torch.zeros(G.num_nodes, dtype=dtype)
    # v0 = 10*torch.ones(G.N, dtype=dtype)
    # v0 = 1+torch.rand(G.N, dtype=dtype)
    v0[25] = 1.
    #v0 = obs_data['v0']

    prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    y_obs = obs_data#['y_obs']
    y_obs = y_obs.reshape(51, 80)[:, 0]  # numpy array, shape (51,)
    #could take mean of 80 time points as well!
    #y_obs = y_obs.reshape(51, 80).mean(axis=1)  # (51,)
    #y_obs = y_true + sigma*noise_vec


    # creating a forward operator
    #graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    #G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process
    problem = forward_operator_stationary_nonlinearreaction(v0, G, nu_prior, prior_map, covid=covid) # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x: torch.sum( (problem.forward(x) - y_obs)**2/sigma2 ) + torch.sum( x**2 )

    x = torch.zeros(110, dtype=torch.float64) # initial guess for the sampler
    x.requires_grad_(True)

    y_obs = torch.as_tensor(y_obs, dtype=torch.float32, device=x.device)
    x_obs = torch.normal(torch.zeros(G.num_edges), torch.ones(G.num_edges)).to(dtype)
    # creating signal/observation with noise std defined in sigma with 1% noise level
    #y_true_torch = torch.from_numpy(y_true)
    #sigma = 0.01 * torch.linalg.norm(y_true_torch)
    sigma = 0.01 * np.linalg.norm(y_obs)
    #sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma

    sigma2 = torch.as_tensor(sigma2, dtype=torch.float32, device=x.device)

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

    stat_data = {'x_MAP': x_MAP}

    dir = './stat/stationary_nonlinearreaction_real_covid/'
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(dir, 'MAP.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plotting the true parameter
    f, axes = plt.subplots(1,2, figsize=[12,12])
    with open('stat_positions_exact.pickle' , 'rb') as handle:
        state_positions = pickle.load(handle)


    plotter = Graph_Plotter(graph, y_obs.detach().numpy(), axes[0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_obs.detach().numpy() ) # plotting the initial condition
    axes[0].set_title('measurement')
    axes[0].set_aspect('equal')


    temp = prior_map(x_obs).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    # plotting the MAP graph weights
    # plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[1,0], pos=state_positions) # initiating the graph plotter
    # plotter.plot_stationary(y_MAP.detach().numpy() ) # plotting the initial condition
    # axes[1,0].set_title('reconstructed measurement')
    # axes[1,0].set_aspect('equal')

    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1], vmin=vmin, vmax=vmax)
    axes[1].set_title('estimated graph weights')
    axes[1].set_aspect('equal')

    plt.show()


def MAP_stationary_nonlinearreaction():
    with open('obs/stationary_nonlinearreaction/obs.pickle', 'rb') as handle:
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
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process
    problem = forward_operator_stationary_nonlinearreaction(v0, G, nu_prior, prior_map) # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x: torch.sum( (problem.forward(x) - y_obs)**2/sigma2 ) + torch.sum( x**2 )

    x = torch.zeros(110, dtype=torch.float64) # initial guess for the sampler
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

    stat_data = {'x_MAP': x_MAP}

    dir = './stat/stationary_nonlinearreaction/'
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(dir, 'MAP.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plotting the true parameter
    f, axes = plt.subplots(2,2, figsize=[12,12])
    with open('stat_positions_exact.pickle' , 'rb') as handle:
        state_positions = pickle.load(handle)


    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0,0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true.detach().numpy() ) # plotting the initial condition
    axes[0,0].set_title('noise free measurement')
    axes[0,0].set_aspect('equal')

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0,0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[0,1])
    axes[0,1].set_title('true graph weights')
    axes[0,1].set_aspect('equal')

    temp = prior_map(x_true).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    # plotting the MAP graph weights
    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[1,0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_MAP.detach().numpy() ) # plotting the initial condition
    axes[1,0].set_title('reconstructed measurement')
    axes[1,0].set_aspect('equal')

    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[1,0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1,1], vmin=vmin, vmax=vmax)
    axes[1,1].set_title('estimated graph weights')
    axes[1,1].set_aspect('equal')

    plt.show()


def MAP_heat():
    with open('obs/heat/obs.pickle', 'rb') as handle:
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

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process
    problem = forward_operator_heat(v0, G, prior_map, nu_prior, dt_prior, T_prior) # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x, w: torch.sum( (problem.forward(x, w) - y_obs)**2/sigma2 ) + torch.sum( (x)**2 ) + torch.sum( w**2 )

    x = torch.zeros(110, dtype=torch.float64) # initial guess for the sampler
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
    with open('./stat_positions_exact.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    x_MAP = x
    w_noise_MAP = w_noise

    y_MAP = problem.forward(x_MAP, w_noise_MAP)
    stat_data = {'x_MAP': x_MAP, 'w_noise_MAP': w_noise_MAP}

    dir = './stat/heat/'
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(dir, 'MAP.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true[-1].detach().numpy() ) # plotting the initial condition
    anim1 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)
    axes[0].set_title('noise-free measurements')
    #axes[0].set_aspect('equal')

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[1])
    axes[1].set_title('true graph weights')
    #axes[1].set_aspect('equal')
    
    temp = prior_map(x_true).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    f, axes = plt.subplots(1,2, figsize=[12,6])

    # plotting the MAP graph weights
    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_MAP[-1].detach().numpy() ) # plotting the initial condition
    anim2 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_MAP.shape[0], interval=100)
    axes[0].set_title('reconstructed measurements')
    #axes[0].set_aspect('equal')

    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1])#, vmin=vmin, vmax=vmax)
    axes[0].set_title('estimated graph weights')
    #axes[0].set_aspect('equal')


    plt.show()


def MAP_heat_linearreaction_real_covid(covid=True):
    # covid=True activates clampint to [0,1]
    # ---------- helpers ----------
    def as_nodes_time(y, n_nodes=51, n_time=80):
        """
        Return y as torch.Tensor shaped (n_nodes, n_time).
        Accepts (n_nodes,n_time), (n_time,n_nodes), (steps,n_nodes), (n_nodes,steps), or flat (n_nodes*n_time,).
        If steps != n_time, this returns (n_nodes,steps) and you must subsample externally.
        """
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)

        if y.ndim == 2:
            # (nodes,time)
            if y.shape[0] == n_nodes:
                return y
            # (time,nodes) or (steps,nodes)
            if y.shape[1] == n_nodes:
                return y.T

        if y.ndim == 1 and y.numel() == n_nodes * n_time:
            return y.view(n_nodes, n_time)

        raise ValueError(f"Unexpected y shape: {tuple(y.shape)}")

    with open('data/covid_data/y_cases_normalized.pkl', 'rb') as handle:
        obs_data = pickle.load(handle)

    dtype = torch.float64
    n_nodes, n_time = 51, 80

    graph = pickle.load(
        open('./data/covid_data/g.pkl', "rb"))  # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph)  # Initiating a graph Gaussian process
    x_obs = torch.normal(torch.zeros(G.num_edges), torch.ones(G.num_edges)).to(dtype)
    y_obs = obs_data  # ['y_obs']
    y_obs = y_obs.reshape(51, 80)  # numpy array, shape (51,)

    x = torch.zeros(110, dtype=torch.float64) # initial guess for the sampler
    x.requires_grad_(True)

    y_obs = torch.as_tensor(y_obs, dtype=torch.float64, device=x.device)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01 * torch.linalg.norm(y_obs)
    sigma2 = sigma * sigma

    nu_prior = 1  # regularity
    # The non-linear mapping that makes the prior positive
    prior_map_scale = .1  # output variance parameter
    prior_map_mean = .0

    v0 = torch.randn(G.num_nodes, dtype=dtype)

    T_prior = 5.
    MAX_ITER_prior = n_time-1 #int(T_prior / dt_prior)  # number of iterations to hit maximum time
    dt_prior = T_prior/MAX_ITER_prior #~0.063

    prior_map = lambda x: prior_map_mean + prior_map_scale * torch.exp(x)

    w_noise_true = torch.randn(MAX_ITER_prior, v0.shape[0]).to(dtype)
    G.compute_laplacian_from_tensor_autograd(prior_map(x_obs), normalized=True)

    # creating a forward operator
    graph = pickle.load(
        open('./data/covid_data/g.pkl', "rb"))  # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph)  # Initiating a graph Gaussian process
    problem = forward_operator_heat_linearreaction(v0, G, prior_map, nu_prior, dt_prior, T_prior,covid=covid)  # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x, w: torch.sum((as_nodes_time(problem.forward(x, w), 51,80) - y_obs) ** 2 / sigma2) + torch.sum(
        (x) ** 2) + torch.sum(w ** 2)

    x = torch.zeros(110, dtype=torch.float64)  # initial guess for the sampler
    x.requires_grad_(True)
    w_noise = torch.zeros_like(w_noise_true).to(torch.float64)
    w_noise.requires_grad_(True)
    optimizer = LBFGS([x, w_noise], max_iter=5000, line_search_fn="strong_wolfe")

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
    f, axes = plt.subplots(1, 2, figsize=[12, 6])
    with open('./stat_positions_exact.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    x_MAP = x
    w_noise_MAP = w_noise
    y_MAP = problem.forward(x_MAP, w_noise_MAP)

    stat_data = {'x_MAP': x_MAP, 'w_noise_MAP': w_noise_MAP}

    dir = './stat/heat_linearreaction_real_covid/'
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(dir, 'MAP.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Ensure y_obs and y_MAP is (time, nodes)
    y_obs_plot = y_obs.detach().cpu().numpy().T
    y_MAP = problem.forward(x_MAP, w_noise_MAP)
    y_MAP_plot = y_MAP.detach().cpu().numpy() #80,51

    with open('./stat_positions_exact.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    # --- measurements animation ---
    plotter = Graph_Plotter(graph, y_obs_plot, axes[0],
                            pos=state_positions)  # initiating the graph plotter
    plotter.plot_stationary(y_obs_plot[-1])  # plotting the initial condition
    anim1 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_obs_plot.shape[0], interval=100)
    axes[0].set_title('measurements')

    temp = prior_map(x_obs).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    plotter = Graph_Plotter(graph, y_MAP_plot, axes[1], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1])  # , vmin=vmin, vmax=vmax)
    axes[1].set_title('estimated graph weights')

    plt.show()


def MAP_heat_linearreaction():
    with open('obs/heat_linearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    # x_smooth = obs_data['x_smooth']
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

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale * torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01 * torch.linalg.norm(y_true)
    sigma2 = sigma * sigma
    y_obs = y_true + sigma * noise_vec

    # creating a forward operator
    graph = pickle.load(
        open('./data/covid_data/g.pkl', "rb"))  # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph)  # Initiating a graph Gaussian process
    problem = forward_operator_heat_linearreaction(v0, G, prior_map, nu_prior, dt_prior, T_prior)  # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x, w: torch.sum((problem.forward(x, w) - y_obs) ** 2 / sigma2) + torch.sum(
        (x) ** 2) + torch.sum(w ** 2)

    x = torch.zeros(110, dtype=torch.float64)  # initial guess for the sampler
    x.requires_grad_(True)
    w_noise = torch.zeros_like(w_noise_true).to(torch.float64)
    w_noise.requires_grad_(True)
    optimizer = LBFGS([x, w_noise], max_iter=5000, line_search_fn="strong_wolfe")

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
    f, axes = plt.subplots(1, 2, figsize=[12, 6])
    with open('./stat_positions_exact.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    x_MAP = x
    w_noise_MAP = w_noise
    y_MAP = problem.forward(x_MAP, w_noise_MAP)

    stat_data = {'x_MAP': x_MAP, 'w_noise_MAP': w_noise_MAP}

    dir = './stat/heat_linearreaction/'
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(dir, 'MAP.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0],
                            pos=state_positions)  # initiating the graph plotter
    plotter.plot_stationary(y_true[-1].detach().numpy())  # plotting the initial condition
    anim1 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)
    axes[0].set_title('noise-free measurements')
    # axes[0].set_aspect('equal')

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[1])
    axes[1].set_title('true graph weights')
    # axes[1].set_aspect('equal')

    temp = prior_map(x_true).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    f, axes = plt.subplots(1, 2, figsize=[12, 6])

    # plotting the MAP graph weights
    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[0], pos=state_positions)  # initiating the graph plotter
    plotter.plot_stationary(y_MAP[-1].detach().numpy())  # plotting the initial condition
    anim2 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_MAP.shape[0], interval=100)
    axes[0].set_title('reconstructed measurements')
    # axes[0].set_aspect('equal')

    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1])  # , vmin=vmin, vmax=vmax)
    axes[0].set_title('estimated graph weights')
    # axes[0].set_aspect('equal')

    plt.show()


def MAP_heat_nonlinearreaction_real_covid(covid=True):
    # covid=True activates clampint to [0,1]
    # ---------- helpers ----------
    def as_nodes_time(y, n_nodes=51, n_time=80):
        """
        Return y as torch.Tensor shaped (n_nodes, n_time).
        Accepts (n_nodes,n_time), (n_time,n_nodes), (steps,n_nodes), (n_nodes,steps), or flat (n_nodes*n_time,).
        If steps != n_time, this returns (n_nodes,steps) and you must subsample externally.
        """
        if not torch.is_tensor(y):
            y = torch.as_tensor(y)

        if y.ndim == 2:
            # (nodes,time)
            if y.shape[0] == n_nodes:
                return y
            # (time,nodes) or (steps,nodes)
            if y.shape[1] == n_nodes:
                return y.T

        if y.ndim == 1 and y.numel() == n_nodes * n_time:
            return y.view(n_nodes, n_time)

        raise ValueError(f"Unexpected y shape: {tuple(y.shape)}")

    with open('data/covid_data/y_cases_normalized.pkl', 'rb') as handle:
        obs_data = pickle.load(handle)

    dtype = torch.float64
    n_nodes, n_time = 51, 80

    graph = pickle.load(
        open('./data/covid_data/g.pkl', "rb"))  # the graph containing the nodal information and the connections

    G = Matern_graph_pytorch(graph)  # Initiating a graph Gaussian process
    x_obs = torch.normal(torch.zeros(G.num_edges), torch.ones(G.num_edges)).to(dtype)
    y_obs = obs_data
    y_obs = y_obs.reshape(n_nodes, n_time)  # numpy array, shape (51,)

    x = torch.zeros(110, dtype=torch.float64)  # initial guess for the sampler
    x.requires_grad_(True)

    y_obs = torch.as_tensor(y_obs, dtype=torch.float64, device=x.device)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01 * torch.linalg.norm(y_obs)
    sigma2 = sigma * sigma

    nu_prior = 1  # regularity
    # The non-linear mapping that makes the prior positive
    prior_map_scale = .1  # output variance parameter
    prior_map_mean = .0

    v0 = torch.randn(G.num_nodes, dtype=dtype)

    T_prior = 5.
    MAX_ITER_prior = n_time - 1  # int(T_prior / dt_prior)  # number of iterations to hit maximum time
    dt_prior = T_prior / MAX_ITER_prior

    prior_map = lambda x: prior_map_mean + prior_map_scale * torch.exp(x)

    w_noise_true = torch.randn(MAX_ITER_prior, v0.shape[0]).to(dtype)
    G.compute_laplacian_from_tensor_autograd(prior_map(x_obs), normalized=True)

    # creating a forward operator
    graph = pickle.load(
        open('./data/covid_data/g.pkl', "rb"))  # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph)  # Initiating a graph Gaussian process
    problem = forward_operator_heat_nonlinearreaction(v0, G, prior_map, nu_prior, dt_prior,
                                                   T_prior,covid=covid)  # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x, w: torch.sum(
        (as_nodes_time(problem.forward(x, w), 51, 80) - y_obs) ** 2 / sigma2) + torch.sum(
        (x) ** 2) + torch.sum(w ** 2)

    x = torch.zeros(110, dtype=torch.float64)  # initial guess for the sampler
    x.requires_grad_(True)
    w_noise = torch.zeros_like(w_noise_true).to(torch.float64)
    w_noise.requires_grad_(True)
    optimizer = LBFGS([x, w_noise], max_iter=5000, line_search_fn="strong_wolfe")

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
    f, axes = plt.subplots(1, 2, figsize=[12, 6])
    with open('./stat_positions_exact.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    x_MAP = x
    w_noise_MAP = w_noise
    y_MAP = problem.forward(x_MAP, w_noise_MAP)

    stat_data = {'x_MAP': x_MAP, 'w_noise_MAP': w_noise_MAP}

    dir = './stat/heat_nonlinearreaction_real_covid/'
    if not os.path.exists(dir):
        os.makedirs(dir)

    with open(os.path.join(dir, 'MAP.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # Ensure y_obs and y_MAP is (time, nodes)
    y_obs_plot = y_obs.detach().cpu().numpy().T
    y_MAP_plot = y_MAP.detach().cpu().numpy()  # 80,51

    with open('./stat_positions_exact.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    # --- measurements animation ---
    plotter = Graph_Plotter(graph, y_obs_plot, axes[0],
                            pos=state_positions)  # initiating the graph plotter
    plotter.plot_stationary(y_obs_plot[-1])  # plotting the initial condition
    anim1 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_obs_plot.shape[0], interval=100)
    axes[0].set_title('measurements')

    temp = prior_map(x_obs).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    plotter = Graph_Plotter(graph, y_MAP_plot, axes[1], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1])  # , vmin=vmin, vmax=vmax)
    axes[1].set_title('estimated graph weights')

    plt.show()


def MAP_heat_nonlinearreaction():
    with open('obs/heat_nonlinearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    # x_smooth = obs_data['x_smooth']
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

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale * torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01 * torch.linalg.norm(y_true)
    sigma2 = sigma * sigma
    y_obs = y_true + sigma * noise_vec

    # creating a forward operator
    graph = pickle.load(
        open('./data/covid_data/g.pkl', "rb"))  # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph)  # Initiating a graph Gaussian process
    problem = forward_operator_heat_nonlinearreaction(v0, G, prior_map, nu_prior, dt_prior, T_prior)  # creating a forward operator

    # defining the log-posterior with a standard normal Gaussian prior
    negative_log_posterior = lambda x, w: torch.sum((problem.forward(x, w) - y_obs) ** 2 / sigma2) + torch.sum(
        (x) ** 2) + torch.sum(w ** 2)

    x = torch.zeros(110, dtype=torch.float64)  # initial guess for the sampler
    x.requires_grad_(True)
    w_noise = torch.zeros_like(w_noise_true).to(torch.float64)
    w_noise.requires_grad_(True)
    optimizer = LBFGS([x, w_noise], max_iter=5000, line_search_fn="strong_wolfe")

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
    f, axes = plt.subplots(1, 2, figsize=[12, 6])
    with open('./stat_positions_exact.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    x_MAP = x
    w_noise_MAP = w_noise
    y_MAP = problem.forward(x_MAP, w_noise_MAP)

    stat_data = {'x_MAP': x_MAP, 'w_noise_MAP': w_noise_MAP}

    dir = './stat/heat_nonlinearreaction/'
    if not os.path.exists(dir):
            os.makedirs(dir)

    with open(os.path.join(dir, 'MAP.pickle'), 'wb') as handle:
        pickle.dump(stat_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0],
                            pos=state_positions)  # initiating the graph plotter
    plotter.plot_stationary(y_true[-1].detach().numpy())  # plotting the initial condition
    anim1 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)
    axes[0].set_title('noise-free measurements')
    # axes[0].set_aspect('equal')

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[1])
    axes[1].set_title('true graph weights')
    # axes[1].set_aspect('equal')

    temp = prior_map(x_true).detach().numpy()
    vmin = np.min(temp)
    vmax = np.max(temp)

    f, axes = plt.subplots(1, 2, figsize=[12, 6])

    # plotting the MAP graph weights
    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[0], pos=state_positions)  # initiating the graph plotter
    plotter.plot_stationary(y_MAP[-1].detach().numpy())  # plotting the initial condition
    anim2 = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_MAP.shape[0], interval=100)
    axes[0].set_title('reconstructed measurements')
    # axes[0].set_aspect('equal')

    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), axes[1])  # , vmin=vmin, vmax=vmax)
    axes[0].set_title('estimated graph weights')
    # axes[0].set_aspect('equal')

    plt.show()


if __name__ == '__main__':
    #MAP_stationary()
    MAP_stationary_linearreaction()
    #MAP_stationary_nonlinearreaction()
    #MAP_heat()
    #MAP_heat_linearreaction()
    #MAP_heat_nonlinearreaction()

    #MAP_stationary_linearreaction_real_covid()
    #MAP_stationary_nonlinearreaction_real_covid()
    #MAP_heat_linearreaction_real_covid()
    #MAP_heat_nonlinearreaction_real_covid()