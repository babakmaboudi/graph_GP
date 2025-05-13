import numpy as np
import torch
from torch.optim import LBFGS
import pickle
import matplotlib.pyplot as plt
from Matern_prior import Matern_graph_pytorch, Graph_Plotter
from matplotlib import animation



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

def estimate_MAP():
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
    log_posterior = lambda x, w: -0.5*torch.sum( (problem.forward(x, w, nu=nu, dt=dt, T=T) - y_obs)**2/sigma2 ) - 0.5*torch.sum( (x)**2 ) - 0.5*torch.sum( w**2 )

    x = torch.zeros(G.num_edges, dtype=dtype) # initial guess for the sampler
    x.requires_grad_(True)
    w_noise = torch.zeros_like( w_noise_true ).to(dtype)
    w_noise.requires_grad_(True)
    optimizer = LBFGS( [x, w_noise], max_iter=5000, line_search_fn="strong_wolfe" )

    def closure():
        optimizer.zero_grad()

        # negative log posterior
        loss = -log_posterior(x, w_noise)

        # computing the gradient
        loss.backward()
        print(loss)
        return loss

    optimizer.step(closure)

    MAP_data = {'x_MAP': x, 'w_MAP': w_noise}

    with open('./stat/heat/reg1/MAP.pickle', 'wb') as handle:
        pickle.dump(MAP_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    print(torch.abs(x-x_true)/x_true )
    y_MAP = problem.forward(x, w_noise, nu=nu, dt=dt, T=T)

    x_prior = torch.randn_like(x)
    w_prior = torch.randn_like(w_noise) 
    y_prior = problem.forward(x_prior, w_prior, nu=nu, dt=dt, T=T)



    # plotting the true parameter
    #f, ax = plt.subplots()
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    #plotter = Graph_Plotter(graph, y_true.detach().numpy(), ax, pos=state_positions) # initiating the graph plotter
    #plotter.plot_stationary(y_true[0].detach().numpy()) # plotting the initial condition
    #anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)

    # plotting the true parameter
    #f, ax = plt.subplots()
    #with open('./stat_positions.pickle', 'rb') as handle:
    #    state_positions = pickle.load(handle)

    #y_MAP = problem.forward( x, w_noise )

    #plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), ax, pos=state_positions) # initiating the graph plotter
    #plotter.plot_stationary(y_MAP[0].detach().numpy()) # plotting the initial condition
    #anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)

    f,ax = plt.subplots(1)
    plotter = Graph_Plotter(graph, y_MAP.detach().numpy(), ax, pos=state_positions)
    plotter.plot_graph_wieghts(y_MAP.detach().numpy(), (torch.abs(x - x_true) ).detach().numpy(), ax)
    
    
    f,axes = plt.subplots(1,4)
    axes[0].imshow(y_true.detach().numpy())
    axes[0].set_title('true')
    axes[1].imshow(y_obs.detach().numpy())
    axes[1].set_title('noisey')
    axes[2].imshow(y_MAP.detach().numpy())
    axes[2].set_title('MAP')
    axes[3].imshow(y_prior.detach().numpy())
    axes[3].set_title('sample from prior')


    f,ax = plt.subplots(1)
    ax.imshow( torch.abs( (y_true - y_MAP)/(y_true) ).detach().numpy() )
    plt.show()

if __name__ == '__main__':
    estimate_MAP()
