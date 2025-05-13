import numpy as np
import torch
from torch.optim import LBFGS
import pickle
import matplotlib.pyplot as plt
from Matern_prior import Matern_graph_pytorch, Graph_Plotter



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

        self.G.update_L( 0.1*torch.exp(p), normalize=True ) # here we create a (non-linear) log-Gaussian prior
        return self.G.sample_stationary(self.v0)

def estimate_MAP():
    """
    This function will sample from the posterior distribution constructed by a Grap Gaussian process.
    This function considers an observation file './obs/obs_stationary.pickle' with components:
    
    y_true: noise free observation
    noise_vec: a normalized noise vector follosing a Gaussian distribution
    x_true: (not needed for sampling) the true solution to the inverse problem
    """

    # loading the observation vector
    with open('./obs/stationary/reg1/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    y_true =  torch.from_numpy( obs_data['y_true'] ).to(torch.float64)
    noise_vec = torch.from_numpy( obs_data['noise_vec'] ).to(torch.float64)
    x_true = torch.from_numpy( obs_data['x_true'] ).to(torch.float64)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)/torch.sqrt( torch.tensor( y_true.shape[0] ) )
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph, normalize=True) # Initiating a graph Gaussian process
    v0 = torch.zeros(G.N, dtype=torch.float64) # defining the initial condition (source term) of the stationary Gaussian process
    v0[25] = 10. # Outbreak of value 10 at the 26th state of the US
    problem = forward_operator(v0, G) # creating a forward operator  

    # defining the log-posterior with a standard normal Gaussian prior
    log_posterior = lambda x: -0.5*torch.sum( (problem.forward(x) - y_obs)**2/sigma2 ) - 0.5*torch.sum( x**2 )

    x = torch.zeros(G.num_edges, dtype=torch.float64) # initial guess for the sampler
    x.requires_grad_(True)
    optimizer = LBFGS( [x], max_iter=1000, line_search_fn="strong_wolfe" )

    def closure():
        optimizer.zero_grad()

        # negative log posterior
        loss = -log_posterior(x)

        # computing the gradient
        loss.backward()
        print(loss)
        return loss

    optimizer.step(closure)

    # First we plot the true solution with x_true
    out_true = problem.forward(x_true )
    f, ax = plt.subplots()
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    plotter = Graph_Plotter(graph, out_true.detach().numpy(), ax, pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(out_true.detach().numpy()) # plotting the initial condition
    plotter.draw_labels()
    ax.set_title('exact signal')

    # We now plot the mean solution
    out_est = problem.forward(x)
    # plotting the dynamics
    f, ax = plt.subplots()
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    plotter = Graph_Plotter(graph, out_est.detach().numpy(), ax, pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(out_est.detach().numpy()) # plotting the initial condition
    plotter.draw_labels()
    #anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=out.shape[0], interval=10) # animating the dynamics
    ax.set_title('estimated signal with the MAP graph weights')

    f,ax = plt.subplots(1)
    plotter = Graph_Plotter(graph, out_est.detach().numpy(), ax, pos=state_positions)
    plotter.plot_graph_wieghts(out_est.detach().numpy(), (torch.abs(x - x_true) ).detach().numpy(), ax)
    plt.show()

if __name__ == '__main__':
    estimate_MAP()
