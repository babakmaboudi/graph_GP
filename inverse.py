import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Matern_prior import Matern_graph, Graph_Plotter

import Matern_prior
import sampler


class forward_operator():
    """
    This class creates a forward operator for the Graph stationary Gaussian process.
    To initiate it takes the initial conidition (source term) of the stationary 
    Gaussian process and a Graph Gaussian process.

    The forward operato takes the graph weights and updates the Laplacian operator 
    of the Graph GP and then solves the stationary graph GP for the sources term 
    in the right-hand-side.
    """
    def __init__(self, v0, G):
        self.v0 = v0
        self.G = G
    def forward(self, p):
        # This function takes graph weights p and solves the stationary graph 
        # GP for source term v0
        self.G.update_L( 0.1*np.exp(p), normalize=True ) # here we create a (non-linear) log-Gaussian prior
        return self.G.sample_stationary(self.v0)

def sample_posterior():
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

    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    x_true = obs_data['x_true']

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*np.linalg.norm(y_true)/np.sqrt(y_true.shape[0])
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph(graph, normalize=True) # Initiating a graph Gaussian process
    v0 = np.zeros(G.N) # defining the initial condition (source term) of the stationary Gaussian process
    v0[25] = 10. # Outbreak of value 10 at the 26th state of the US
    problem = forward_operator(v0, G) # creating a forward operator  

    # defining the log-posterior with a standard normal Gaussian prior
    log_posterior = lambda x: -0.5*np.sum( (problem.forward(x) - y_obs)**2/sigma2 ) - 0.5*np.sum( x**2 )

    # defining the sampler
    x0 = np.zeros(G.num_edges) # initial guess for the sampler
    MH = sampler.MH(x0, log_posterior) # defining the Metropolis-Hastings sampling method
    
    
    MH.scale = 0.007 # the step size for the Metropilis-Hastings algorithm
    MH.warm_up(2000000) # warm-up and adapting step-size tuning
    print('scale is : ',  MH.scale) # printing the step-size
    
    # uncomment the following lines for sampling with a constant step size
    #MH.sample(5000000)
    #print('acceptance rate is: ', np.mean( np.array(MH.acc) ))
    
    # extracting and saving the samples
    samples = MH.get_samples()

    np.savez('./stat/stationary/reg1/stat.npz',samples=samples)

    # uncomment to compute the mean and the std
    #mean = np.mean(samples, axis=0)
    #std = np.std(samples, axis=0)

    # uncomment for plotting the traces
    #f,ax = plt.subplots()
    #for i in range(5):
    #    ax.plot(samples[:,-i-1])

    # uncomment to plot the mean
    #f,ax = plt.subplots()
    #ax.plot(x_true)
    #ax.plot(mean)

    # uncomment to plot the std
    #f,ax = plt.subplots()
    #ax.plot(std)

if(__name__ == "__main__"):
    sample_posterior()
