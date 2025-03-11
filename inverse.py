import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Matern_prior import Matern_graph, Graph_Plotter

import Matern_prior
import sampler

def create_observation_old():
    # loading the graph of the US states
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

    # defining a Matern prior on the graph
    G = Matern_graph(graph, normalize=True)

    # defining an outbreak in the first state (Alabama)
    v0 = np.zeros(G.N)
    v0[31] = 10.

    x_true = np.random.standard_normal(G.num_edges)
    G.update_L( np.exp(x_true) )

    #y_true = G.sample_heat(nu = 1, v0=v0, dt=0.05)
    y_true = G.sample_stationary(v0)

    y_true = y_true.reshape(-1)
    
    noise_vec = np.random.standard_normal(y_true.shape)
    noise_vec = noise_vec/np.linalg.norm(noise_vec)

    sigma = 0.5*np.linalg.norm(y_true)/np.sqrt(51)
    y_obs = y_true + sigma*noise_vec

    f,ax = plt.subplots(1)
    ax.plot(y_true,'b')
    ax.plot(y_obs,'r')

    obs_data = {'x_true': x_true, 'y_true': y_true, 'noise_vec': noise_vec}

    with open('./obs/obs_stationary.pickle', 'wb') as handle:
        pickle.dump(obs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

class forward_operator():
    def __init__(self, v0, G):
        self.v0 = v0
        self.G = G
    def forward(self, p):
        self.G.update_L( 0.1*np.exp(p), normalize=True )
        return self.G.sample_stationary(self.v0)


def sample_posterior_old():
    with open('./obs/obs_stationary.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    x_true = obs_data['x_true']

    # creating signal/observation with noise std defined in sigma
    sigma = 0.5*np.linalg.norm(y_true)/np.sqrt(y_true.shape[0])
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))
    G = Matern_graph(graph, normalize=True)
    v0 = np.zeros(G.N)
    v0[31] = 10.

    problem = forward_operator(v0, G)

    # defining the log likelihood
    log_likelihood = lambda x: -0.5*np.sum( (problem.forward(x) - y_obs)**2/sigma2 ) - 0.5*np.sum( x**2 )

    # defining the pre-conditioned Crank-Nickolson sampler
    x0 = np.ones( G.num_edges ) 
    pCN = sampler.MH(x0, log_likelihood)
    pCN.scale = 0.002

    num_samples_warm_up = 10000
    num_samples = 200000
    # sampling stage
    print('warm up')
    pCN.warm_up(num_samples_warm_up, skip_len=100)
    print('step size is {}'.format(pCN.scale))

    print('sampling')
    pCN.sample(num_samples)

    samples = pCN.get_samples()
    print(samples.shape)
    print('acceptance rate is: ', np.mean( np.array(pCN.acc) ))

    mean = np.mean(samples[-10000:,:],axis=0)

    print(mean)
    print(x_true)

def sample_posterior():
    with open('./obs/obs_stationary.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    x_true = obs_data['x_true']

    # creating signal/observation with noise std defined in sigma
    sigma = 0.01*np.linalg.norm(y_true)/np.sqrt(y_true.shape[0])
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))
    G = Matern_graph(graph, normalize=True)
    v0 = np.zeros(G.N)
    v0[25] = 10.

    problem = forward_operator(v0, G)

    # defining the log-posterior
    log_posterior = lambda x: -0.5*np.sum( (problem.forward(x) - y_obs)**2/sigma2 ) - 0.5*np.sum( x**2 )

    # defining the sampler
    x0 = np.zeros(G.num_edges) # initial guess for the sampler
    MH = sampler.MH(x0, log_posterior)
    #MH.scale = 0.05
    #print('warm_up')
    #MH.warm_up(10000, skip_len=100)
    #print(MH.scale)
    MH.scale = 0.007
    MH.warm_up(2000000)
    print('scale is : ',  MH.scale)
    #MH.sample(5000000)
    #print('acceptance rate is: ', np.mean( np.array(MH.acc) ))
    samples = MH.get_samples()
    #samples = samples[4000000::1000,:]
    mean = np.mean(samples, axis=0)
    std = np.std(samples, axis=0)

    np.savez('./stat/stats.npz',samples=samples)

    f,ax = plt.subplots()
    for i in range(5):
        ax.plot(samples[:,-i-1])

    f,ax = plt.subplots()
    ax.plot(x_true)
    ax.plot(mean)

    f,ax = plt.subplots()
    ax.plot(std)

    print(mean)
    print(x_true)
    plt.show()

    
    #p = np.random.standard_normal(G.num_edges)
    #out = problem.forward(p)

    #f, ax = plt.subplots()
    #plt.plot(y_true)
    #plt.plot(y_obs)
    #plt.plot(out)

    #print(out)
    #f, ax = plt.subplots()
    #with open('./stat_positions.pickle', 'rb') as handle:
    #    state_positions = pickle.load(handle)
    #plotter = Graph_Plotter(graph, out, ax, pos=state_positions) # initiating the graph plotter
    #plotter.plot_stationary(out) # plotting the initial condition
    #plt.show()





def test():
    # loading the graph of the US states
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

    # defining a Matern prior on the graph
    G = Matern_graph(graph, normalize=True)

    # defining an outbreak in the first state (Alabama)
    v0 = np.zeros(G.N)
    v0[25] = 10.

    # solving the dynamics following the heat equation

    x_true = np.random.standard_normal(G.num_edges)
    G.update_L( 0.1*np.exp(x_true) , normalize=True)
    #out = G.sample_heat(nu = 1, v0=v0, dt=0.01)
    y_true = G.sample_stationary(v0)
    print(y_true)

    noise_vec = np.random.standard_normal(y_true.shape)
    noise_vec = noise_vec/np.linalg.norm(noise_vec)

    sigma = 0.01*np.linalg.norm(y_true)/np.sqrt(y_true.shape)
    y_obs = y_true + sigma*noise_vec

    f,ax = plt.subplots()
    ax.plot(y_true)
    ax.plot(y_obs)


    # plotting the dynamics
    f, ax = plt.subplots()
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    plotter = Graph_Plotter(graph, y_true, ax, pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true) # plotting the initial condition
    #anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=out.shape[0], interval=10) # animating the dynamics

    obs_data = {'x_true': x_true, 'y_true': y_true, 'noise_vec': noise_vec}

    #with open('./obs/obs_stationary.pickle', 'wb') as handle:
    #    pickle.dump(obs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.show()
    

if(__name__ == "__main__"):
    #create_observation()
    #sample_posterior()
    test()
