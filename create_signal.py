import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Matern_prior import Matern_graph, Graph_Plotter

def create_signal():
    # loading the graph of the US states
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

    # defining a Matern prior on the graph
    G = Matern_graph(graph, normalize=True)

    # defining an outbreak in the first state (Alabama)
    v0 = np.zeros(G.N)
    v0[25] = 10.

    # creating a graph plotter object

    # plotting 5 prior samples:
    # loading a fix location of the states
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    
    num_prior_samples = 5
    f, axes = plt.subplots(1,num_prior_samples, figsize=[15,3])

    for i in range(num_prior_samples):
        p = np.random.standard_normal(G.num_edges)
        G.update_L( 0.1*np.exp(p) )
        out = G.sample_stationary(v0)

        #f, ax = plt.subplots()
        with open('./stat_positions.pickle', 'rb') as handle:
            state_positions = pickle.load(handle)
        plotter = Graph_Plotter(graph, out, axes[i], pos=state_positions) # initiating the graph plotter
        plotter.plot_stationary(out) # plotting the initial condition

    plt.tight_layout()
    plt.savefig('./obs/stationary/reg1/prior.pdf')

        



    # create a signal
    x_true = np.random.standard_normal(G.num_edges) # the unknown for the inverse problem
    G.update_L( 0.1*np.exp(x_true) , normalize=True) # updating the graph weights accroding to the unknown
    y_true = G.sample_stationary(v0) # creating a noise free signal

    # creating a normalized noise vector
    noise_vec = np.random.standard_normal(y_true.shape)
    noise_vec = noise_vec/np.linalg.norm(noise_vec) # normalizing accroding to the l2 norm of the noise

    # visualizeing the noisy measurement with 1% noise
    sigma = 0.01*np.linalg.norm(y_true)/np.sqrt(y_true.shape)
    y_obs = y_true + sigma*noise_vec

    # plotting the noisy measurement
    f,ax = plt.subplots()
    ax.plot(y_true)
    ax.plot(y_obs)
    ax.legend('noise free signal', 'noisy signal')
    plt.savefig('./obs/stationary/reg1/signal.pdf')

    # plotting the true parameter
    f, ax = plt.subplots()
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    plotter = Graph_Plotter(graph, y_true, ax, pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true) # plotting the initial condition
    plt.savefig('./obs/stationary/reg1/noise_free_signal.pdf')


    # saving the signal file
    obs_data = {'x_true': x_true, 'y_true': y_true, 'noise_vec': noise_vec}


    with open('./obs/stationary/reg1/obs.pickle', 'wb') as handle:
        pickle.dump(obs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.show()
    
if __name__ == '__main__':
    create_signal()


