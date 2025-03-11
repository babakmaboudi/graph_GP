import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Matern_prior import Matern_graph, Graph_Plotter

import Matern_prior

class forward_operator():
    def __init__(self, v0, G):
        self.v0 = v0
        self.G = G
    def forward(self, p):
        self.G.update_L( 0.1*np.exp(p), normalize=True )
        return self.G.sample_stationary(self.v0)

if __name__ == '__main__':
    with open('./obs/obs_stationary.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    x_true = obs_data['x_true']


    data = np.load('./stat/stats.npz')
    samples = data['samples']

    samples = samples[1000000::100,:]
    
    mean = np.mean(samples, axis=0)


    # loading the graph of the US states
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))
    # defining a Matern prior on the graph
    G = Matern_graph(graph, normalize=True)
    v0 = np.zeros(G.N)
    v0[25] = 1.
    problem = forward_operator(v0, G)

    out_true = problem.forward(x_true)

    # plotting the dynamics
    f, ax = plt.subplots()
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    plotter = Graph_Plotter(graph, out_true, ax, pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(out_true) # plotting the initial condition
    #anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=out.shape[0], interval=10) # animating the dynamics

    out_est = problem.forward(mean)
    # plotting the dynamics
    f, ax = plt.subplots()
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    plotter = Graph_Plotter(graph, out_est, ax, pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(out_est) # plotting the initial condition
    #anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=out.shape[0], interval=10) # animating the dynamics
    plt.show()
