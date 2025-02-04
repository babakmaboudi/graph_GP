import numpy as np
import networkx as nx
import pickle
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.animation as animation

from Matern_prior import Matern_graph, graph_plotter 

# loading the graph of the US states
graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

# defining a Matern prior on the graph
G = Matern_graph(graph)

# defining an outbreak in the first state (Alabama)
v0 = np.zeros(G.N)
v0[0] = 1000.

# solving the dynamics following the heat equation
out = G.sample_heat(v0, nu = 1)

# plotting the dynamics
f, ax = plt.subplots()
plotter = graph_plotter(graph, out, ax) # initiating the graph plotter
plotter.plot_stationary(v0) # plotting the initial condition
anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=out.shape[0], interval=10) # animating the dynamics

plt.show()
