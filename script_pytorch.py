import torch
import pickle
from Matern_prior import Matern_graph_pytorch, Graph_Plotter
import matplotlib.pyplot as plt
from matplotlib import animation
# Class for creating a Laplacian operator on a circular graph
# The code is inspired by the work "Non-separable Spatio-temporal Graph Kernels via 
# SPDEs" (2022) by Nikitin et al.

# loading the graph of the US states
graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

# defining a Matern prior on the graph
G = Matern_graph_pytorch(graph, normalize=True)

# defining an outbreak in the first state (Alabama)
v0 = torch.zeros(G.N, dtype=torch.float64)
v0[0] = 10.

# solving the dynamics following the heat equation
out = G.sample_heat(nu = 1, v0=v0, dt=0.1)
out = out.detach().numpy()

# plotting the dynamics
f, ax = plt.subplots()
plotter = Graph_Plotter(graph, out, ax) # initiating the graph plotter
plotter.plot_stationary(out[0]) # plotting the initial condition
anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=out.shape[0], interval=10) # animating the dynamics

plt.show()
