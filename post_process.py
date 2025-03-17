import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Matern_prior import Matern_graph, Graph_Plotter
from inverse import forward_operator

# loading the observation file
with open('./obs/stationary/reg1/obs.pickle', 'rb') as handle:
    obs_data = pickle.load(handle)

y_true = obs_data['y_true']
noise_vec = obs_data['noise_vec']
x_true = obs_data['x_true']


# loading samples and pre-processing samples
data = np.load('./stat/stationary/reg1/stat.npz')
samples = data['samples']
samples = samples[1000000::100,:] # burning the first 1e6 samples and sub-sampling every 100 samples

# cearting the mean of the samples
mean = np.mean(samples, axis=0)

# creating the trace plot of the first 10 graph weights
f,ax = plt.subplots()
for i in range(10):
    plt.plot(samples[:,i])
ax.set_title('trace plot')

# Visualizing the solution: For this we set the graph weights to the mean and solve the graph GP to see the nodal distribution (disease spread)
# loading the graph of the US states

# First we plot the true solution with x_true
graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))
G = Matern_graph(graph, normalize=True)
v0 = np.zeros(G.N)
v0[25] = 10.
problem = forward_operator(v0, G)
out_true = problem.forward(x_true)
f, ax = plt.subplots()
with open('./stat_positions.pickle', 'rb') as handle:
    state_positions = pickle.load(handle)
plotter = Graph_Plotter(graph, out_true, ax, pos=state_positions) # initiating the graph plotter
plotter.plot_stationary(out_true) # plotting the initial condition
ax.set_title('exact signal')

# We now plot the mean solution
out_est = problem.forward(mean)
# plotting the dynamics
f, ax = plt.subplots()
with open('./stat_positions.pickle', 'rb') as handle:
    state_positions = pickle.load(handle)
plotter = Graph_Plotter(graph, out_est, ax, pos=state_positions) # initiating the graph plotter
plotter.plot_stationary(out_est) # plotting the initial condition
#anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=out.shape[0], interval=10) # animating the dynamics
ax.set_title('estimated signal with the mean graph weights')
plt.show()
