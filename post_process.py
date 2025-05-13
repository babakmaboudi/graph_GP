import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Matern_prior import Matern_graph, Graph_Plotter
from inverse import forward_operator
import corner
import math

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
# f,ax = plt.subplots()
# for i in range(10):
#     plt.plot(samples[:,i])
# ax.set_title('trace plot')

# Load state positions from pickle file
with open('./stat_positions.pickle', 'rb') as handle:
    state_positions = pickle.load(handle)

# Create a random subset of states (indices)
num_traces = 9
random_indices = np.random.choice(len(state_positions), num_traces, replace=False)

# Plot the traces
f, ax = plt.subplots()

# Convert the dictionary keys to a list for indexing
state_names = list(state_positions.keys())
target = {'illinois', 'iowa', 'kansas', 'nebraska', 'oklahoma', 'arkansas', 'tennessee', 'kentucky', 'missouri'}
state_indices = [i for i, s in enumerate(state_names) if s in target]

ax.plot(samples[:, 25], label=state_names[25])  # Use state name for label

for i in random_indices:
    ax.plot(samples[:, i], label=state_names[i])  # Use state name for label

# Add legend with state names
ax.legend(title='States')

# Set title and display the plot
ax.set_title('Trace Plot')
plt.show()

#Corner plot:
# Subset the samples to only the selected random traces
# To append index 25 to it (using np.append) Initial state
#random_indices = np.append(random_indices, 25)
#subset_samples = samples[:, random_indices]
subset_samples = samples[:, state_indices]

# Create the corner plot
cl = 0.95
fig = corner.corner(subset_samples, labels=[list(state_positions.keys())[i] for i in random_indices],
                       range=[(min_val, max_val)
                              for min_val, max_val in zip(subset_samples.min(axis=0),
                                                          subset_samples.max(axis=0))],
                    title_quantiles=[(1 - cl) / 2, 0.5, 1 - (1 - cl) / 2],
                    quantiles=[(1 - cl) / 2, 0.5, 1 - (1 - cl) / 2],
                    show_titles=True)
axes = np.array(fig.axes).reshape((len(random_indices), len(random_indices)))
for i in range(len(random_indices)):
    for j in range(len(random_indices)):
        ax = axes[i, j]
        if i == j:
            # Plot the MAP line
            ax.axvline(mean[i], color="cornflowerblue", label="Mean")
        elif i > j:
            # Plot the mean lines
            ax.axvline(x=mean[j], color="cornflowerblue", label="Mean")
            ax.axhline(y=mean[i], color="cornflowerblue", label="Mean")

        if i == len(random_indices) - 1 and j == len(random_indices)-1:  # Add legend only once, for the last subplot
            ax.legend(loc='best')
# Show the plot
plt.show()

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
plotter.draw_labels()
ax.set_title('exact signal')

# We now plot the mean solution
out_est = problem.forward(mean)
# plotting the dynamics
f, ax = plt.subplots()
with open('./stat_positions.pickle', 'rb') as handle:
    state_positions = pickle.load(handle)
plotter = Graph_Plotter(graph, out_est, ax, pos=state_positions) # initiating the graph plotter
plotter.plot_stationary(out_est) # plotting the initial condition
plotter.draw_labels()
#anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=out.shape[0], interval=10) # animating the dynamics
ax.set_title('estimated signal with the mean graph weights')
plt.show()
