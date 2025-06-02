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

# Visualizing the solution: For this we set the graph weights to the mean and solve the graph GP to see the nodal distribution (disease spread)
# loading the graph of the US states
# First we plot the true solution with x_true
graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))
G = Matern_graph(graph, normalize=True)
v0 = np.zeros(G.N)
v0[25] = 10.
problem = forward_operator(v0, G)
out_true = problem.forward(x_true)

out_est = problem.forward(mean)
with open('./stat_positions.pickle', 'rb') as handle:
    state_positions = pickle.load(handle)



def plot_trace(samples, state_positions, selected_indices=None):
    """
    Plot trace plots for a subset of already selected state indices.
    """
    f, ax = plt.subplots()
    state_names = list(state_positions.keys())
    for i in range(samples.shape[1]):
        if selected_indices!=None:
            ax.plot(samples[:, i], label=state_names[selected_indices[i]])
        else:
            ax.plot(samples[:, i])
    ax.legend(title='States')
    plt.show()

def plot_corner(samples, mean, state_positions, selected_indices=None, cl=0.95,
                            x_true=None):
    """
    Plot a corner plot using pre-selected samples and indices.
    """
    if selected_indices!=None:
        labels = [list(state_positions.keys())[i] for i in selected_indices]
    else:
        labels = list(state_positions.keys())
        selected_indices = np.range(len(list(state_positions.keys())))
    fig = corner.corner(
        samples,
        labels=labels,
        range=[(samples[:, i].min(), samples[:, i].max()) for i in range(samples.shape[1])],
        title_quantiles=[(1 - cl) / 2, 0.5, 1 - (1 - cl) / 2],
        quantiles=[(1 - cl) / 2, 0.5, 1 - (1 - cl) / 2],
        show_titles=True
    )

    axes = np.array(fig.axes).reshape((len(selected_indices), len(selected_indices)))
    for i in range(len(selected_indices)):
        for j in range(len(selected_indices)):
            ax = axes[i, j]
            if i == j:
                ax.axvline(mean[i], color="cornflowerblue", label="Mean")
                # ax.axvline(map[i], color="red", label="MAP")
                if x_true is not None:
                    ax.axvline(x_true[i], color="green", label="True")

            elif i > j:
                ax.axvline(x=mean[j], color="cornflowerblue")
                ax.axhline(y=mean[i], color="cornflowerblue")
                # ax.axvline(x=map[j], color="red", label="MAP")
                # ax.axhline(y=map[i], color="red", label="MAP")
                if x_true is not None:
                    ax.axvline(x=x_true[j], color="green")
                    ax.axhline(y=x_true[i], color="green")
            if i == len(selected_indices) - 1 and j == len(selected_indices) - 1:
                ax.legend(loc='best')
    plt.show()

def plot_graph_signal(signal, graph, state_positions, title):
    """
    Plot a signal on the graph using Graph_Plotter.
    """
    f, ax = plt.subplots()
    plotter = Graph_Plotter(graph, signal, ax, pos=state_positions)
    plotter.plot_stationary(signal)
    plotter.draw_labels()
    ax.set_title(title)
    plt.show()

def main():
    # Define number of random traces
    num_traces = 9
    fixed_index = 25

    # Pick random indices and optionally add the fixed index
    selected_indices = np.random.choice(len(state_positions), num_traces, replace=False).tolist()
    if fixed_index is not None and fixed_index not in selected_indices:
        selected_indices.append(fixed_index)

    # Subset the samples
    subset_samples = samples[:, selected_indices]

    plot_graph_signal(signal=out_est,graph=graph, state_positions=state_positions, title="exact signal")
    plot_trace(subset_samples, state_positions, selected_indices)
    plot_corner(subset_samples, mean, state_positions, selected_indices, x_true=x_true)

if __name__ == "__main__":
    main()