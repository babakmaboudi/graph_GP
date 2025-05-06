import os
import pickle
import json
import time
import argparse
import copy
import numpy as np
import networkx as nx
from Matern_prior import Matern_graph, Graph_Plotter
from inverse import sample_posterior_real, forward_operator
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import sklearn
from sklearn.preprocessing import FunctionTransformer
from post_process import plot_trace, plot_graph_signal, plot_corner


# Some functions from spatiotemporal-graph-kernels repo!!!
def parse_arguments():
    parser = argparse.ArgumentParser(description='COVID-19 across the US')
    parser.add_argument('--log_target', action='store_true', default=False,
                        help='Apply log transform to the target.')
    parser.add_argument('--no-log_target', dest='log_target', action='store_false')

    parser.add_argument('--use_flight_graph', action='store_true', default=False,
                        help='Use graph that contains information about the flights.')
    parser.add_argument('--no-use_flight_graph', dest='use_flight_graph', action='store_false')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--use_normalized_target', action='store_true', default=False,
                       help='Normalize the target by population in a state.')
    group.add_argument('--no-use_normalized_target', dest='use_normalized_target', action='store_false')

    group = parser.add_mutually_exclusive_group()
    group.add_argument('--interpolation', action='store_true', default=False,
                       help='Evaluate the models on the interpolation task.')
    group.add_argument('--extrapolation', dest='interpolation', action='store_false')

    parser.add_argument('--dump_directory', type=str, help='Path to directory with results.',
                        default="dump_directory")

    parser.add_argument('--num_test_weeks', type=int, help='Number of test weeks.', default=2)

    return parser.parse_args()


def plot_nodes_with_colors(g, signal, title="2 component graph", layout=nx.spring_layout, ax=None):
    if layout is None:
        layout = nx.spring_layout

    nodes = g.nodes()
    print("nodes", len(nodes), "signal", print(len(signal)))
    assert len(nodes) == len(signal)

    # drawing nodes and edges separately so we can capture collection for colobar
    pos = layout(g)
    nx.draw_networkx_edges(g, pos, alpha=0.2)
    nc = nx.draw_networkx_nodes(g, pos, nodelist=nodes, node_color=signal,
                                node_size=100, cmap=plt.cm.jet, ax=ax)
    plt.title(title)
    plt.colorbar(nc)
    plt.axis('off')
    plt.show()


args = parse_arguments()

DATA_FOLDER = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "./data/covid_data/")
INTERPOLATION = args.interpolation
USE_FLIGHT_GRAPH = args.use_flight_graph
if USE_FLIGHT_GRAPH:
    GRAPH_PATH = os.path.join(DATA_FOLDER, "state_graph.pkl")
else:
    GRAPH_PATH = os.path.join(DATA_FOLDER, "g.pkl")
graph = pickle.load(open(GRAPH_PATH, "rb"))


IS_PREDICT_CASES = True
LOG_TARGET = args.log_target
USE_NORMALIZED_TARGET = args.use_normalized_target

# RANDOM_SEEDS = [23, 42, 82, 100, 123,
#                2 * 23, 2 * 42, 2 * 82, 2 * 100, 2 * 123]

X_PATH = os.path.join(DATA_FOLDER, "X.pkl")
#data from New York Times 2020, https://github.com/nytimes/covid-19-data
if USE_NORMALIZED_TARGET:#cases for each state and every week
    Y_CASES_PATH = os.path.join(DATA_FOLDER, "y_cases_normalized.pkl")
    Y_DEATHS_PATH = os.path.join(DATA_FOLDER, "y_deaths_normalized.pkl")
else:
    Y_CASES_PATH = os.path.join(DATA_FOLDER, "y_cases.pkl")
    Y_DEATHS_PATH = os.path.join(DATA_FOLDER, "y_deaths.pkl")

FROM_STATE_TO_ID_PATH = os.path.join(DATA_FOLDER, "from_state_to_id.pkl")
DUMP_DIRECTORY = args.dump_directory
DUMP_EVERYTHING = False
os.makedirs(DUMP_DIRECTORY, exist_ok=True)

X = pickle.load(open(X_PATH, "rb"))

y_cases = pickle.load(open(Y_CASES_PATH, "rb")) #states x time but in row vector
y_deaths = pickle.load(open(Y_DEATHS_PATH, "rb"))
from_state_to_id = pickle.load(open(FROM_STATE_TO_ID_PATH, "rb"))

if IS_PREDICT_CASES:
    y = y_cases
else:
    y = y_deaths

y[y < 0] = 0

y = y.reshape(51,-1).transpose()# format time x states

with open('./stat_positions.pickle', 'rb') as handle:
    state_positions = pickle.load(handle)
def plot_animation_covid(state_positions, graph, y):
    f, ax = plt.subplots()
    plotter = Graph_Plotter(graph, y, ax, pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y[-1]) # plotting the initial condition
    animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y.shape[0], interval=100)
    plt.show()

plot = True # True
sample = False# True

# loading samples and pre-processing samples
data = np.load('./stat/stationary/reg1/stat_real.npz')
samples = data['samples']
samples = samples[1000000::100,:] # burning the first 1e6 samples and sub-sampling every 100 samples

# cearting the mean of the samples
mean = np.mean(samples, axis=0)
num_traces = 10
#random_indices = np.random.choice(len(state_positions), num_traces, replace=False)
G = Matern_graph(graph, normalize=True)
v0 = np.zeros(G.N)
problem = forward_operator(v0, G)
out_est = problem.forward(mean)

# Define number of random traces
num_traces = 10
fixed_index = None

# Pick random indices and optionally add the fixed index
selected_indices = np.random.choice(len(state_positions), num_traces, replace=False).tolist()
if fixed_index is not None and fixed_index not in selected_indices:
    selected_indices.append(fixed_index)

# Subset the samples
subset_samples = samples[:, selected_indices]

if plot:
    plot_animation_covid(state_positions, graph, y)
    plot_trace(subset_samples, state_positions, selected_indices)
    plot_corner(subset_samples, mean=mean, state_positions=state_positions, selected_indices=selected_indices)
if sample:
    sample_posterior_real(y, graph)
#print(y_cases.reshape(51,-1).transpose()[0]) #time x states







