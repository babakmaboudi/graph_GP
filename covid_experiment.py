import os
import pickle
import json
import time
import argparse
import copy
import numpy as np
import networkx as nx
from Matern_prior import Matern_graph

import matplotlib.pyplot as plt

import sklearn
from sklearn.preprocessing import FunctionTransformer



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


def generate_dataset(X, y, num_training_data, num_testing_data, start=0, log_target=False, rs=42,
                     interpolation=False):
    start_test = start + num_training_data
    end_test = start_test + num_testing_data

    if interpolation:
        train_X, test_X, train_y, test_y = sklearn.model_selection.train_test_split(
            X[start:end_test], y[start:end_test],
            test_size=0.1, random_state=rs)
        train_y, test_y = train_y[:, np.newaxis], test_y[:, np.newaxis]
    else:
        train_X, train_y = X[start:start_test], y[start:start_test, np.newaxis]
        test_X, test_y = X[start_test:end_test], y[start_test:end_test, np.newaxis]

    if log_target:
        qt = FunctionTransformer(func=np.log1p, inverse_func=np.expm1)
        train_y = qt.fit_transform(train_y)
        test_y = qt.transform(test_y)
    else:
        qt = None

    return train_X, train_y, test_X, test_y, qt


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
N_NODES = len(graph.nodes())

NUM_TEST_WEEKS = args.num_test_weeks

NUM_TRAIN = 33 * N_NODES
NUM_TEST = NUM_TEST_WEEKS * N_NODES
START = 4 * N_NODES * 2
N_ITER = 5_000

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
y_cases = pickle.load(open(Y_CASES_PATH, "rb"))
y_deaths = pickle.load(open(Y_DEATHS_PATH, "rb"))
from_state_to_id = pickle.load(open(FROM_STATE_TO_ID_PATH, "rb"))

if IS_PREDICT_CASES:
    y = y_cases
else:
    y = y_deaths

y[y < 0] = 0
print(y_cases[:80], y_cases[80:159])



#
G = Matern_graph(graph, normalize=True)
#
w = np.random.standard_normal(G.N)
#out = G.sample_posterior()
# print(out)
#
# # plot_heat(graph)
#
#
# if __name__ == "__main__":
#     results = {}
#     # for kernel_name, kernel in exp_kernels.items():
#     #    results[kernel_name] = {}
#     # for i, rs in tqdm(enumerate(RANDOM_SEEDS), total=len(RANDOM_SEEDS)):
#     rs = 1234
#     N = len(graph.nodes())
#     # utils.set_all_random_seeds(rs)
#     train_X, train_y, test_X, test_y, qt = generate_dataset(
#          X, y, NUM_TRAIN, NUM_TEST,
#          start=START + len(graph.nodes()), log_target=LOG_TARGET, rs=rs,
#          interpolation=INTERPOLATION)
#     #print("Evaluating kernel ", kernel_name)
#     # start = time.time()
#     # need to set up some sampler or inference method:
#     # result, gprocess = utils_opt.evaluate_kernel_mcmc(
#     #    copy.deepcopy(kernel), train_X, train_y, test_X, test_y, graph,
#     #    transformer=qt,
#     #    n_iter=N_ITER, dump_directory=DUMP_DIRECTORY,
#     #    dump_everything=DUMP_EVERYTHING, optimizer_name="LBFGS")
#     # results[rs] = result
#     # results[kernel_name][rs]["time"] = time.time() - start
#     # json.dump(results, open(os.path.join(DUMP_DIRECTORY, "results.json"), "w"))
#     # print("leny", len(train_y))
#     print("nodes", N_NODES)
#     print("out", len(out))
#     dt = 0.001
#     thin = int(0.1/dt)
#     every_100th = out[::thin]
#     signal = np.diag(every_100th)
#     print("shape", signal.shape)
#     plot_nodes_with_colors(graph, signal)
