import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from prior import Matern_graph_pytorch
from plot_tools import Graph_Plotter
import torch
import os

import networkx as nx

def make_1d_grid_graph(N: int, weighted: bool = True, store_pos_as: str = "pos"):
    """
    1D discretization of [0,1] with N nodes.
    - Nodes are labeled 0..N-1 (works well with your adjacency_matrix code).
    - Node attribute `pos` stores the coordinate (x, 0.0) for easy plotting.
    - Edge weights are 1/h^2 if weighted=True (finite-difference scaling).
    """
    if N < 2:
        raise ValueError("N must be >= 2")

    G = nx.path_graph(N)  # edges: (0,1), (1,2), ..., (N-2,N-1)

    xs = np.linspace(0.0, 1.0, N)
    h = xs[1] - xs[0]

    # Store node locations
    for i, x in enumerate(xs):
        # choose either x (scalar) or (x, 0) tuple; tuple is nicer for plotting
        G.nodes[i][store_pos_as] = (float(x), 0.0)

    # Add weights
    if weighted:
        w = 1.0 / (h * h)
        for u, v in G.edges():
            G.edges[u, v]["weight"] = float(w)
    else:
        for u, v in G.edges():
            G.edges[u, v]["weight"] = 1.0

    return G

N = 32
graph = make_1d_grid_graph(N, weighted = False)

G = Matern_graph_pytorch(graph)
p = torch.ones(G.num_edges)

G.compute_laplacian_from_tensor_autograd(p, normalized=True)
print(G.L)
