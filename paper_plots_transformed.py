import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.ticker as mticker
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.gridspec as gridspec
from prior import Matern_graph_pytorch
from plot_tools import Graph_Plotter, Graph_Plotter_With_US_Map
import torch
import os

# NOTE: Posterior means shown for graph edge weights are computed in physical
# weight space: mean(prior_map(x_samples)), not prior_map(mean(x_samples)).
import cartopy.io.shapereader as shpreader
import torch

def add_small_cbar(fig, ax, mappable, ticks=3, label=None):
    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="2.5%", pad=0.02)  # thin bar
    cb = fig.colorbar(mappable, cax=cax)

    # exactly 3 ticks
    vmin, vmax = mappable.get_clim()
    cb.set_ticks(np.linspace(vmin, vmax, ticks))
    cb.ax.tick_params(labelsize=7, length=2, pad=1)

    if label is not None:
        cb.set_label(label, fontsize=8)

    return cb

def add_short_cbar_below(fig, ax_for_cbar, mappable, *, ticks=3, fmt="%.1f"):
    cb = fig.colorbar(mappable, cax=ax_for_cbar, orientation="horizontal")

    vmin, vmax = mappable.get_clim()
    cb.set_ticks(np.linspace(vmin, vmax, ticks))
    cb.ax.xaxis.set_major_formatter(mticker.FormatStrFormatter(fmt))

    cb.ax.tick_params(labelsize=10, length=2, pad=1)
    # Remove outline to look cleaner in papers (optional)
    cb.outline.set_linewidth(0.6)

    return cb

def plot_signal_stationary():
    # LaTeX-style fonts (same as your others)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "font.size": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    # --- load data
    with open('obs/stationary_linearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data["x_true"]
    y_true = obs_data["y_true"]
    v0 = obs_data["v0"]
    prior_map_mean = obs_data["prior_map_mean"]
    prior_map_scale = obs_data["prior_map_scale"]

    prior_map = lambda x: prior_map_mean + prior_map_scale * torch.exp(x)

    graph = pickle.load(open("./data/covid_data/g.pkl", "rb"))

    with open("./stat_positions_lon_lat.pickle", "rb") as handle:
        state_positions = pickle.load(handle)

    us_states_path = shpreader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_1_states_provinces_lakes"
    )

    # If y_true is (T,N), pick one snapshot
    y0 = y_true.detach().numpy()
    if y0.ndim == 2:
        y0 = y0[0]

    # v0 might be torch tensor; make it a numpy vector like y0
    v0_np = v0.detach().numpy() if torch.is_tensor(v0) else np.asarray(v0)
    if v0_np.ndim == 2:
        v0_np = v0_np[0]

    # True edge weights
    w_true = prior_map(x_true).detach().numpy()

    # --- figure: 3 panels + cbars below (same style as MAP plots)
    fig = plt.figure(figsize=(10.5, 3.8))
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1.0, 0.06],
        left=0.01, right=0.99,
        bottom=0.08, top=0.90,
        wspace=0.04, hspace=0.12
    )

    ax_map = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cbar = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Make each colorbar axis SHORTER and centered (exact same approach)
    for cax in ax_cbar:
        pos = cax.get_position()
        shrink = 0.40
        new_w = pos.width * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        cax.set_position([new_x, pos.y0, new_w, pos.height])

    pad = 150_000

    # --- (a) Signal y0 on nodes
    plotter0 = Graph_Plotter_With_US_Map(
        graph=graph, out=y0, ax=ax_map[1],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter0.focus_conus(pad=pad)
    plotter0.plot_stationary(y0)  # sets plotter0.nc
    ax_map[1].set_title(r"(b) Noise-free signal", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[1], plotter0.nc, ticks=3, fmt="%.1f")

    # --- (b) Source term v0 on nodes
    plotter1 = Graph_Plotter_With_US_Map(
        graph=graph, out=v0_np, ax=ax_map[0],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter1.focus_conus(pad=pad)
    plotter1.plot_stationary(v0_np)  # sets plotter1.nc
    ax_map[0].set_title(r"(a) Source term", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[0], plotter1.nc, ticks=3, fmt="%.1f")

    # --- (c) True parameter on edges
    plotter2 = Graph_Plotter_With_US_Map(
        graph=graph, out=y0, ax=ax_map[2],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter2.focus_conus(pad=pad)
    m2 = plotter2.plot_graph_wieghts(w_true, ax=ax_map[2])  # returns edge mappable
    ax_map[2].set_title(r"(c) True parameter", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[2], m2, ticks=3, fmt="%.1f")

    # Clean colorbar axes
    for cax in ax_cbar:
        cax.yaxis.set_visible(False)

    # Save (optional; change name if you want)
    os.makedirs("./plots", exist_ok=True)
    fig.savefig("./plots/signal_stationary.pdf", bbox_inches="tight")

    plt.show()

def plot_signal_heat():
    # LaTeX-style fonts (same as your others)
    plt.rcParams.update({
        "text.usetex": True,
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman"],
        "axes.titlesize": 9,
        "axes.labelsize": 9,
        "font.size": 8,
        "xtick.labelsize": 7,
        "ytick.labelsize": 7,
    })

    # --- load data
    with open('obs/heat_linearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data["x_true"]
    y_true = obs_data["y_true"]      # expected shape (T, N)
    v0 = obs_data["v0"]
    prior_map_mean = obs_data["prior_map_mean"]
    prior_map_scale = obs_data["prior_map_scale"]

    prior_map = lambda x: prior_map_mean + prior_map_scale * torch.exp(x)

    graph = pickle.load(open("./data/covid_data/g.pkl", "rb"))

    with open("./stat_positions_lon_lat.pickle", "rb") as handle:
        state_positions = pickle.load(handle)

    us_states_path = shpreader.natural_earth(
        resolution="50m",
        category="cultural",
        name="admin_1_states_provinces_lakes"
    )

    # --- numpy conversions
    # v0
    v0_np = v0.detach().numpy() if torch.is_tensor(v0) else np.asarray(v0)
    if v0_np.ndim == 2:
        v0_np = v0_np[0]

    # y_true for imshow
    y_np = y_true.detach().numpy() if torch.is_tensor(y_true) else np.asarray(y_true)
    if y_np.ndim != 2:
        raise ValueError(f"Expected y_true to be 2D (T,N). Got shape {y_np.shape}")

    T, N = y_np.shape

    # For map panels, we still need a single snapshot (e.g. t=0) to set out/vmin/vmax
    y0 = y_np[0]

    # True edge weights
    w_true = prior_map(x_true).detach().numpy()

    # --- figure: 3 panels + cbars below (same style as MAP plots)
    fig = plt.figure(figsize=(10.5, 3.8))
    gs = gridspec.GridSpec(
        2, 3,
        height_ratios=[1.0, 0.06],
        left=0.01, right=0.99,
        bottom=0.08, top=0.90,
        wspace=0.04, hspace=0.12
    )

    ax_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cbar = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Make each colorbar axis SHORTER and centered (exact same approach)
    for cax in ax_cbar:
        pos = cax.get_position()
        shrink = 0.40
        new_w = pos.width * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        cax.set_position([new_x, pos.y0, new_w, pos.height])

    # -------------------------
    # (a) Source term on map
    # -------------------------
    pad = 150_000
    plotter0 = Graph_Plotter_With_US_Map(
        graph=graph, out=v0_np, ax=ax_top[0],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter0.focus_conus(pad=pad)
    plotter0.plot_stationary(v0_np)  # sets plotter0.nc
    ax_top[0].set_title(r"(a) Initial condition", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[0], plotter0.nc, ticks=3, fmt="%.1f")

    # -------------------------
    # (b) Time-dependent signal as heatmap
    # x-axis: state index [0..N-1]
    # y-axis: time in [0,5]
    # -------------------------
    t_min, t_max = 0.0, 5.0
    extent = [0, N, t_min, t_max]  # [x0, x1, y0, y1]

    im = ax_top[1].imshow(
        y_np,
        aspect="auto",
        origin="lower",
        extent=extent,
        interpolation="nearest"
    )

    pos = ax_top[1].get_position()
    shrink = 0.92
    new_h = pos.height * shrink
    ax_top[1].set_position([pos.x0, pos.y0 + (pos.height - new_h), pos.width, new_h])

    ax_top[1].set_title(r"(b) Noise free signal", fontsize=15)
    ax_top[1].set_xlabel("State index", labelpad=2)
    ax_top[1].set_ylabel("Time", labelpad=1)

    add_short_cbar_below(fig, ax_cbar[1], im, ticks=3, fmt="%.1f")

    # -------------------------
    # (c) True parameter on edges (map)
    # -------------------------
    plotter2 = Graph_Plotter_With_US_Map(
        graph=graph, out=y0, ax=ax_top[2],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter2.focus_conus(pad=pad)
    m2 = plotter2.plot_graph_wieghts(w_true, ax=ax_top[2])  # returns edge mappable
    ax_top[2].set_title(r"(c) True parameter", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[2], m2, ticks=3, fmt="%.1f")

    # Clean colorbar axes
    for cax in ax_cbar:
        cax.yaxis.set_visible(False)

    # Save
    os.makedirs("./plots", exist_ok=True)
    fig.savefig("./plots/signal_heat.pdf", bbox_inches="tight")

    plt.show()


def plot_MAP_stationary_linearreaction():
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "font.size": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    })
    with open('obs/stationary_linearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process

    with open('./stat/stationary_linearreaction/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)


    with open('./stat_positions_lon_lat.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    us_states_path = shpreader.natural_earth(resolution="50m",category="cultural",name="admin_1_states_provinces_lakes")

    x_MAP = MAP_data['x_MAP']
    with open('./stat/stationary_linearreaction/samples.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)
    x_samples = stat_data['samples']['x']
    x_mean = torch.mean(x_samples,dim=0)
    samples_pushed_forward = torch.zeros_like(x_samples)
    for i in range(x_samples.shape[0]):
        samples_pushed_forward[i,:] = prior_map(x_samples[i])
    # Posterior summaries in physical edge-weight space
    # E[w | y] must be computed after transforming each posterior sample.
    w_mean = torch.mean(samples_pushed_forward, dim=0)
    std = torch.std(samples_pushed_forward, dim=0)

        # --- Create a 2-row GridSpec: top row maps, bottom row colorbars
    fig = plt.figure(figsize=(10.5, 3.8))  # a bit taller to fit cbars nicely
    #gs = gridspec.GridSpec(
    #    2, 3,
    #    height_ratios=[1.0, 0.06],   # bottom row thin
    #    hspace=0.15,                # spacing between map row and cbar row
    #    wspace=0.06                 # spacing between columns
    #)
    gs = gridspec.GridSpec(
    2, 3,
    height_ratios=[1.0, 0.06],
    left=0.01, right=0.99,
    bottom=0.08, top=0.90,
    wspace=0.04, hspace=0.12
    )

    ax_map = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cbar = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Make each colorbar axis SHORTER and centered
    for cax in ax_cbar:
        pos = cax.get_position()
        shrink = 0.4  # 65% width
        new_w = pos.width * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        cax.set_position([new_x, pos.y0, new_w, pos.height])

    # --- MAP panel
    plotter0 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[0],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter0.focus_conus(pad=150_000)
    m0 = plotter0.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), ax=ax_map[0])
    ax_map[0].set_title("(a) MAP", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[0], m0, ticks=3, fmt="%.1f")

    # --- Mean panel
    plotter1 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[1],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter1.focus_conus(pad=150_000)
    m1 = plotter1.plot_graph_wieghts(w_mean.detach().numpy(), ax=ax_map[1])
    ax_map[1].set_title("(b) Posterior mean", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[1], m1, ticks=3, fmt="%.1f")

    # --- Std panel
    plotter2 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[2],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter2.focus_conus(pad=150_000)
    m2 = plotter2.plot_graph_wieghts(std.detach().numpy(), ax=ax_map[2])
    ax_map[2].set_title("(c) Posterior std", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[2], m2, ticks=3, fmt="%.1f")

    # Ensure colorbar axes look clean
    for cax in ax_cbar:
        cax.yaxis.set_visible(False)

    #fig.subplots_adjust(left=0.02,right=0.98,bottom=0.08,top=0.92,wspace=0.05,hspace=0.12)

    os.makedirs("./plots", exist_ok=True)

    fig.savefig(
    "./plots/stationary_linearreaction.pdf",
    bbox_inches="tight"
    )

    plt.show()

def plot_stationary_linearreaction_real_covid():
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "font.size": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    })
    with open('obs/stationary_linearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process

    with open('./stat/stationary_linearreaction_real_covid/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)


    with open('./stat_positions_lon_lat.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    us_states_path = shpreader.natural_earth(resolution="50m",category="cultural",name="admin_1_states_provinces_lakes")

    x_MAP = MAP_data['x_MAP']
    with open('./stat/stationary_linearreaction_real_covid/samples.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)
    x_samples = stat_data['samples']['x']
    x_mean = torch.mean(x_samples,dim=0)
    samples_pushed_forward = torch.zeros_like(x_samples)
    for i in range(x_samples.shape[0]):
        samples_pushed_forward[i,:] = prior_map(x_samples[i])
    # Posterior summaries in physical edge-weight space
    # E[w | y] must be computed after transforming each posterior sample.
    w_mean = torch.mean(samples_pushed_forward, dim=0)
    std = torch.std(samples_pushed_forward, dim=0)

        # --- Create a 2-row GridSpec: top row maps, bottom row colorbars
    fig = plt.figure(figsize=(10.5, 3.8))  # a bit taller to fit cbars nicely
    #gs = gridspec.GridSpec(
    #    2, 3,
    #    height_ratios=[1.0, 0.06],   # bottom row thin
    #    hspace=0.15,                # spacing between map row and cbar row
    #    wspace=0.06                 # spacing between columns
    #)
    gs = gridspec.GridSpec(
    2, 3,
    height_ratios=[1.0, 0.06],
    left=0.01, right=0.99,
    bottom=0.08, top=0.90,
    wspace=0.04, hspace=0.12
    )

    ax_map = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cbar = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Make each colorbar axis SHORTER and centered
    for cax in ax_cbar:
        pos = cax.get_position()
        shrink = 0.4  # 65% width
        new_w = pos.width * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        cax.set_position([new_x, pos.y0, new_w, pos.height])

    # --- MAP panel
    plotter0 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[0],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter0.focus_conus(pad=150_000)
    m0 = plotter0.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), ax=ax_map[0])
    ax_map[0].set_title("(a) MAP", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[0], m0, ticks=3, fmt="%.1f")

    # --- Mean panel
    plotter1 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[1],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter1.focus_conus(pad=150_000)
    m1 = plotter1.plot_graph_wieghts(w_mean.detach().numpy(), ax=ax_map[1])
    ax_map[1].set_title("(b) Posterior mean", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[1], m1, ticks=3, fmt="%.1f")

    # --- Std panel
    plotter2 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[2],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter2.focus_conus(pad=150_000)
    m2 = plotter2.plot_graph_wieghts(std.detach().numpy(), ax=ax_map[2])
    ax_map[2].set_title("(c) Posterior std", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[2], m2, ticks=3, fmt="%.1f")

    # Ensure colorbar axes look clean
    for cax in ax_cbar:
        cax.yaxis.set_visible(False)

    #fig.subplots_adjust(left=0.02,right=0.98,bottom=0.08,top=0.92,wspace=0.05,hspace=0.12)

    os.makedirs("./plots", exist_ok=True)

    fig.savefig(
    "./plots/stationary_linearreaction_real_covid.pdf",
    bbox_inches="tight"
    )

    plt.show()

def plot_MAP_stationary_nonlinearreaction():
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "font.size": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    })
    with open('obs/stationary_nonlinearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process

    with open('./stat/stationary_nonlinearreaction/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)


    with open('./stat_positions_lon_lat.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    us_states_path = shpreader.natural_earth(resolution="50m",category="cultural",name="admin_1_states_provinces_lakes")

    x_MAP = MAP_data['x_MAP']
    with open('./stat/stationary_nonlinearreaction/samples.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)
    x_samples = stat_data['samples']['x']
    x_mean = torch.mean(x_samples,dim=0)
    samples_pushed_forward = torch.zeros_like(x_samples)
    for i in range(x_samples.shape[0]):
        samples_pushed_forward[i,:] = prior_map(x_samples[i])
    # Posterior summaries in physical edge-weight space
    # E[w | y] must be computed after transforming each posterior sample.
    w_mean = torch.mean(samples_pushed_forward, dim=0)
    std = torch.std(samples_pushed_forward, dim=0)

        # --- Create a 2-row GridSpec: top row maps, bottom row colorbars
    fig = plt.figure(figsize=(10.5, 3.8))  # a bit taller to fit cbars nicely
    #gs = gridspec.GridSpec(
    #    2, 3,
    #    height_ratios=[1.0, 0.06],   # bottom row thin
    #    hspace=0.15,                # spacing between map row and cbar row
    #    wspace=0.06                 # spacing between columns
    #)
    gs = gridspec.GridSpec(
    2, 3,
    height_ratios=[1.0, 0.06],
    left=0.01, right=0.99,
    bottom=0.08, top=0.90,
    wspace=0.04, hspace=0.12
    )

    ax_map = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cbar = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Make each colorbar axis SHORTER and centered
    for cax in ax_cbar:
        pos = cax.get_position()
        shrink = 0.4  # 65% width
        new_w = pos.width * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        cax.set_position([new_x, pos.y0, new_w, pos.height])

    # --- MAP panel
    plotter0 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[0],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter0.focus_conus(pad=150_000)
    m0 = plotter0.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), ax=ax_map[0])
    ax_map[0].set_title("(a) MAP", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[0], m0, ticks=3, fmt="%.1f")

    # --- Mean panel
    plotter1 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[1],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter1.focus_conus(pad=150_000)
    m1 = plotter1.plot_graph_wieghts(w_mean.detach().numpy(), ax=ax_map[1])
    ax_map[1].set_title("(b) Posterior mean", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[1], m1, ticks=3, fmt="%.1f")

    # --- Std panel
    plotter2 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[2],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter2.focus_conus(pad=150_000)
    m2 = plotter2.plot_graph_wieghts(std.detach().numpy(), ax=ax_map[2])
    ax_map[2].set_title("(c) Posterior std", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[2], m2, ticks=3, fmt="%.1f")

    # Ensure colorbar axes look clean
    for cax in ax_cbar:
        cax.yaxis.set_visible(False)

    #fig.subplots_adjust(left=0.02,right=0.98,bottom=0.08,top=0.92,wspace=0.05,hspace=0.12)

    os.makedirs("./plots", exist_ok=True)

    fig.savefig(
    "./plots/stationary_nonlinearreaction.pdf",
    bbox_inches="tight"
    )

    plt.show()

def plot_stationary_nonlinearreaction_real_covid():
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "font.size": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    })
    with open('obs/stationary_nonlinearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process

    with open('./stat/stationary_nonlinearreaction_real_covid/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)


    with open('./stat_positions_lon_lat.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    us_states_path = shpreader.natural_earth(resolution="50m",category="cultural",name="admin_1_states_provinces_lakes")

    x_MAP = MAP_data['x_MAP']
    with open('./stat/stationary_nonlinearreaction_real_covid/samples.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)
    x_samples = stat_data['samples']['x']
    x_mean = torch.mean(x_samples,dim=0)
    samples_pushed_forward = torch.zeros_like(x_samples)
    for i in range(x_samples.shape[0]):
        samples_pushed_forward[i,:] = prior_map(x_samples[i])
    # Posterior summaries in physical edge-weight space
    # E[w | y] must be computed after transforming each posterior sample.
    w_mean = torch.mean(samples_pushed_forward, dim=0)
    std = torch.std(samples_pushed_forward, dim=0)

        # --- Create a 2-row GridSpec: top row maps, bottom row colorbars
    fig = plt.figure(figsize=(10.5, 3.8))  # a bit taller to fit cbars nicely
    #gs = gridspec.GridSpec(
    #    2, 3,
    #    height_ratios=[1.0, 0.06],   # bottom row thin
    #    hspace=0.15,                # spacing between map row and cbar row
    #    wspace=0.06                 # spacing between columns
    #)
    gs = gridspec.GridSpec(
    2, 3,
    height_ratios=[1.0, 0.06],
    left=0.01, right=0.99,
    bottom=0.08, top=0.90,
    wspace=0.04, hspace=0.12
    )

    ax_map = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cbar = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Make each colorbar axis SHORTER and centered
    for cax in ax_cbar:
        pos = cax.get_position()
        shrink = 0.4  # 65% width
        new_w = pos.width * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        cax.set_position([new_x, pos.y0, new_w, pos.height])

    # --- MAP panel
    plotter0 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[0],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter0.focus_conus(pad=150_000)
    m0 = plotter0.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), ax=ax_map[0])
    ax_map[0].set_title("(a) MAP", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[0], m0, ticks=3, fmt="%.1f")

    # --- Mean panel
    plotter1 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[1],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter1.focus_conus(pad=150_000)
    m1 = plotter1.plot_graph_wieghts(w_mean.detach().numpy(), ax=ax_map[1])
    ax_map[1].set_title("(b) Posterior mean", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[1], m1, ticks=3, fmt="%.1f")

    # --- Std panel
    plotter2 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[2],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter2.focus_conus(pad=150_000)
    m2 = plotter2.plot_graph_wieghts(std.detach().numpy(), ax=ax_map[2])
    ax_map[2].set_title("(c) Posterior std", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[2], m2, ticks=3, fmt="%.1f")

    # Ensure colorbar axes look clean
    for cax in ax_cbar:
        cax.yaxis.set_visible(False)

    #fig.subplots_adjust(left=0.02,right=0.98,bottom=0.08,top=0.92,wspace=0.05,hspace=0.12)

    os.makedirs("./plots", exist_ok=True)

    fig.savefig(
    "./plots/stationary_nonlinearreaction_real_covid.pdf",
    bbox_inches="tight"
    )

    plt.show()

def plot_heat_linearreaction():
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "font.size": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    })
    with open('obs/heat_linearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process

    with open('./stat/heat_linearreaction/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)


    with open('./stat_positions_lon_lat.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    us_states_path = shpreader.natural_earth(resolution="50m",category="cultural",name="admin_1_states_provinces_lakes")

    x_MAP = MAP_data['x_MAP']
    with open('./stat/heat_linearreaction/samples.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)
    x_samples = stat_data['samples']['x']
    x_mean = torch.mean(x_samples,dim=0)
    samples_pushed_forward = torch.zeros_like(x_samples)
    for i in range(x_samples.shape[0]):
        samples_pushed_forward[i,:] = prior_map(x_samples[i])
    # Posterior summaries in physical edge-weight space
    # E[w | y] must be computed after transforming each posterior sample.
    w_mean = torch.mean(samples_pushed_forward, dim=0)
    std = torch.std(samples_pushed_forward, dim=0)

        # --- Create a 2-row GridSpec: top row maps, bottom row colorbars
    fig = plt.figure(figsize=(10.5, 3.8))  # a bit taller to fit cbars nicely
    #gs = gridspec.GridSpec(
    #    2, 3,
    #    height_ratios=[1.0, 0.06],   # bottom row thin
    #    hspace=0.15,                # spacing between map row and cbar row
    #    wspace=0.06                 # spacing between columns
    #)
    gs = gridspec.GridSpec(
    2, 3,
    height_ratios=[1.0, 0.06],
    left=0.01, right=0.99,
    bottom=0.08, top=0.90,
    wspace=0.04, hspace=0.12
    )

    ax_map = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cbar = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Make each colorbar axis SHORTER and centered
    for cax in ax_cbar:
        pos = cax.get_position()
        shrink = 0.4  # 65% width
        new_w = pos.width * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        cax.set_position([new_x, pos.y0, new_w, pos.height])

    # --- MAP panel
    plotter0 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[0],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter0.focus_conus(pad=150_000)
    m0 = plotter0.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), ax=ax_map[0])
    ax_map[0].set_title("(a) MAP", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[0], m0, ticks=3, fmt="%.1f")

    # --- Mean panel
    plotter1 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[1],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter1.focus_conus(pad=150_000)
    m1 = plotter1.plot_graph_wieghts(w_mean.detach().numpy(), ax=ax_map[1])
    ax_map[1].set_title("(b) Posterior mean", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[1], m1, ticks=3, fmt="%.1f")

    # --- Std panel
    plotter2 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[2],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter2.focus_conus(pad=150_000)
    m2 = plotter2.plot_graph_wieghts(std.detach().numpy(), ax=ax_map[2])
    ax_map[2].set_title("(c) Posterior std", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[2], m2, ticks=3, fmt="%.1f")

    # Ensure colorbar axes look clean
    for cax in ax_cbar:
        cax.yaxis.set_visible(False)

    #fig.subplots_adjust(left=0.02,right=0.98,bottom=0.08,top=0.92,wspace=0.05,hspace=0.12)

    os.makedirs("./plots", exist_ok=True)

    fig.savefig(
    "./plots/heat_linearreaction.pdf",
    bbox_inches="tight"
    )

    plt.show()


def plot_heat_nonlinearreaction():
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "font.size": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    })
    with open('obs/heat_nonlinearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process

    with open('./stat/heat_nonlinearreaction/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)


    with open('./stat_positions_lon_lat.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    us_states_path = shpreader.natural_earth(resolution="50m",category="cultural",name="admin_1_states_provinces_lakes")

    x_MAP = MAP_data['x_MAP']
    with open('./stat/heat_nonlinearreaction/samples.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)
    x_samples = stat_data['samples']['x']
    x_mean = torch.mean(x_samples,dim=0)
    samples_pushed_forward = torch.zeros_like(x_samples)
    for i in range(x_samples.shape[0]):
        samples_pushed_forward[i,:] = prior_map(x_samples[i])
    # Posterior summaries in physical edge-weight space
    # E[w | y] must be computed after transforming each posterior sample.
    w_mean = torch.mean(samples_pushed_forward, dim=0)
    std = torch.std(samples_pushed_forward, dim=0)

        # --- Create a 2-row GridSpec: top row maps, bottom row colorbars
    fig = plt.figure(figsize=(10.5, 3.8))  # a bit taller to fit cbars nicely
    #gs = gridspec.GridSpec(
    #    2, 3,
    #    height_ratios=[1.0, 0.06],   # bottom row thin
    #    hspace=0.15,                # spacing between map row and cbar row
    #    wspace=0.06                 # spacing between columns
    #)
    gs = gridspec.GridSpec(
    2, 3,
    height_ratios=[1.0, 0.06],
    left=0.01, right=0.99,
    bottom=0.08, top=0.90,
    wspace=0.04, hspace=0.12
    )

    ax_map = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cbar = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Make each colorbar axis SHORTER and centered
    for cax in ax_cbar:
        pos = cax.get_position()
        shrink = 0.4  # 65% width
        new_w = pos.width * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        cax.set_position([new_x, pos.y0, new_w, pos.height])

    # --- MAP panel
    plotter0 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[0],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter0.focus_conus(pad=150_000)
    m0 = plotter0.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), ax=ax_map[0])
    ax_map[0].set_title("(a) MAP", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[0], m0, ticks=3, fmt="%.1f")

    # --- Mean panel
    plotter1 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[1],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter1.focus_conus(pad=150_000)
    m1 = plotter1.plot_graph_wieghts(w_mean.detach().numpy(), ax=ax_map[1])
    ax_map[1].set_title("(b) Posterior mean", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[1], m1, ticks=3, fmt="%.1f")

    # --- Std panel
    plotter2 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[2],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter2.focus_conus(pad=150_000)
    m2 = plotter2.plot_graph_wieghts(std.detach().numpy(), ax=ax_map[2])
    ax_map[2].set_title("(c) Posterior std", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[2], m2, ticks=3, fmt="%.1f")

    # Ensure colorbar axes look clean
    for cax in ax_cbar:
        cax.yaxis.set_visible(False)

    #fig.subplots_adjust(left=0.02,right=0.98,bottom=0.08,top=0.92,wspace=0.05,hspace=0.12)

    os.makedirs("./plots", exist_ok=True)

    fig.savefig(
    "./plots/heat_nonlinearreaction.pdf",
    bbox_inches="tight"
    )

    plt.show()

def plot_heat_linearreaction_real_covid():
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "font.size": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    })
    with open('obs/heat_linearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process

    with open('./stat/heat_linearreaction_real_covid/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)


    with open('./stat_positions_lon_lat.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    us_states_path = shpreader.natural_earth(resolution="50m",category="cultural",name="admin_1_states_provinces_lakes")

    x_MAP = MAP_data['x_MAP']
    with open('./stat/heat_linearreaction_real_covid/samples.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)
    x_samples = stat_data['samples']['x']
    x_mean = torch.mean(x_samples,dim=0)
    samples_pushed_forward = torch.zeros_like(x_samples)
    for i in range(x_samples.shape[0]):
        samples_pushed_forward[i,:] = prior_map(x_samples[i])
    # Posterior summaries in physical edge-weight space
    # E[w | y] must be computed after transforming each posterior sample.
    w_mean = torch.mean(samples_pushed_forward, dim=0)
    std = torch.std(samples_pushed_forward, dim=0)

        # --- Create a 2-row GridSpec: top row maps, bottom row colorbars
    fig = plt.figure(figsize=(10.5, 3.8))  # a bit taller to fit cbars nicely
    #gs = gridspec.GridSpec(
    #    2, 3,
    #    height_ratios=[1.0, 0.06],   # bottom row thin
    #    hspace=0.15,                # spacing between map row and cbar row
    #    wspace=0.06                 # spacing between columns
    #)
    gs = gridspec.GridSpec(
    2, 3,
    height_ratios=[1.0, 0.06],
    left=0.01, right=0.99,
    bottom=0.08, top=0.90,
    wspace=0.04, hspace=0.12
    )

    ax_map = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cbar = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Make each colorbar axis SHORTER and centered
    for cax in ax_cbar:
        pos = cax.get_position()
        shrink = 0.4  # 65% width
        new_w = pos.width * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        cax.set_position([new_x, pos.y0, new_w, pos.height])

    # --- MAP panel
    plotter0 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[0],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter0.focus_conus(pad=150_000)
    m0 = plotter0.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), ax=ax_map[0])
    ax_map[0].set_title("(a) MAP", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[0], m0, ticks=3, fmt="%.1f")

    # --- Mean panel
    plotter1 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[1],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter1.focus_conus(pad=150_000)
    m1 = plotter1.plot_graph_wieghts(w_mean.detach().numpy(), ax=ax_map[1])
    ax_map[1].set_title("(b) Posterior mean", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[1], m1, ticks=3, fmt="%.1f")

    # --- Std panel
    plotter2 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[2],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter2.focus_conus(pad=150_000)
    m2 = plotter2.plot_graph_wieghts(std.detach().numpy(), ax=ax_map[2])
    ax_map[2].set_title("(c) Posterior std", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[2], m2, ticks=3, fmt="%.1f")

    # Ensure colorbar axes look clean
    for cax in ax_cbar:
        cax.yaxis.set_visible(False)

    #fig.subplots_adjust(left=0.02,right=0.98,bottom=0.08,top=0.92,wspace=0.05,hspace=0.12)

    os.makedirs("./plots", exist_ok=True)

    fig.savefig(
    "./plots/heat_linearreaction_real_covid.pdf",
    bbox_inches="tight"
    )

    plt.show()

def plot_heat_nonlinearreaction_real_covid():
    plt.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman"],
    "axes.titlesize": 9,
    "axes.labelsize": 9,
    "font.size": 8,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    })
    with open('obs/heat_linearreaction/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']

    prior_map = prior_map = lambda x: prior_map_mean + prior_map_scale*torch.exp(x)

    # creating signal/observation with noise std defined in sigma with 1% noise level
    sigma = 0.01*torch.linalg.norm(y_true)
    sigma2 = sigma*sigma
    y_obs = y_true + sigma*noise_vec

    # creating a forward operator
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb")) # the graph containing the nodal information and the connections
    G = Matern_graph_pytorch(graph) # Initiating a graph Gaussian process

    with open('./stat/heat_nonlinearreaction_real_covid/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)


    with open('./stat_positions_lon_lat.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    us_states_path = shpreader.natural_earth(resolution="50m",category="cultural",name="admin_1_states_provinces_lakes")

    x_MAP = MAP_data['x_MAP']
    with open('./stat/heat_nonlinearreaction_real_covid/samples.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)
    x_samples = stat_data['samples']['x']
    x_mean = torch.mean(x_samples,dim=0)
    samples_pushed_forward = torch.zeros_like(x_samples)
    for i in range(x_samples.shape[0]):
        samples_pushed_forward[i,:] = prior_map(x_samples[i])
    # Posterior summaries in physical edge-weight space
    # E[w | y] must be computed after transforming each posterior sample.
    w_mean = torch.mean(samples_pushed_forward, dim=0)
    std = torch.std(samples_pushed_forward, dim=0)

        # --- Create a 2-row GridSpec: top row maps, bottom row colorbars
    fig = plt.figure(figsize=(10.5, 3.8))  # a bit taller to fit cbars nicely
    #gs = gridspec.GridSpec(
    #    2, 3,
    #    height_ratios=[1.0, 0.06],   # bottom row thin
    #    hspace=0.15,                # spacing between map row and cbar row
    #    wspace=0.06                 # spacing between columns
    #)
    gs = gridspec.GridSpec(
    2, 3,
    height_ratios=[1.0, 0.06],
    left=0.01, right=0.99,
    bottom=0.08, top=0.90,
    wspace=0.04, hspace=0.12
    )

    ax_map = [fig.add_subplot(gs[0, i]) for i in range(3)]
    ax_cbar = [fig.add_subplot(gs[1, i]) for i in range(3)]

    # Make each colorbar axis SHORTER and centered
    for cax in ax_cbar:
        pos = cax.get_position()
        shrink = 0.4  # 65% width
        new_w = pos.width * shrink
        new_x = pos.x0 + (pos.width - new_w) / 2
        cax.set_position([new_x, pos.y0, new_w, pos.height])

    # --- MAP panel
    plotter0 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[0],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter0.focus_conus(pad=150_000)
    m0 = plotter0.plot_graph_wieghts(prior_map(x_MAP).detach().numpy(), ax=ax_map[0])
    ax_map[0].set_title("(a) MAP", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[0], m0, ticks=3, fmt="%.1f")

    # --- Mean panel
    plotter1 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[1],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter1.focus_conus(pad=150_000)
    m1 = plotter1.plot_graph_wieghts(w_mean.detach().numpy(), ax=ax_map[1])
    ax_map[1].set_title("(b) Posterior mean", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[1], m1, ticks=3, fmt="%.1f")

    # --- Std panel
    plotter2 = Graph_Plotter_With_US_Map(
        graph=graph, out=y_true.detach().numpy(), ax=ax_map[2],
        pos=state_positions, us_states_path=us_states_path,
        pos_is_lonlat=True, plot_crs="EPSG:5070"
    )
    plotter2.focus_conus(pad=150_000)
    m2 = plotter2.plot_graph_wieghts(std.detach().numpy(), ax=ax_map[2])
    ax_map[2].set_title("(c) Posterior std", fontsize=15)
    add_short_cbar_below(fig, ax_cbar[2], m2, ticks=3, fmt="%.1f")

    # Ensure colorbar axes look clean
    for cax in ax_cbar:
        cax.yaxis.set_visible(False)

    #fig.subplots_adjust(left=0.02,right=0.98,bottom=0.08,top=0.92,wspace=0.05,hspace=0.12)

    os.makedirs("./plots", exist_ok=True)

    fig.savefig(
    "./plots/heat_nonlinearreaction_real_covid.pdf",
    bbox_inches="tight"
    )

    plt.show()



if __name__ == "__main__":
    plot_signal_stationary()
    plot_signal_heat()
    plot_MAP_stationary_linearreaction()
    plot_MAP_stationary_nonlinearreaction()
    plot_heat_linearreaction()
    plot_heat_nonlinearreaction()

    plot_stationary_linearreaction_real_covid()
    plot_stationary_nonlinearreaction_real_covid()
    plot_heat_linearreaction_real_covid()
    plot_heat_nonlinearreaction_real_covid()
