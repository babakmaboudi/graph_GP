import torch
import matplotlib
import matplotlib.pyplot as plt
import pickle
import os
import numpy as np
# LaTeX-style fonts
matplotlib.rcParams.update({
    "text.usetex": True,
    "font.family": "serif",
    "font.serif": ["Computer Modern"],
    "font.size": 10,
    "axes.labelsize":16,
    "axes.titlesize":16,
    "legend.fontsize": 12,
})


def hpd_interval_numpy(samples: np.ndarray, mass: float = 0.94):
    if samples.ndim != 2:
        raise ValueError("samples must be 2D [N, M].")
    if not (0 < mass < 1):
        raise ValueError("mass must be in (0, 1).")

    N, M = samples.shape
    k = int(np.floor(mass * N))
    if k <= 0:
        raise ValueError("mass too small for given N.")
    sortx = np.sort(samples, axis=0)  # [N, M]

    if k >= N:
        return sortx[0, :], sortx[-1, :]

    # Window widths
    widths = sortx[k:, :] - sortx[:-k, :]  # [N-k, M]
    idx = np.argmin(widths, axis=0)        # [M]
    lower = sortx[idx, np.arange(M)]
    upper = sortx[idx + k, np.arange(M)]
    return lower, upper

def plot_signal():
    with open('obs/graphical_heat/heat/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    v0 = obs_data['v0']
    N = obs_data['N']

    nodes = torch.linspace(1, N, N)

    # 1 row, 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(10, 3), constrained_layout=True)

    # ---- Plot 1: edge weights w ----
    axes[0].plot(x_true, 'o', linewidth=1.5)
    axes[0].grid(True)
    axes[0].set_title(r"(a) true edge weights")
    axes[0].set_xlabel(r"edge index")
    axes[0].set_ylabel(r"$w_i$")

    # ---- Plot 2: initial nodal values e ----
    axes[1].plot(nodes, v0, linewidth=1.5)
    axes[1].set_title(r"(b) initial nodal values")
    axes[1].set_xlabel("node index")
    axes[1].set_ylabel("$\mathbf{f}_0$")

    # ---- Plot 3: time series ----
    im = axes[2].imshow(y_true, extent=[1, 32, 0, 1], aspect='auto', origin='lower')
    axes[2].set_title(r"(c) signal")
    axes[2].set_xlabel("node index")
    axes[2].set_ylabel("$t$")

    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    os.makedirs("plots/graphical_heat", exist_ok=True)

    plt.savefig("plots/graphical_heat/signal.pdf", bbox_inches="tight")
    plt.show()

def plot_signal2():
    with open('obs/graphical_heat/heat/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    #x_smooth = obs_data['x_smooth']
    y_true = obs_data['y_true']
    noise_vec = obs_data['noise_vec']
    nu_prior = obs_data['nu_prior']
    prior_map_mean = obs_data['prior_map_mean']
    prior_map_scale = obs_data['prior_map_scale']
    v0 = obs_data['v0']
    dt_prior = obs_data['dt_prior']
    T_prior = obs_data['T_prior']
    MAX_ITER_prior = obs_data['MAX_ITER_prior']
    w_noise_true = obs_data['w_noise_true']
    N = obs_data['N']

    x = torch.linspace(1,N,N)
    plt.figure()
    plt.plot(x_true)

    plt.figure()
    plt.plot(x,v0)

    plt.figure()
    plt.imshow(y_true)
    plt.show()

def plot_estimates():
    with open('obs/graphical_heat/heat/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    N = obs_data['N']

    with open('./stat/graphical_heat/heat/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)

    x_MAP = MAP_data['x_MAP'].detach().cpu().numpy()

    with open('./stat/graphical_heat/heat/samples.pickle', 'rb') as handle:
        samples_data = pickle.load(handle)

    x_samples = samples_data['x_samples'].detach().cpu().numpy()

    # Posterior summaries
    x_mean = np.mean(x_samples, axis=0)
    hdi_lower, hdi_upper = hpd_interval_numpy(x_samples, 0.95)  # shape (N,), (N,)

    # Convert to asymmetric error bars around the mean
    yerr = np.vstack([x_mean - hdi_lower, hdi_upper - x_mean])   # shape (2, N)

    # x-axis: edge index (1..N)
    edges = np.arange(1, N )

    fig, ax = plt.subplots(1, 1, figsize=(8, 3.5), constrained_layout=True)

    # 95% credible interval + posterior mean (dots)
    ax.errorbar(
        edges, x_mean, yerr=yerr,
        fmt='o', markersize=4, elinewidth=1.2, capsize=2,
        label=r"Posterior mean with 95\% HDI"
    )

    # True and MAP as points
    ax.plot(edges, x_true, 'o', markersize=4, label=r"True $w$")
    ax.plot(edges, x_MAP,  'o', markersize=4, label=r"MAP")

    ax.set_xlabel(r"Edge index")
    ax.set_ylabel(r"Weight $w$")
    ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)
    ax.legend(loc="best", frameon=True)

    os.makedirs("plots/graphical_heat", exist_ok=True)
    plt.savefig("plots/graphical_heat/weights_inference.pdf", bbox_inches="tight")
    plt.show()

def plot_trace(indices=(0, 5, 10, 15)):
    """
    Plot stacked trace plots for selected indices.
    Wide figure, normal plot height.
    """
    with open('./stat/graphical_heat/heat/samples.pickle', 'rb') as handle:
        samples_data = pickle.load(handle)

    x_samples = samples_data['x_samples'].detach().cpu().numpy()
    num_samples = x_samples.shape[0]
    iters = np.arange(num_samples)

    n_plots = len(indices)

    # Wide but normal-height figure
    fig, axes = plt.subplots(
        n_plots, 1,
        figsize=(6.0, 3.0),   # wider, but standard height
        sharex=True,
        constrained_layout=True
    )

    if n_plots == 1:
        axes = [axes]

    for ax, idx in zip(axes, indices):
        ax.plot(iters, x_samples[:, idx], linewidth=1)
        ax.set_ylabel(rf"$w_{{{idx+1}}}$")
        ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.6)

    axes[-1].set_xlabel(r"NUTS Iteration")

    os.makedirs("plots/graphical_heat", exist_ok=True)
    plt.savefig("plots/graphical_heat/trace_plot.pdf", bbox_inches="tight")
    plt.show()

def plot_noise_increments_comparison():
    # ---- Load truth ----
    with open('obs/graphical_heat/heat/obs.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    w_noise_true = obs_data["w_noise_true"]
    # ensure numpy
    if hasattr(w_noise_true, "detach"):
        w_noise_true = w_noise_true.detach().cpu().numpy()
    else:
        w_noise_true = np.asarray(w_noise_true)

    # ---- Load samples ----
    with open('./stat/graphical_heat/heat/MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)

    w_noise_MAP = MAP_data['w_noise_MAP'].detach().cpu().numpy()


    with open('./stat/graphical_heat/heat/samples.pickle', 'rb') as handle:
        samples_data = pickle.load(handle)

    w_noise_samples = samples_data['w_samples'].detach().cpu().numpy()

    # ---- Posterior summaries ----
    w_noise_mean = np.mean(w_noise_samples, axis=0)  # (T, D)
    w_noise_std = np.std(w_noise_samples, axis=0)  # (T, D)

    # ---- Scalar comparison metrics ----
    rmse = np.sqrt(np.mean((w_noise_mean - w_noise_true) ** 2))
    a = w_noise_mean.ravel()
    b = w_noise_true.ravel()
    corr = np.corrcoef(a, b)[0, 1] if (np.std(a) > 0 and np.std(b) > 0) else np.nan

    print(f"Noise increments comparison:")
    print(f"  RMSE(mean vs true) = {rmse:.4e}")
    print(f"  Corr(mean, true)   = {corr:.4f}")

    # ---- Plot: True / Posterior mean ----
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.2), constrained_layout=True)

    im0 = axes[0].imshow(w_noise_true, extent=[1, 32, 0, 1], aspect="auto", origin="lower")
    axes[0].set_title(r"(a) True $\mathbf \varepsilon$")
    axes[0].set_xlabel(r"Node index")
    axes[0].set_ylabel(r"time $t$")
    fig.colorbar(im0, ax=axes[0], fraction=0.046, pad=0.04)

    im1 = axes[1].imshow(w_noise_mean, extent=[1, 32, 0, 1], aspect="auto", origin="lower")
    axes[1].set_title(r"(b) Expected $\mathbf \varepsilon$")
    axes[1].set_xlabel(r"Noise index")
    axes[1].set_ylabel(r"time $t$")
    fig.colorbar(im1, ax=axes[1], fraction=0.046, pad=0.04)

    im2 = axes[2].imshow(w_noise_std, extent=[1, 32, 0, 1], aspect="auto", origin="lower")
    axes[2].set_title(r"(c) Standard deviation in $\mathbf \varepsilon$")
    axes[2].set_xlabel(r"Noise index")
    axes[2].set_ylabel(r"time $t$")
    fig.colorbar(im2, ax=axes[2], fraction=0.046, pad=0.04)



    os.makedirs("plots/graphical_heat", exist_ok=True)
    plt.savefig("plots/graphical_heat/noise_increments_comparison.pdf", bbox_inches="tight")
    plt.show()

if __name__ == '__main__':
    #plot_signal()
    #plot_estimates()
    #plot_trace()
    plot_noise_increments_comparison()
