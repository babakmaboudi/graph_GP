from os import stat
import numpy as np
import torch
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1 import make_axes_locatable
from prior import heat_periodic_pytorch_new
import pickle
import matplotlib

matplotlib.rc('xtick', labelsize=16) 
matplotlib.rc('ytick', labelsize=16) 

def compute_hpdi(samples, alpha=0.05):
    samples = np.sort(samples)
    n = samples.shape[0]
    interval_idx_inc = int(np.floor((1-alpha)*n))
    n_intervals = n - interval_idx_inc
    intervals = [ (samples[i], samples[i+interval_idx_inc]) for i in range(n_intervals) ]
    widths = [interval[1] - interval[0] for interval in intervals ]
    min_idx = np.argmin(widths)
    hpd_min, hpd_max = intervals[min_idx]
    return np.array([hpd_min, hpd_max])

def plot_init_cond_and_x_true():
    x = np.linspace(0,1,127)
    y = np.exp(-0.5 * ((x - 0.5) / 0.05) ** 2)

    f,ax = plt.subplots(1)
    ax.plot(x, y, linewidth=3)
    ax.set_xlabel(r'$\xi$', fontsize=16)
    ax.set_ylabel(r'$u_0$', fontsize=16)
    plt.tight_layout()
    plt.grid()
    plt.savefig('./plots/init_cond.pdf')

    with open('obs/torch/heat_1D/obs_smooth.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']

    f,ax = plt.subplots(1)
    ax.plot(x_true, 'o', linewidth=3)
    ax.set_xlabel(r'$x_i$', fontsize=16)
    ax.set_ylabel('value', fontsize=16)
    ax.grid()
    plt.tight_layout()
    plt.savefig('./plots/x_true_smooth.pdf')
    plt.show()



def plot_signal():
    with open('obs/torch/heat_1D/obs_smooth.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    v0 = obs_data['v0']
    T_max = obs_data['T_max']
    dt = obs_data['dt']
    N = obs_data['N']
    nu = obs_data['nu']
    MAX_ITER = int(T_max/dt)

    f,ax = plt.subplots(1, figsize=[4,4.5])
    cmap = ax.imshow( np.rot90(y_true) )
    ax.set_xlabel(r'time index $t$', fontsize=16)
    ax.set_ylabel(r'index $i$ ($x$)', fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar( cmap, cax=cax )
    plt.tight_layout()
    plt.savefig('./plots/signal_smooth.pdf')
    plt.show()

def plot_MAP():
    with open('./obs/torch/heat_1D/obs_non_smooth.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    v0 = obs_data['v0']
    T_max = obs_data['T_max']
    dt = obs_data['dt']
    N = obs_data['N']
    nu = obs_data['nu']
    MAX_ITER = int(T_max/dt)

    prior = heat_periodic_pytorch_new(N)

    with open('./stat/heat_1D/smooth-MAP.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)

    x_map = stat_data['x_map']
    w_map = stat_data['w_map']

    prior.update_L(  torch.exp(x_map) )
    y_map = prior.sample_heat( v0, nu=nu, T=T_max, dt=dt , w_noise=w_map)

    f,ax = plt.subplots(1, figsize=[4,4.5])
    cmap = ax.imshow( np.rot90(y_map.detach().numpy()) )
    ax.set_xlabel(r'time index $t$', fontsize=16)
    ax.set_ylabel(r'index $i$ ($x$)', fontsize=16)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar( cmap, cax=cax )
    plt.tight_layout()
    plt.savefig('./plots/data_map_smooth.pdf')
    plt.show()
    exit()

    f,axes = plt.subplots(1)
    axes.plot(torch.exp(x_true).detach().numpy(), label='true')
    axes.plot(torch.exp( x_map ).detach().numpy(), label='estimated')

    plt.show()

def plot_trace():
    with open('./stat/heat_1D/non-smooth-samples.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)
    
    x_samples = stat_data['x']
    w_samples = stat_data['w']

    indecies = np.random.permutation( x_samples.shape[1] )
    indecies2 = np.random.permutation( w_samples.shape[1] )

    n_plots = 5
    f,axes = plt.subplots(n_plots,2)
    for i in range(n_plots):
        axes[i,0].plot( x_samples[:,indecies[i]] )
        axes[i,1].plot( w_samples[:,indecies2[i],indecies[i]] )
        axes[i,0].set_ylabel(r'$x_{%d}$' % (indecies[i]), fontsize=16)
        axes[i,1].set_ylabel(r'$w_{%d,%d}$' % (indecies[i],indecies2[i]), fontsize=16)
        axes[i,0].set_yticks([0, 2])
        axes[i,0].set_yticklabels([0, 2])
        axes[i,1].set_yticks([0, 2])
        axes[i,1].set_yticklabels([0, 2])

    for i in range(n_plots-1):
        axes[i,0].set_xticklabels([])
        axes[i,1].set_xticklabels([])





    axes[n_plots-1,0].set_xlabel('sample index', fontsize=16)
    axes[n_plots-1,1].set_xlabel('sample index', fontsize=16)

    axes[0,0].set_title(r'$x_i$', fontsize=16)
    axes[0,1].set_title(r'$w_{ij}$', fontsize=16)


    plt.tight_layout()
    plt.subplots_adjust(hspace=0)

    plt.savefig('./plots/trace.pdf')

    plt.show()
    




def plot_mean():
    with open('./obs/torch/heat_1D/obs_smooth.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    v0 = obs_data['v0']
    T_max = obs_data['T_max']
    dt = obs_data['dt']
    N = obs_data['N']
    nu = obs_data['nu']
    MAX_ITER = int(T_max/dt)

    prior = heat_periodic_pytorch_new(N)

    with open('./stat/heat_1D/smooth-samples.pickle', 'rb') as handle:
        stat_data = pickle.load(handle)

    
    x_samples = stat_data['x']
    w_samples = stat_data['w']

    x_mean = torch.mean(x_samples, dim=0)
    w_mean = torch.mean(w_samples, dim=0)

    prior.update_L(  torch.exp(x_mean) )
    y_map = prior.sample_heat( v0, nu=nu, T=T_max, dt=dt , w_noise=w_mean)

    f,ax = plt.subplots(1, figsize=[4,4.5])
    cmap = ax.imshow( np.rot90(y_map.detach().numpy()) )
    ax.set_xlabel(r'time index $t$', fontsize=14)
    ax.set_ylabel(r'index $i$ ($x$)', fontsize=14)
    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='5%', pad=0.05)
    f.colorbar( cmap, cax=cax )
    plt.tight_layout()
    plt.savefig('./plots/data_mean_smooth.pdf')
    plt.show()
    exit()

    f,axes = plt.subplots(1)
    axes.plot(torch.exp(x_true).detach().numpy(), label='true')
    axes.plot(torch.exp( x_map ).detach().numpy(), label='estimated')

    plt.show()

def plot_parameters():
    with open('./obs/torch/heat_1D/obs_non_smooth.pickle', 'rb') as handle:
        obs_data = pickle.load(handle)

    x_true = obs_data['x_true']
    y_true = obs_data['y_true']
    v0 = obs_data['v0']
    T_max = obs_data['T_max']
    dt = obs_data['dt']
    N = obs_data['N']
    nu = obs_data['nu']
    MAX_ITER = int(T_max/dt)

    with open('./stat/heat_1D/non-smooth-MAP.pickle', 'rb') as handle:
        MAP_data = pickle.load(handle)

    x_map = MAP_data['x_map']
    w_map = MAP_data['w_map']

    with open('./stat/heat_1D/non-smooth-samples.pickle', 'rb') as handle:
        samples_data = pickle.load(handle)

    
    x_samples = samples_data['x']
    w_samples = samples_data['w']

    x_mean = torch.mean(x_samples, dim=0)
    w_mean = torch.mean(w_samples, dim=0)

    hpdi_intervals = []
    for i in range(x_samples.shape[1]):
        hpdi_interval = compute_hpdi( x_samples[:,i] )
        hpdi_intervals.append(hpdi_interval)
    hpdi_intervals = np.array(hpdi_intervals).T

    grid = np.linspace(0,N-1,N-1, endpoint=False)

    plt.figure()
    plt.fill_between(grid, hpdi_intervals[0], hpdi_intervals[1], color='blue', alpha=0.15, label='95% HPDI')
    plt.plot(grid, x_true.detach().numpy(), label='true')
    plt.plot(grid, x_map.detach().numpy(), label='MAP')
    plt.plot(grid, x_mean.detach().numpy(), label='mean')

    plt.xlabel('edge index', fontsize=16)
    plt.ylabel('value', fontsize=16)

    plt.legend(fontsize=12)
    plt.savefig('./plots/estimations.pdf')
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    plot_init_cond_and_x_true()
    #plot_signal()
    #plot_MAP()
    #plot_mean()
    #plot_parameters()
    #plot_trace()
