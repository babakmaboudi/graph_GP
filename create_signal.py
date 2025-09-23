import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from prior import Matern_graph, Matern_graph_pytorch, heat_periodic_pytorch_new
from plot_tools import Graph_Plotter
import torch
import torch.nn.functional as F

def create_signal_heat_numpy():
    # loading the graph of the US states
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

    # defining a Matern prior on the graph
    G = Matern_graph(graph, normalize=True)

    # defining an initial nodal value distribution
    v0_prior = np.zeros(G.N)
    #v0_prior[25] = 40.

    # parameters used in the prior
    nu_prior = 1 # regularity
    dt_prior = 0.1 # time-step size
    T_prior = 5. # maximum time for simulation
    MAX_ITER_prior = int(T_prior / dt_prior) # number of iterations to hit maximum time

    # The non-linear mapping that makes the prior positive
    prior_map_parameter = 0.1 # output variance parameter
    prior_map_text = 'lambda x: prior_map_parameter*np.exp(x)' # the copy of the mapping for future reference
    prior_map = lambda x: prior_map_parameter*np.exp(x) # the actual mapping

    # create a signal
    x_true = np.random.standard_normal(G.num_edges) # the unknown for the inverse problem
    w_noise_true = np.random.standard_normal([MAX_ITER_prior, v0_prior.shape[0]]) # the noise vector for the Gaussian processes
    G.update_L( prior_map(x_true) , normalize=True) # updating the graph weights accroding to the unknown
    y_true = G.sample_heat(v0_prior, nu=nu_prior, dt=dt_prior, w_noise=w_noise_true ) # creating a noise free signal

    # creating a normalized noise vector
    noise_vec = np.random.standard_normal(y_true.shape)
    noise_vec = noise_vec/np.linalg.norm(noise_vec) # normalizing accroding to the l2 norm of the noise

    # visualizeing the noisy measurement with 1% noise
    sigma = 0.01*np.linalg.norm(y_true)/np.sqrt(y_true.shape[0]*y_true.shape[1])
    y_obs = y_true + sigma*noise_vec
    
    # saving the signal file
    obs_data = {'porb_type': 'heat', 'x_true': x_true, 'y_true': y_true, 'noise_vec': noise_vec, 'nu_prior': nu_prior, 'dt_prior': dt_prior, 'T_prior': T_prior, 'prior_map_parameter': prior_map_parameter, 'prior_map_text': prior_map_text, 'MAX_ITER_prior': MAX_ITER_prior}

    with open('./obs/numpy/heat/obs.pickle', 'wb') as handle:
        pickle.dump(obs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plotting the true parameter
    f, axes = plt.subplots(1,2, figsize=[12,6])
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    plotter = Graph_Plotter(graph, y_true, axes[0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true[-1]) # plotting the initial condition
    anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true, axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true), axes[1])

    plt.show()

def create_signal_stationary_numpy():
    # loading the graph of the US states
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

    # defining a Matern prior on the graph
    G = Matern_graph(graph, normalize=True)

    # defining an outbreak in the first state (Alabama)
    v0_prior = np.zeros(G.N)
    v0_prior[25] = 1.

    # parameters used in the prior
    nu_prior = 1 # regularity

    # The non-linear mapping that makes the prior positive
    prior_map_parameter = 0.1 # output variance parameter
    prior_map_text = 'lambda x: 1+ prior_map_parameter*np.exp(x)' # the copy of the mapping for future reference
    prior_map = lambda x: 1+ prior_map_parameter*np.exp(x) # the actual mapping

    # create a signal
    x_true = np.random.standard_normal(G.num_edges) # the unknown for the inverse problem
    G.update_L( prior_map(x_true) , normalize=True) # updating the graph weights accroding to the unknown
    y_true = G.sample_stationary(v0_prior, nu=nu_prior ) # creating a noise free signal

    # creating a normalized noise vector
    noise_vec = np.random.standard_normal(y_true.shape)
    noise_vec = noise_vec/np.linalg.norm(noise_vec) # normalizing accroding to the l2 norm of the noise

    # visualizeing the noisy measurement with 1% noise
    sigma = 0.01*np.linalg.norm(y_true)
    y_obs = y_true + sigma*noise_vec
    
    # saving the signal file
    obs_data = {'porb_type': 'stationary', 'x_true': x_true, 'y_true': y_true, 'noise_vec': noise_vec, 'nu_prior': nu_prior, 'prior_map_parameter': prior_map_parameter, 'prior_map_text': prior_map_text}

    with open('./obs/numpy/stationary/obs.pickle', 'wb') as handle:
        pickle.dump(obs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plotting the true parameter
    f, axes = plt.subplots(1,2, figsize=[12,6])
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    plotter = Graph_Plotter(graph, y_true, axes[0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true) # plotting the initial condition

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true, axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true), axes[1])

    plt.show()


def create_signal_heat_pytorch():
    # loading the graph of the US states
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

    # defining a Matern prior on the graph
    G = Matern_graph_pytorch(graph, normalize=True)

    # data type used for pytorch
    dtype = torch.float64
    # defining initial nodal values
    v0 = torch.randn(G.N, dtype=dtype)
    #v0[25] = 1.

    # parameters used in the prior
    nu_prior = 1 # regularity
    dt_prior = 0.1 # time-step size
    T_prior = 5. # maximum time for simulation
    MAX_ITER_prior = int(T_prior / dt_prior) # number of iterations to hit maximum time

    # The non-linear mapping that makes the prior positive
    prior_map_scale = .1 # output variance parameter
    prior_map_mean = .0
    prior_map_text = 'lambda x:  prior_map_mean + prior_map_scale*prior_map_parameter*torch.exp(x)' # the copy of the mapping for future reference
    prior_map = lambda x:  prior_map_mean + prior_map_scale*torch.exp(x) # the actual mapping

    # create a signal
    #x_true = torch.randn(G.num_edges).to(dtype) # the unknown for the inverse problem
    x_true = torch.normal( torch.zeros(G.num_edges ), torch.ones(G.num_edges) ).to(dtype)
    w_noise_true = torch.randn(MAX_ITER_prior, v0.shape[0]).to(dtype)
    G.update_L( prior_map(x_true) , normalize=True) # updating the graph weights accroding to the unknown
    y_true = G.sample_heat(v0, nu=nu_prior, dt=dt_prior, T=T_prior, w_noise=w_noise_true ) # creating a noise free signal

    # creating a normalized noise vector
    noise_vec = torch.randn(y_true.shape)
    noise_vec = noise_vec/torch.linalg.norm(noise_vec) # normalizing accroding to the l2 norm of the noise

    # visualizeing the noisy measurement with 1% noise
    #sigma = 0.01*torch.linalg.norm(y_true)/torch.sqrt(torch.tensor(y_true.shape[0]*y_true.shape[1]) )
    sigma = 0.01*torch.linalg.norm( y_true, dim=1 )/torch.sqrt(torch.tensor(y_true.shape[1]) )
    sigma = sigma.view(-1,1)

    y_obs = y_true + sigma*noise_vec
     
    # saving the signal file
    obs_data = {'porb_type': 'heat', 'x_true': x_true, 'y_true': y_true, 'noise_vec': noise_vec, 'v0':v0, 'w_noise_true': w_noise_true, 'nu_prior': nu_prior, 'dt_prior': dt_prior, 'T_prior': T_prior, 'prior_map_mean': prior_map_mean, 'prior_map_scale': prior_map_scale, 'prior_map_text': prior_map_text, 'MAX_ITER_prior': MAX_ITER_prior}

    with open('./obs/torch/heat/obs.pickle', 'wb') as handle:
        pickle.dump(obs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plotting the true parameter
    f, axes = plt.subplots(1,2, figsize=[12,6])
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true[-1].detach().numpy() ) # plotting the initial condition
    anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[1])

    plt.show()

def create_signal_stationary_pytorch():
    # loading the graph of the US states
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

    # defining a Matern prior on the graph
    G = Matern_graph_pytorch(graph, normalize=True)

    # data type used for pytorch
    dtype = torch.float64
    # defining initial nodal values
    v0 = torch.zeros(G.N, dtype=dtype)
    #v0 = 10*torch.ones(G.N, dtype=dtype)
    #v0 = 1+torch.rand(G.N, dtype=dtype)
    v0[25] = 10.

    # parameters used in the prior
    nu_prior = 1 # regularity

    # The non-linear mapping that makes the prior positive
    prior_map_scale = .1 # output variance parameter
    prior_map_mean = .0
    prior_map_text = 'lambda x:  prior_map_mean + prior_map_scale*prior_map_parameter*torch.exp(x)' # the copy of the mapping for future reference
    prior_map = lambda x:  prior_map_mean + prior_map_scale*torch.exp(x) # the actual mapping

    # create a signal
    #x_true = torch.randn(G.num_edges).to(dtype) # the unknown for the inverse problem
    x_true = torch.normal( torch.zeros(G.num_edges ), torch.ones(G.num_edges) ).to(dtype)
    G.update_L( prior_map(x_true) , normalize=True) # updating the graph weights accroding to the unknown
    y_true = G.sample_stationary(v0, nu=nu_prior ) # creating a noise free signal

    # creating a normalized noise vector
    noise_vec = torch.randn(y_true.shape)
    noise_vec = noise_vec/torch.linalg.norm(noise_vec) # normalizing accroding to the l2 norm of the noise

    # visualizeing the noisy measurement with 1% noise
    #sigma = 0.01*torch.linalg.norm(y_true)/torch.sqrt(torch.tensor(y_true.shape[0]*y_true.shape[1]) )
    sigma = 0.01*torch.linalg.norm( y_true )
    sigma = sigma.view(-1,1)

    y_obs = y_true + sigma*noise_vec
     
    # saving the signal file
    obs_data = {'porb_type': 'stationary', 'v0': v0, 'x_true': x_true, 'y_true': y_true, 'noise_vec': noise_vec, 'nu_prior': nu_prior, 'prior_map_mean': prior_map_mean, 'prior_map_scale': prior_map_scale, 'prior_map_text': prior_map_text}

    with open('./obs/torch/stationary/obs.pickle', 'wb') as handle:
        pickle.dump(obs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # plotting the true parameter
    f, axes = plt.subplots(1,2, figsize=[12,6])
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true.detach().numpy() ) # plotting the initial condition

    # plotting the true graph weights
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), axes[0], pos=state_positions)
    plotter.plot_graph_wieghts(prior_map(x_true).detach().numpy(), axes[1])

    plt.show()

def smooth_signal_1d(N, sigma=2.0, device=None):
    """
    Generate a 1D smooth signal with local correlation using Gaussian filtering.
    """
    x = torch.arange(-3*sigma, 3*sigma+1, device=device)
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()
    
    # Convert to shape (1, 1, kernel_size) for conv1d
    kernel = kernel.view(1, 1, -1)
    kernel_size = kernel.shape[-1]
    
    # White noise signal
    white_noise = torch.randn(1, 1, N, device=device)
    
    # Pad the signal using reflection padding to avoid edge artifacts
    pad_size = kernel_size // 2
    white_noise_padded = F.pad(white_noise, (pad_size, pad_size), mode='reflect')
    
    # Apply convolution
    smooth_signal = F.conv1d(white_noise_padded, kernel)
    return smooth_signal.view(-1)

def smooth_signal_1d_scaled(N, sigma=2.0, device=None):
    """
    Generate a 1D smooth signal with local correlation using Gaussian filtering
    and scale it between 0.5 and 1.
    """
    x = torch.arange(-3*sigma, 3*sigma+1, device=device)
    kernel = torch.exp(-0.5 * (x / sigma)**2)
    kernel = kernel / kernel.sum()
    
    # Convert to shape (1, 1, kernel_size) for conv1d
    kernel = kernel.view(1, 1, -1)
    kernel_size = kernel.shape[-1]
    
    # White noise signal
    white_noise = torch.randn(1, 1, N, device=device)
    
    # Pad the signal using reflection padding to avoid edge artifacts
    pad_size = kernel_size // 2
    white_noise_padded = F.pad(white_noise, (pad_size, pad_size), mode='reflect')
    
    # Apply convolution
    smooth_signal = F.conv1d(white_noise_padded, kernel).view(-1)
    
    # Min-max normalize to [0, 1]
    min_val = smooth_signal.min()
    max_val = smooth_signal.max()
    normalized_signal = (smooth_signal - min_val) / (max_val - min_val)
    
    # Scale to [0.5, 1]
    scaled_signal = 0.5 + 0.5 * normalized_signal

def create_signal_1D_heat():
    dtype = torch.float64
    N = 128
    dx = 1./N

    x = torch.linspace(0, 1.-dx, N)
    input = torch.exp(-0.5 * ((x - 0.5) / 0.05) ** 2).to(dtype)

    prior = heat_periodic_pytorch_new(N)

    #x_true = torch.randn(G.num_edges).to(dtype) # the unknown for the inverse problem
    #x_true = torch.randn(N-1).to(dtype)
    #x = torch.linspace(0, 1.-dx, N-1).to(dtype)
    #x_true = -0.5 * ((x - 0.5) / 0.05) ** 2

    #x_true = smooth_signal_1d(N-1, sigma=5.).to(dtype)
    x_true = torch.randn(N-1, dtype=torch.float64)

    prior.update_L(  torch.exp(x_true) )
    T_max=.5
    dt = 0.005
    MAX_ITER = int(T_max/dt)
    w = torch.randn(MAX_ITER, N).to(dtype)
    nu = 0
    y_true = prior.sample_heat( input, nu=nu, T=T_max, dt=dt , w_noise=w)

    # saving the signal file
    obs_data = {'porb_type': 'heat_1D', 'v0': input, 'x_true': x_true, 'y_true': y_true, 'dt': dt, 'T_max': T_max, 'N': N, 'nu':nu}

    with open('./obs/torch/heat_1D/obs_non_smooth.pickle', 'wb') as handle:
        pickle.dump(obs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.imshow(y_true)
    plt.show()



if __name__ == '__main__':
    #create_signal_heat_numpy()
    #create_signal_stationary_numpy()
    #create_signal_heat_pytorch()
    #create_signal_stationary_pytorch()

    create_signal_1D_heat()


