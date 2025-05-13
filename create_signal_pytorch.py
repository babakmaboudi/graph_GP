import numpy as np
import torch
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Matern_prior import Matern_graph, Graph_Plotter, Matern_graph_pytorch

def create_signal():
    # loading the graph of the US states
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

    # defining a Matern prior on the graph
    G = Matern_graph_pytorch(graph, normalize=True)

    dtype = torch.float64
    # defining an outbreak in the first state (Alabama)
    v0 = torch.zeros(G.N, dtype=dtype)
    v0[25] = 1.

    # creating a graph plotter object

    # plotting 5 prior samples:
    # loading a fix location of the states
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    
    nu = 1
    dt = 0.05
    T = 4.
    MAX_ITER = int(T / dt)

    # create a signal
    #x_true = torch.randn(G.num_edges).to(dtype) # the unknown for the inverse problem
    x_true = torch.normal( torch.zeros(G.num_edges ), torch.ones(G.num_edges) ).to(dtype)
    w_noise_true = torch.randn(MAX_ITER, v0.shape[0]).to(dtype)
    G.update_L( 0.1*torch.exp(x_true) , normalize=True) # updating the graph weights accroding to the unknown
    y_true = G.sample_heat(v0, nu=nu, dt=dt, T=T, w_noise=w_noise_true ) # creating a noise free signal


    # creating a normalized noise vector
    noise_vec = torch.randn(y_true.shape)
    noise_vec = noise_vec/torch.linalg.norm(noise_vec) # normalizing accroding to the l2 norm of the noise

    # visualizeing the noisy measurement with 1% noise
    #sigma = 0.01*torch.linalg.norm(y_true)/torch.sqrt(torch.tensor(y_true.shape[0]*y_true.shape[1]) )
    sigma = 0.01*torch.linalg.norm( y_true, dim=1 )/torch.sqrt(torch.tensor(y_true.shape[1]) )
    sigma = sigma.view(-1,1)

    y_obs = y_true + sigma*noise_vec

    #plotting the true parameter
    f, ax = plt.subplots()
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    #f,ax = plt.subplots(1)
    plotter = Graph_Plotter(graph, y_true.detach().numpy(), ax, pos=state_positions) # initiating the graph plotter
    #plotter.plot_graph_wieghts(y_true[0].detach().numpy(), x_true.detach().numpy(), ax)

    ones = torch.linspace(0,1,y_true[0].shape[0] )
    plotter.plot_stationary(ones.detach().numpy()) # plotting the initial condition
    anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)

    # saving the signal file
    obs_data = {'x_true': x_true, 'w_noise_true': w_noise_true, 'y_true': y_true, 'noise_vec': noise_vec, 'dtype': dtype, 'dt': dt, 'T':T, 'MAX_ITER': MAX_ITER, 'nu':nu, 'v0': v0, 'sigma': sigma}

    with open('./obs/heat/reg1/obs.pickle', 'wb') as handle:
        pickle.dump(obs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.show()
    
if __name__ == '__main__':
    create_signal()
