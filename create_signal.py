import numpy as np
import pickle
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from Matern_prior import Matern_graph, Graph_Plotter

def create_signal():
    # loading the graph of the US states
    graph = pickle.load(open('./data/covid_data/g.pkl', "rb"))

    # defining a Matern prior on the graph
    G = Matern_graph(graph, normalize=True)

    # defining an outbreak in the first state (Alabama)
    v0 = np.zeros(G.N)
    v0[25] = 40.

    # creating a graph plotter object

    # plotting 5 prior samples:
    # loading a fix location of the states
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)
    


    nu = 1
    dt = 0.1
    T = 5.
    MAX_ITER = int(T / dt)

    # create a signal
    x_true = np.random.standard_normal(G.num_edges) # the unknown for the inverse problem
    w_noise = np.
    G.update_L( 0.1*np.exp(x_true) , normalize=True) # updating the graph weights accroding to the unknown
    y_true = G.sample_heat(v0, nu=nu, dt=dt ) # creating a noise free signal

    # creating a normalized noise vector
    noise_vec = np.random.standard_normal(y_true.shape)
    noise_vec = noise_vec/np.linalg.norm(noise_vec) # normalizing accroding to the l2 norm of the noise

    # visualizeing the noisy measurement with 1% noise
    sigma = 0.01*np.linalg.norm(y_true)/np.sqrt(y_true.shape[0]*y_true.shape[1])
    y_obs = y_true + sigma*noise_vec

    # plotting the true parameter
    f, ax = plt.subplots()
    with open('./stat_positions.pickle', 'rb') as handle:
        state_positions = pickle.load(handle)

    plotter = Graph_Plotter(graph, y_true, ax, pos=state_positions) # initiating the graph plotter
    plotter.plot_stationary(y_true[0]) # plotting the initial condition
    anim = animation.FuncAnimation(fig=f, func=plotter.update_frame, frames=y_true.shape[0], interval=100)

    # saving the signal file
    obs_data = {'x_true': x_true, 'y_true': y_true, 'noise_vec': noise_vec}


    with open('./obs/heat/reg1/obs.pickle', 'wb') as handle:
        pickle.dump(obs_data, handle, protocol=pickle.HIGHEST_PROTOCOL)

    plt.show()
    
if __name__ == '__main__':
    create_signal()


