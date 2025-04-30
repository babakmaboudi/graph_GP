import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy as scp
import matplotlib.animation as animation
import networkx as nx
from scipy.sparse import csr_matrix
import pickle
import matplotlib.transforms as mtransforms
# Class for creating a Laplacian operator on a circular graph
# The code is inspired by the work "Non-separable Spatio-temporal Graph Kernels via 
# SPDEs" (2022) by Nikitin et al.


class Matern_graph():
    """
    A class to compute the Matern covariance of the form (tau*I + Laplacian)^nu from a graph.
    """


    def __init__(self, graph, N=128, normalize=False): # N is the number of nodes on the circumference of the circle
        """
        Initializes the MaternGraph from a given networkx graph.

        Parameters:
        -----------
        graph: Input graph (networkx.Graph)
        N: Number of nodes (int, default: 128)
        normalize: Flag for whether to normalize the Laplacian (boolean, default: False)

        Returns:
        --------
        None
        """
        element = list(graph.nodes())[0]
        if nx.is_weighted(graph):
            W = csr_matrix(nx.linalg.attrmatrix.attr_matrix(graph, "weight", rc_order=graph.nodes()))
        else:
            if isinstance(element, int):
                W  =  nx.adjacency_matrix(graph, nodelist=range(len(graph.nodes())))
            else:
                W = nx.adjacency_matrix(graph, nodelist=graph.nodes())
        self.W_sparse = W
        self.num_edges = W.data.shape

        self.W = (W.toarray()).astype(float) #dense matrix of graph's adjacency matrix

        self.N = self.W.shape[0]
        # Diagonal matrix of accumulated sums
        Dw = np.diag(np.sum(self.W, axis = 1))

        # Compute Laplacian
        self.L = Dw - self.W

        # Normalize Laplacian if required
        if normalize:
            self.sqrt_norm = np.sqrt(self.normalize_L(Dw))
            self.L = self.sqrt_norm @ self.L @ self.sqrt_norm

        # Set parameters, changed according to formulation from paper
        #self.c = 0.1 time-dependent case
        self.c = 1 #stationary case
        self.tau = 0.1 #length scale: the distance to which nodes are correlated. In the paper tau = 2nu/kappa^2

    def update_L(self, W, normalize=True):
        """Update Laplacian

        Parameters:
        -----------
        W: np array of weights
        normalize: boolean flag, default=True

        Returns:
        --------
        """
        self.W_sparse.data = W
        W = (self.W_sparse.toarray()).astype(float)
        Dw = np.diag(np.sum(W, axis = 1))
        L = Dw - W
        if normalize:
            L = self.sqrt_norm @ L @ self.sqrt_norm
        self.L = L

    def normalize_L(self, Dw):
        """normalize L

        Parameters:
        -----------
        Dw: Diagonal matrix of the sum of the weights

        Returns:
        --------
        """
        normalizer = np.zeros_like(Dw)
        for i in range(Dw.shape[0]):
            if(Dw[i,i] == 0):
                normalizer[i,i] = 0
            else:
                normalizer[i,i] = 1/Dw[i,i]
        return normalizer


    def sample_stationary(self, w, nu=2):#changed from 2 to 3, and added c from paper
        """
        Samples from the stationary Gaussian process by solving (K^nu) v = w, at this point nu = 1 or 2 corresponds to Matern 1/2 kernel or ? should be 3 or 5 for 3/2 or 5/3 kernel.

        Parameters:
        -----------
        w: Input noise vector (np.array).
        nu: Smoothness parameter (int, default: 2).

        Returns:
        ----------
        np.array: Sampled stationary process.
        """
        K_nu = np.linalg.matrix_power(self.c * (self.tau * np.eye(self.N) + self.L), nu)
        return np.linalg.solve(K_nu, w)


    # sampling the non-stationary Gaussian process with pure white noise
    def sample_heat(self, v0=None, nu=2, T=5.0, dt=0.01): #nu changed from 2 to 3
        """
        Samples from the non-stationary Gaussian process using the heat equation.

        Parameters:
        -----------
        nu: Smoothness parameter (int, default: 2).
        T: Maximum time (float, default: 5.0).
        dt: Time step (float, default: 0.01).

        Returns:
        --------
        np.array: Solution over time.
        """

        if(v0 is None):
            # initial condition is a sample from the stationary distribution
            w = np.random.standard_normal(self.N)
            v0 = self.sample_stationary(w, nu=nu)
        #u0 = np.linalg.solve(np.linalg.matrix_power(self.tau*np.eye(self.N)+self.L, nu),w)
        #u0 = np.zeros_like(p)

        # const
        L_nu = np.linalg.matrix_power(self.tau * np.eye(self.N) + self.L, nu)
        MAX_ITER = int(T / dt)
        sol = [ v0 ]

        # this is a discrete heat equation with explicit Euler time steps. The Gaussian noise
        # is discretized as dW ~= sqrt(dt)*e, where e is a standard normal Gaussian vector.
        for i in range(MAX_ITER):
            # uncomment for classic Laplacian operator
            #u = u0 - dt*(self.L@u0) + np.sqrt(dt)*np.random.standard_normal( u0.shape[0] )

            # uncomment for matern Laplacian operator
            #v = v0 - dt*(L@v0) + np.sqrt(dt)*np.random.standard_normal( v0.shape[0] )

            # uncomment for a noise model with a smooth covariance
            noise = self.sample_stationary(np.random.standard_normal( v0.shape[0] ), nu=nu)
            v = v0 - dt * (L_nu @ v0) + np.sqrt(dt) * noise
            v0 = v
            sol.append(v)
        return np.array(sol)

class KL_expansion(Matern_graph):
    def __init__(self, graph, N=128, normalize=False):
        super().__init__(graph, N=N, normalize=normalize)

    def create_eigen_decomposition(self, nu=2, num_terms=10):
        self.num_terms = num_terms
        K_nu = np.linalg.matrix_power(self.c * (self.tau * np.eye(self.N) + self.L), -nu)
        eig_vals, eig_vecs = np.linalg.eig(K_nu)

        self.lambda_i = np.sqrt(eig_vals[:num_terms])
        self.e_i = eig_vecs[:,:num_terms]

    def assemble(self, p):
        return self.lambda_i*(self.e_i@p)

class Matern_circle_graph():
    """
    Class for creating a Laplacian operator on a circular graph. Code is inspired by the work "Non-separable Spatio-temporal Graph Kernels via
    SPDEs" (2022) by Nikitin et al.Graph consists of nodes on the circumference of
    a circle. The edges of the graph connects only the neighboring nodes weighted by
    the distance. Since the nodes are chosen uniformly, the weights become a constant
    value.
    """


    def __init__(self, N=128): # N is the number of nodes on the circumference of the circle
        """
        Initializes the MaternCircleGraph.

        Parameters:
        N: Number of nodes on the circumference of the circle (int).
        """
        self.N = N
        self.theta = np.linspace(0,2*np.pi,N,endpoint=False) # uniform placement of the nodes
        points = np.exp( 1j*self.theta ) # defining nodes on the unit circle in the complex plane
        
        # the following line tranform complex plane to real plane
        self.x = np.real(points)
        self.y = np.imag(points)

        # initiating an empty adjacency matrix
        self.W = np.zeros([N,N])
    
        # weights on chosen as the distance between neighboring nodes
        self.W[0,-1] = np.sqrt( (self.x[0] - self.x[-1])**2 + (self.y[0]-self.y[-1])**2 )
        self.W[0,1] = np.sqrt( (self.x[1] - self.x[0])**2 + (self.y[1]-self.y[0])**2 )
        for i in range(1,self.x.shape[0]-1):
            self.W[i,i-1] = np.sqrt( (self.x[i] - self.x[i-1])**2 + (self.y[i]-self.y[i-1])**2 )
            self.W[i,i+1] = np.sqrt( (self.x[i+1] - self.x[i])**2 + (self.y[i+1]-self.y[i])**2 )
        self.W[-1,0] = np.sqrt( (self.x[-1] - self.x[0])**2 + (self.y[-1]-self.y[0])**2 )
        self.W[-1,-2] = np.sqrt( (self.x[-1] - self.x[-2])**2 + (self.y[-1]-self.y[-2])**2 )

        # defining the discretization element: length of a descrete arc
        dx = 2*np.pi/self.N
        Dw = np.diag( np.sum(self.W, axis = 1)) # diagonal matrix of accumulated sums
        self.L = (Dw - self.W)/dx/dx # The graph Laplacian operator (differs from the paper by the nomalaization constant 1/dx/dx)

        #self.tau = 1/(0.5*2*np.pi)/(0.5*2*np.pi)
        self.tau = 0.1 # The length scale: the distantce to which nodes are correlated. In the paper tau = 2nu/kappa^2


    def sample_stationary(self, w, nu=2):
        """
        Samples a stationary Gaussian process by solving (K^nu) v = w, at this point nu = 1 or 2

        Parameters:
        - w (np.ndarray): Random input.
        - nu (int): Smoothness level.

        Returns:
        - np.ndarray: Sampled process.
        """
        return np.linalg.solve(np.linalg.matrix_power(self.tau*np.eye(self.N)+self.L, nu),w)


    def sample_heat(self, nu=2):
        """
       Samples a non-stationary Gaussian process with pure white noise.

       Parameters:
       - nu (int): Smoothness level.

       Returns:
       - np.ndarray: Sampled process.
       """
        # initial condition is a sample from the stationary distribution
        w = np.random.standard_normal(self.N)
        v0 = self.sample_stationary(w, nu=nu)
        #u0 = np.linalg.solve(np.linalg.matrix_power(self.tau*np.eye(self.N)+self.L, nu),w)
        #u0 = np.zeros_like(p)
        T = 5. # maximum time
        dt = 0.01*80 # time step

        # const
        L = np.linalg.matrix_power(self.tau*np.eye(self.N)+self.L, nu)
        MAX_ITER = int( T/dt )
        sol = [ v0 ]

        # this is a discrete heat equation with explicit Euler time steps. The Gaussian noise
        # is discretized as dW ~= sqrt(dt)*e, where e is a standard normal Gaussian vector.
        for i in range(MAX_ITER):
            # uncomment for classic Laplacian operator
            #u = u0 - dt*(self.L@u0) + np.sqrt(dt)*np.random.standard_normal( u0.shape[0] )

            # uncomment for matern Laplacian operator
            #v = v0 - dt*(L@v0) + np.sqrt(dt)*np.random.standard_normal( v0.shape[0] )

            # uncomment for a noise model with a smooth covariance
            v = v0 - dt*(L@v0) + np.sqrt(dt)*self.sample_stationary(np.random.standard_normal( v0.shape[0] ), nu=nu)
            v0 = v
            sol.append(v)
        return np.array(sol)


class Plotter:
    """
    Class for plotting a sample in a plt axis handler.
    """

    def plot_sample(self, u, ax):
        """
        Plot sample in a plt axis handler

        Parameters:
        -----------
        u: displacement (np.array)
        ax: Matplotlib axis object

        Returns:
        --------
        """
        nbin = 64 # number of intensity colors
        cmap = matplotlib.colormaps['plasma'] # choice of color palette
        colors = cmap(np.linspace(0, 1, nbin)) # color codes in RGB

        dist = np.max(u) - np.min(u) # normalizing value to make a double into an integer
        for i in range(self.x.shape[0]-1):
            color_id = int( (u[i]-np.min(u) )/dist * (nbin-1) ) # transforimg a double into an integer between 0 and nbin
            x = self.x[i:i+2]
            y = self.y[i:i+2]
            ax.plot( x, y , color=colors[color_id], linewidth=3 )
        ax.set_aspect('equal')

    def plot_sample_displacement(self, u, ax):
        """
        Plot graph samples as normal displacements (currently has a bug)?

        Parameters:
        -----------
        u: displacement (np.array)
        ax: Matplotlib axis object

        Returns:
        --------
        """
        nbin = 64
        cmap = matplotlib.colormaps['plasma']
        colors = cmap(np.linspace(0, 1, nbin))

        points = (1+0.05*u)*np.exp( 1j*self.theta )
        xx = np.real(points)
        yy = np.imag(points)


        dist = np.max(u) - np.min(u)
        for i in range(self.x.shape[0]-1):
            color_id = int( (u[i]-np.min(u) )/dist * (nbin-1) )
            x = xx[i:i+2] 
            y = yy[i:i+2]
            ax.plot( x, y, color='blue', linewidth=3 )
        #ax.plot(self.x,self.y)
        #ax.scatter(self.x,self.y, c=u, '.')
        ax.set_aspect('equal')


class Graph_Plotter():
    """
    Handles plotting of graph-based processes.
    """
    def __init__(self, graph, out, ax, pos=None):
        """
        Initializes the GraphPlotter.

        Parameters:
        -----------
        graph: NetworkX graph instance
        out: Simulation output
        ax: Matplotlib axis object
        """
        self. out = out
        self.graph = graph
        self.nodes = graph.nodes()
        self.ax = ax
        #self.pos = nx.random_layout(self.graph)
        if(pos is None):
            self.pos = nx.spring_layout(self.graph)
        else:
            self.pos = pos
        self.cmap = matplotlib.colormaps['plasma']

        nx.draw_networkx_edges(self.graph, self.pos, alpha=0.2)


    def draw_labels(self, labels=None, font_size=8, x_offset=0.02, y_offset=0.02):
        """
        Draw labels for nodes with an offset from the node positions.

        Parameters:
        -----------
        labels: dict of labels, default None
            The labels for each node. If None, node names are used as labels.
        font_size: int, default 8
            The font size of the labels.
        x_offset: float, default 0.02
            The horizontal offset for the label position (positive to move right, negative to move left).
        y_offset: float, default 0.02
            The vertical offset for the label position (positive to move up, negative to move down).

        Returns:
        --------
        None
        """
        if labels is None:
            # Default: label each node with its name
            labels = {node: str(node) for node in self.nodes}

        # Manually adjust the positions and add the labels with offset
        for node, (x, y) in self.pos.items():
            label = labels[node]
            # Offset the labels using data units instead of axes
            if label=="vermont":
                self.ax.text(x -0.05, y -0.01, label, fontsize=font_size, ha='center', va='center')
            elif label=="rhode island":
                self.ax.text(x - 0.04, y - 0.02, label, fontsize=font_size, ha='center', va='center')
            elif label=="new jersey":
                self.ax.text(x - 0.04, y - 0.02, label, fontsize=font_size, ha='center', va='center')
            elif label=="north carolina":
                self.ax.text(x - 0.04, y - 0.02, label, fontsize=font_size, ha='center', va='center')
            else:
                self.ax.text(x + x_offset, y + y_offset, label, fontsize=font_size, ha='center', va='center')


    def plot_stationary(self, signal):
        """
        Plots stationary signals on graph nodes.

        Parameters:
        -----------
        signal: np.array

        Returns:
        --------
        """
        self.nc = nx.draw_networkx_nodes(self.graph, self.pos, nodelist=self.nodes, node_color=signal, node_size=100, cmap=self.cmap, ax=self.ax)


    def update_frame(self,frame):
        """
        Updates node colors for animation frames.

        Parameters:
        -----------
        frame: animation frame

        Returns:
        --------
        """
        signal = self.out[frame]
        self.nc.set_array(np.array(signal))
        self.ax.set_title('{}'.format(frame))
