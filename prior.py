import numpy as np
import matplotlib.pyplot as plt
#import matplotlib
#import scipy as scp
#import matplotlib.animation as animation
from scipy.sparse import csr_matrix
#import pickle
#import matplotlib.transforms as mtransforms
import torch
import networkx as nx
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
    def sample_heat(self, v0, nu=2, T=5.0, dt=0.01, w_noise=None): #nu changed from 2 to 3
        """
        Samples from the non-stationary Gaussian process using the heat equation.

        Parameters:
        -----------
        nu: Smoothness parameter (int, default: 2).
        T: Maximum time (float, default: 5.0).
        dt: Time step (float, default: 0.01).
        w_noise: Stochastic drive in the SPDE (Brownian motion)

        Returns:
        --------
        np.array: Solution over time.
        """
        
        # uncomment if you want a random initial condition
        #if(v0 is None):
        #    # initial condition is a sample from the stationary distribution
        #    w = np.random.standard_normal(self.N)
        #    v0 = self.sample_stationary(w, nu=nu)
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
            #noise = self.sample_stationary(np.random.standard_normal( v0.shape[0] ), nu=nu)
            noise = w_noise[i]
            v = v0 - dt * (L_nu @ v0) + np.sqrt(dt) * noise
            v0 = v
            sol.append(v)
        return np.array(sol)

class Matern_graph_pytorch():
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
        crow_indices = torch.tensor(self.W_sparse.indptr, dtype=torch.int64)
        col_indices = torch.tensor(self.W_sparse.indices, dtype=torch.int64)
        values = torch.tensor(self.W_sparse.data, dtype=torch.float64)
        self.W_sparse_tensor = torch.sparse_csr_tensor(crow_indices, col_indices, values, size=self.W_sparse.shape)

        self.num_edges = W.data.shape

        self.W_dense_tensor = self.W_sparse_tensor.to_dense().to(torch.float64)

        self.N = self.W_sparse_tensor.shape[0]
        # Diagonal matrix of accumulated sums
        Dw_tensor = torch.diag( torch.sum( self.W_dense_tensor, dim=1) )

        # Compute Laplacian
        self.L_tensor = Dw_tensor - self.W_dense_tensor

        # Normalize Laplacian if required
        if normalize:
            self.sqrt_norm_tensor = torch.sqrt( self.normalize_L( Dw_tensor ) )
            self.L_tensor = self.sqrt_norm_tensor @ self.L_tensor @ self.sqrt_norm_tensor

        # Set parameters, changed according to formulation from paper
        #self.c = 0.1 time-dependent case
        self.c = 1 #stationary case
        self.tau = 0.1 #length scale: the distance to which nodes are correlated. In the paper tau = 2nu/kappa^2


    def update_L(self, W, normalize=True):
        self.W_sparse_tensor = torch.sparse_csr_tensor( self.W_sparse_tensor.crow_indices(), self.W_sparse_tensor.col_indices(), W, size=self.W_sparse_tensor.shape, dtype=W.dtype )
        W_dense = self.W_sparse_tensor.to_dense()
        Dw = torch.diag( torch.sum( W_dense, dim=1 ) )
        L_tensor = Dw - W_dense
        if normalize:
            L_tensor = self.sqrt_norm_tensor @ L_tensor @ self.sqrt_norm_tensor
        self.L_tensor = L_tensor

    def normalize_L(self, Dw):
        """normalize L

        Parameters:
        -----------
        Dw: Diagonal matrix of the sum of the weights

        Returns:
        --------
        """
        normalizer = torch.zeros_like(Dw)
        for i in range(Dw.shape[0]):
            if(Dw[i,i] == 0):
                normalizer[i,i] = 0
            else:
                normalizer[i,i] = 1/Dw[i,i]
        return normalizer

    def sample_stationary(self, w, nu=2):
        I = torch.eye( self.N, dtype=self.L_tensor.dtype )
        temp = self.c * ( self.tau * I + self.L_tensor )
        K_nu_tensor = torch.linalg.matrix_power(temp, nu)

        return torch.linalg.solve( K_nu_tensor, w )

    def sample_heat(self, v0, nu=2, T=5.0, dt=0.01, w_noise=None):
        I = torch.eye( self.N, dtype=self.L_tensor.dtype )
        temp = self.c * ( self.tau * I + self.L_tensor )
        K_nu_tensor = torch.linalg.matrix_power(temp, nu)


        MAX_ITER = int(T / dt)
        sol = [ v0 ]

        #if(w_noise is None):
        #    w_noise = torch.randn(MAX_ITER, v0.shape[0], dtype=torch.float64)

        for i in range(MAX_ITER):
            # uncomment for smooth noise
            #noise = self.sample_stationary( w_noise[i] , nu=nu)
            #v = v0 - dt * (K_nu_tensor @ v0) + torch.sqrt( torch.tensor(dt) ) * noise

            # uncomment for white noise
            v = v0 - dt * (K_nu_tensor @ v0) + torch.sqrt( torch.tensor(dt) ) * w_noise[i]
            v0 = v
            sol.append( v )

        return torch.stack(sol)



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
        dt = 0.01 # time step

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

class edge_correlation():
    def __init__(self, graph, num_edges):
        self.graph = graph
        self.num_edges = num_edges

        element = list(graph.nodes())[0]
        if nx.is_weighted(graph):
            W = csr_matrix(nx.linalg.attrmatrix.attr_matrix(graph, "weight", rc_order=graph.nodes()))
        else:
            if isinstance(element, int):
                W  =  nx.adjacency_matrix(graph, nodelist=range(len(graph.nodes())))
            else:
                W = nx.adjacency_matrix(graph, nodelist=graph.nodes())
        rows, cols = W.nonzero()

        edges = [(i, j) for i, j in zip(rows, cols) if i >= j]

        self.nodes = list(self.graph.nodes)

        #W_weighted = W.copy()
        self.cov = torch.zeros( len(edges), len(edges) ).to(torch.float64)

        for idx1 in range( len(edges) ):
            n1 = int( edges[idx1][0] )
            n2 = int( edges[idx1][1] )
            for idx2 in range( len(edges) ):
                n3 = int( edges[idx2][0] )
                n4 = int( edges[idx2][1] )
                d = self.find_dist(n1,n2,n3,n4)
                d = d + 1
                if(idx1 == idx2):
                    d = 0
                self.cov[idx1,idx2] = torch.exp( torch.tensor(-0.5*d**2) )

        mat = self.cov.detach().numpy()
        eigvals, eigvecs = np.linalg.eig(mat)
        print(eigvals)
        exit()
        sample = np.random.multivariate_normal(np.zeros(110), self.cov)
        print(sample)
        exit()
        print( torch.min(self.cov) )
        plt.imshow( self.cov )
        plt.show()
        self.dist = torch.distributions.MultivariateNormal(torch.zeros(len(edges)), covariance_matrix=self.cov)

    def sample(self, num_samples):
        return self.dist.sample(num_samples,)


    def find_dist(self, i1,i2,i3,i4):
        d1 = nx.shortest_path_length(self.graph, source=self.nodes[i1], target=self.nodes[i3])
        d2 = nx.shortest_path_length(self.graph, source=self.nodes[i1], target=self.nodes[i4])
        d3 = nx.shortest_path_length(self.graph, source=self.nodes[i2], target=self.nodes[i3])
        d4 = nx.shortest_path_length(self.graph, source=self.nodes[i2], target=self.nodes[i4])
        return np.min( np.array([d1,d2,d3,d4]) )
