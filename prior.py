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

class heat_periodic_pytorch():
    """
    Class for creating a Laplacian operator on a periodic domain. Code is inspired by the work "Non-separable Spatio-temporal Graph Kernels via
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
        self.L_domain = 1.
        self.dx = self.L_domain/self.N
        points = torch.linspace(0,self.L_domain - self.dx,self.N)
        
        # initiating an empty adjacency matrix
        self.W = torch.zeros([N,N])
    
        # weights on chosen as the distance between neighboring nodes
        self.W[0,1] = 1
        for i in range(1,self.N-1):
            self.W[i,i-1] = 1
            self.W[i,i+1] = 1
        self.W[-1,-1] = 1
        self.W[-1,-2] = 1

        # defining the discretization element: length of a descrete arc
        Dw = torch.diag( torch.sum(self.W, dim = 1)) # diagonal matrix of accumulated sums
        Dw[0,0] = -2
        Dw[-1,-1] = -2
        self.L = -(Dw - self.W)/self.dx/self.dx # The graph Laplacian operator (differs from the paper by the nomalaization constant 1/dx/dx)

        #self.tau = 1/(0.5*2*np.pi)/(0.5*2*np.pi)
        self.tau = 0.1 # The length scale: the distantce to which nodes are correlated. In the paper tau = 2nu/kappa^2

    def sample_stationary(self, w, nu=2):
        I = torch.eye( self.N, dtype=self.L.dtype )
        temp = ( self.tau * I + self.L )
        K_nu_tensor = torch.linalg.matrix_power(temp, nu)

        return torch.linalg.solve( K_nu_tensor, w )

    def sample_heat(self, v0, nu=2, T=5.0, dt=0.01, w_noise=None):
        #I = torch.eye( self.N, dtype=self.L.dtype )
        #temp = -( self.tau * I + self.L )
        #K_nu_tensor = torch.linalg.matrix_power(temp, nu)

        K_nu_tensor = -self.L


        MAX_ITER = int(T / dt)
        sol = [ v0 ]

        #if(w_noise is None):
        #    w_noise = torch.randn(MAX_ITER, v0.shape[0], dtype=torch.float64)

        for i in range(MAX_ITER):
            # uncomment for smooth noise
            #noise = self.sample_stationary( w_noise[i] , nu=nu)
            #v = v0 - dt * (K_nu_tensor @ v0) + torch.sqrt( torch.tensor(dt) ) * noise

            # uncomment for white noise
            #v = v0 + dt * (K_nu_tensor @ v0) #+ torch.sqrt( torch.tensor(dt) ) * w_noise[i]
            v = torch.linalg.solve( torch.eye(self.N) + dt*K_nu_tensor, v0 )

            v0 = v
            sol.append( v )

        return torch.stack(sol)


class heat_periodic_pytorch_new():
    """
    Class for creating a Laplacian operator on a periodic domain. Code is inspired by the work "Non-separable Spatio-temporal Graph Kernels via
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
        self.dtype = torch.float64
        self.N = N
        self.L_domain = 1.
        self.dx = self.L_domain/self.N
        points = torch.linspace(0,self.L_domain - self.dx,self.N)
        
        # initiating an empty adjacency matrix
        self.W = torch.eye(N-1).to(self.dtype)
   
        self.Dx = torch.zeros(N-1,N)
        # weights on chosen as the distance between neighboring nodes
        self.Dx[0,0] = -1
        self.Dx[0,1] = 1
        for i in range(1,self.N-1):
            self.Dx[i,i] = -1
            self.Dx[i,i+1] = 1
        self.Dx[-1,-2] = -1
        self.Dx[-1,-1] = 1

        self.Dx = self.Dx.to(self.dtype)

        # defining the discretization element: length of a descrete arc
        #Dw = torch.diag( torch.sum(self.W, dim = 1)) # diagonal matrix of accumulated sums
        #self.L = -(Dw - self.W)/self.dx/self.dx # The graph Laplacian operator (differs from the paper by the nomalaization constant 1/dx/dx)
        #self.L[0,-1]=0

        self.L =- self.Dx.T @ self.W @ self.Dx# / self.dx/ self.dx
        self.L[0,0] = -2
        self.L[-1,-1] = -2
        self.L = self.L/self.dx/self.dx

        #self.tau = 1/(0.5*2*np.pi)/(0.5*2*np.pi)
        self.tau = 0.1 # The length scale: the distantce to which nodes are correlated. In the paper tau = 2nu/kappa^2

    def update_L(self, w):
        self.W = torch.diag(w)
        self.L =- self.Dx.T @ self.W @ self.Dx / self.dx/ self.dx
        self.L[0,0] = 1 + w[0]
        self.L[-1,-1] = 1+w[-1]

    def sample_stationary(self, w, nu=2):
        I = torch.eye( self.N, dtype=self.L.dtype )
        temp = ( self.tau * I + self.L  ).to(self.dtype) 
        K_nu_tensor = torch.linalg.matrix_power(temp, nu).to(self.dtype)

        return torch.linalg.solve( K_nu_tensor, w )

    def sample_stationary_with_reaction(self, w, nu=2):
        pass

    def R(self, u):
        pass

    def dR(self, u):
        pass

    def sample_heat(self, v0, nu=0, T=5.0, dt=0.01, w_noise=None):
        #I = torch.eye( self.N, dtype=self.L.dtype )
        #temp = -( self.tau * I + self.L )
        #K_nu_tensor = torch.linalg.matrix_power(temp, nu)

        K_nu_tensor = -self.L


        MAX_ITER = int(T / dt)
        sol = [ v0 ]

        #if(w_noise is None):
        #    w_noise = torch.randn(MAX_ITER, v0.shape[0], dtype=torch.float64)

        for i in range(MAX_ITER):
            # uncomment for smooth noise
            #noise = self.sample_stationary( w_noise[i] , nu=nu)
            #v = v0 - dt * (K_nu_tensor @ v0) + torch.sqrt( torch.tensor(dt) ) * noise

            # uncomment for white noise
            #v = v0 + dt * (K_nu_tensor @ v0) #+ torch.sqrt( torch.tensor(dt) ) * w_noise[i]
            #v = torch.linalg.solve( torch.eye(self.N) + dt*K_nu_tensor, v0 + torch.sqrt(torch.tensor(dt))*w_noise[i] )

            noise = self.sample_stationary( w_noise[i] , nu=nu)
            v = torch.linalg.solve( torch.eye(self.N) + dt*K_nu_tensor, v0 + torch.sqrt(torch.tensor(dt))*noise )
            #v = torch.linalg.solve( torch.eye(self.N).to(self.dtype) + dt*K_nu_tensor, v0 )

            v0 = v
            sol.append( v )

        return torch.stack(sol)

if __name__ == "__main__":
    N = 128
    dx = 1./N

    x = torch.linspace(0, 1.-dx, N)
    input = torch.exp(-0.5 * ((x - 0.5) / 0.05) ** 2)

    prior = heat_periodic_pytorch_new(N)
    
    #out = prior.sample_stationary(input)

    p = torch.randn(N-1)
    prior.update_L( torch.exp(p) )
    
    T_max=2.
    dt = 0.005
    MAX_ITER = int(T_max/dt)
    w = torch.randn(MAX_ITER, N)
    out = prior.sample_heat( input, nu=1, T=T_max, dt=dt, w_noise=w )

    plt.imshow(out.detach().numpy())
    plt.show()
