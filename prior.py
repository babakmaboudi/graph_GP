from decimal import DecimalTuple
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
    def __init__(self, graph, tau=1., dtype=torch.float64): # N is the number of nodes on the circumference of the circle
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
        self.graph = graph
        self.edge_list = list(graph.edges())  # consistent ordering
        self.num_edges = self.graph.number_of_edges()
        self.num_nodes = self.graph.number_of_nodes()

        self.tau=0.1
        self.dtype = dtype


    def get_edge_weights(self) -> torch.Tensor:
        """Get edge weights as a PyTorch tensor."""
        weights = [self.graph[u][v].get("weight", 0.0) for u, v in self.edge_list]
        return torch.tensor(weights, dtype=torch.float64)

    def set_edge_weights(self, weights: torch.Tensor):
        """Set edge weights from a PyTorch tensor."""
        if weights.numel() != len(self.edge_list):
            raise ValueError(f"Expected tensor of size {len(self.edge_list)}, got {weights.numel()}")
        for (u, v), w in zip(self.edge_list, weights.tolist()):
            self.graph[u][v]['weight'] = float(w)

    def update_edge_list(self):
        """Refresh internal edge list (use if graph structure changes)."""
        self.edge_list = list(self.graph.edges())

    def compute_laplacian_from_tensor_autograd(
        self,
        weight_tensor: torch.Tensor,
        normalized: bool = False,
        make_pd: bool = False,
        epsilon: float = 1e-5
    ) -> torch.Tensor:
        """
        Compute a Laplacian matrix using edge weights in a way that supports autograd.
        """
        if weight_tensor.numel() != len(self.edge_list):
            raise ValueError(f"Expected tensor of size {len(self.edge_list)}, got {weight_tensor.numel()}")

        num_nodes = self.graph.number_of_nodes()
        node_list = list(self.graph.nodes())
        node_idx = {node: i for i, node in enumerate(node_list)}

        # Build adjacency matrix using torch (not numpy/scipy!)
        row_idx = []
        col_idx = []
        values = []

        for idx, (u, v) in enumerate(self.edge_list):
            i, j = node_idx[u], node_idx[v]
            row_idx.extend([i, j])
            col_idx.extend([j, i])
            values.extend([weight_tensor[idx], weight_tensor[idx]])  # symmetric

        row_idx = torch.tensor(row_idx, dtype=torch.long)
        col_idx = torch.tensor(col_idx, dtype=torch.long)
        values = torch.stack(values) if isinstance(values[0], torch.Tensor) else torch.tensor(values)

        A = torch.sparse_coo_tensor(
            indices=torch.stack([row_idx, col_idx]),
            values=values,
            size=(num_nodes, num_nodes)
        ).to_dense()

        degrees = A.sum(dim=1)  # Degree vector

        if normalized:
            d_inv_sqrt = torch.pow(degrees + 1e-8, -0.5)
            D_inv_sqrt = torch.diag(d_inv_sqrt)
            self.L = torch.eye(num_nodes) - D_inv_sqrt @ A @ D_inv_sqrt
        else:
            D = torch.diag(degrees)
            self.L = D - A

        if make_pd:
            self.L += epsilon * torch.eye(num_nodes)
        self.L.to(torch.float64)

    def sample_stationary(self, w, nu=2):
        I = torch.eye( 51, dtype=self.L.dtype )
        temp = 1 * ( 1 * I + self.L ).to(self.L.dtype)
        #K_nu_tensor = temp.to(torch.float64)
        K_nu_tensor = torch.linalg.matrix_power(temp, nu ).to(torch.float64)

        return torch.linalg.solve( K_nu_tensor, w )

    def sample_heat(self, v0, nu=2, T=5.0, dt=0.01, w_noise=None):
        I = torch.eye( 51, dtype=self.L.dtype )
        temp = 1. * ( 1. * I + self.L )
        K_nu_tensor = torch.linalg.matrix_power(temp, nu ).to(torch.float64)

        MAX_ITER = int(T / dt)
        sol = [ v0 ]

        #if(w_noise is None):
        #    w_noise = torch.randn(MAX_ITER, v0.shape[0], dtype=torch.float64)

        for i in range(MAX_ITER):
            # uncomment for smooth noise
            #noise = self.sample_stationary( w_noise[i] , nu=nu)
            #v = v0 - dt * (K_nu_tensor @ v0) + torch.sqrt( torch.tensor(dt) ) * noise

            # uncomment for white noise
            #term = w_noise[i]
            term = self.sample_stationary(w_noise[i])
            v = v0 - dt * (K_nu_tensor @ v0) + torch.sqrt( torch.tensor(dt) ) * term#w_noise[i]
            v0 = v
            sol.append( v )

        return torch.stack(sol)

    def sample_stationary_with_linearreaction(self, w, nu=2, beta=2.5, gamma=1.0):
        """
            Sample from the stationary solution of a linear reaction-diffusion system:

                Solves: (tau I + L - tau R_lin)^nu f = w
                where R_lin comes from linearizing R(f) = (beta-gamma)c - beta f^2 around f0:
                R_lin(f) = (-2*beta*f0) * f  ->  R_lin operator = (-2*beta*f0) * I

            where L is the graph Laplacian (possibly scaled), tau is a diffusion scaling parameter,
            R is a linear reaction operator, and ν controls the smoothness (via inverse powers).

            Parameters:
            -----------
            w : torch.Tensor
                Forcing or input term (e.g., white noise sample), shape (N,)
            nu : int, optional
                Power of the inverse operator to apply (default is 2)

            Returns:
            --------
            torch.Tensor
                Solution f of the modified system incorporating linear reaction
        """

        device = self.L.device
        dtype = self.dtype  # or self.L.dtype if you prefer
        N = self.num_nodes
        f0 = (beta-gamma)/beta # equilibrium
        I = torch.eye(self.num_nodes, dtype=self.L.dtype).to(self.dtype)

        # Build diagonal of R_lin: r_i = -2*beta*f0_i
        if torch.is_tensor(f0):
            f0_vec = f0.to(device=device, dtype=dtype).reshape(-1)
            if f0_vec.numel() != N:
                raise ValueError(f"f0 tensor must have shape ({N},), got {tuple(f0.shape)}")
            r_diag = (-2.0 * beta) * f0_vec
            R_lin = torch.diag(r_diag)  # dense; OK for small/medium N
        else:
            r = (-2.0 * beta) * float(f0)
            R_lin = r * I

        A = (self.tau * I + self.L.to(device=device, dtype=dtype) - R_lin).to(dtype)

        # Compute f = A^{-nu} w via repeated solves (numerically safer than forming A^nu)
        x = w
        for _ in range(nu):
            x = torch.linalg.solve(A, x)
        return x

    def sample_stationary_with_nonlinearreaction(
            self, w, nu=2, tol=1e-6, max_iter=20,
            beta=2.5, gamma=1.0,
            alpha=0.008,  # reaction strength (separate from tau)
            mu=0.0,  # leakage / extra damping
            clamp01=True,  # enforce prevalence bounds
            ls_max=10  # line-search steps
    ):
        """
        Solve: ( (tau I + L + mu I)^nu ) f = w + alpha * R(f)

        Newton on: G(f) = K f - alpha R(f) - w = 0
        with:      J(f) = K - alpha R'(f)

        tau controls correlation length / smoothing.
        alpha controls reaction strength.
        mu adds extra damping (helps stability).
        What to do when you detect instability
        •	turn on damped Newton / line search
        •	lower alpha (×0.1)
        •	add leakage mu (e.g. mu=1e-2 then 1e-1)
        •	clamp f to [0,1] if it’s a ratio
        •	start closer to equilibrium instead of zeros
        """
        device = self.L.device
        dtype = self.dtype
        N = self.num_nodes

        I = torch.eye(N, device=device, dtype=dtype)
        w = w.to(device=device, dtype=dtype)

        # Build K = (A)^nu with A = tau I + L + mu I
        A = (self.tau + mu) * I + self.L.to(device=device, dtype=dtype)
        K = torch.linalg.matrix_power(A, nu)

        # initial guess
        f = torch.zeros_like(w, dtype=dtype, device=device)

        for i in range(max_iter):
            # Residual at current f
            Rf = self.Rf(f, beta, gamma)  # shape (N,)
            G = K @ f - alpha * Rf - w  # <-- alpha, NOT tau

            Gnorm = torch.linalg.norm(G)
            if Gnorm < tol:
                break

            # Reaction derivative (diagonal for local reactions)
            dRdf = self.dRdf_diag(f, beta, gamma)
            if dRdf.ndim == 2:
                dRdf_vec = torch.diagonal(dRdf)
            else:
                dRdf_vec = dRdf.reshape(-1)

            if dRdf_vec.numel() != N:
                raise ValueError(f"dRdf has wrong size: got {dRdf_vec.numel()}, expected {N}. "
                                 f"Original shape was {tuple(dRdf.shape)}")

            # Jacobian: J = K - alpha * diag(dR/df)
            # (efficient diagonal update; avoids torch.diag)
            J = K.clone()
            idx = torch.arange(N, device=device)
            J[idx, idx] -= alpha * dRdf_vec

            dRdf_vec = (beta - gamma) - 2 * beta * f  # if SIS reaction

            # If alpha*pos/lam_min is near 1 (or >1), it’s exactly why J loses SPD and Newton struggles.
            # pos = torch.clamp(dRdf_vec, min=0).max().item()
            # lam_min = torch.linalg.eigvalsh(K).min().item()
            # print("max positive R'(f):", pos, "lambda_min(K):", lam_min, "ratio:", alpha * pos / lam_min)

            try:
                torch.linalg.cholesky(J)
                spd = True
            except RuntimeError:
                spd = False
                print(i, "J SPD?", spd)
            # Newton direction
            delta = torch.linalg.solve(J, G)

            # ---- Damped Newton / backtracking line search ----
            step = 1.0
            f_new = f
            for _ in range(ls_max):
                candidate = f - step * delta
                if clamp01:
                    candidate = candidate.clamp(0.0, 1.0)

                Rc = self.Rf(candidate, beta, gamma)
                Gc = K @ candidate - alpha * Rc - w

                if torch.linalg.norm(Gc) < (1.0 - 1e-4 * step) * Gnorm:
                    f_new = candidate
                    break

                step *= 0.5

            f = f_new

        return f
    # def sample_stationary_with_nonlinearreaction(self, w, nu=2, tol=1e-6, max_iter=20, beta=2.5, gamma=1.0):
    #     """
    #         Sample from the stationary solution of a nonlinear reaction-diffusion system using Newton iteration:
    #
    #             (τ I + L)^ν f = w + alpha*R(f)
    #             Newton on: G(f) = K f - tau R(f) - w = 0
    #             with:      J(f) = K - tau R'(f)
    #
    #         where L is the graph Laplacian (already population-scaled), R(f) is a nonlinear reaction term,
    #         and ν controls the smoothing behavior (fractional diffusion). This function solves for f iteratively
    #         using Newton's method.
    #
    #         Parameters:
    #         -----------
    #         w : torch.Tensor
    #             Input (e.g., noise sample), shape (N,)
    #         nu : int, optional
    #             Power of the smoothing operator (default: 2)
    #         tol : float, optional
    #             Tolerance for Newton iteration convergence (default: 1e-6)
    #         max_iter : int, optional
    #             Maximum number of Newton iterations (default: 20)
    #
    #         Returns:
    #         --------
    #         torch.Tensor
    #             Approximate solution f of the nonlinear stationary PDE
    #     """
    #
    #     I = torch.eye(self.num_nodes, dtype=self.L.dtype).to(self.dtype)
    #
    #     A = self.tau * I + self.L #L is already scaled by population
    #     K = torch.linalg.matrix_power(A, nu)
    #
    #     # initial guess
    #     f = torch.zeros_like(w, dtype=self.dtype)
    #     alpha=1e-2
    #     for i in range(max_iter):
    #         Rf = self.Rf(f, beta, gamma)  # shape (N,)
    #         # Residual: G(f) = K f - R(f) - w
    #         G = K @ f - alpha*Rf - w
    #
    #         if torch.linalg.norm(G) < tol:
    #             break
    #
    #         # R'(f): if your reaction is local, this is diagonal as a vector
    #         # dRdf_vec should be shape (N,), representing derivative at each node
    #         dRdf = self.dRdf_diag(f, beta, gamma)
    #
    #         # Make dRdf_vec be length N (diagonal entries)
    #         if dRdf.ndim == 2:
    #             # dRdf is N×N, take its diagonal
    #             dRdf_vec = torch.diagonal(dRdf)
    #         else:
    #             # dRdf is already length N (or N×1)
    #             dRdf_vec = dRdf.reshape(-1)
    #
    #         # Sanity check
    #         if dRdf_vec.numel() != self.num_nodes:
    #             raise ValueError(f"dRdf has wrong size: got {dRdf_vec.numel()}, expected {self.num_nodes}. "
    #                              f"Original shape was {tuple(dRdf.shape)}")
    #
    #         J = K - self.tau * torch.diag(dRdf_vec)
    #
    #         # Newton step: J delta = G
    #         delta = torch.linalg.solve(J, G)
    #         f = f - delta
    #
    #     return f


    def Rf(self, f, beta, gamma):
        """
            Nonlinear reaction function R(f) for the reaction-diffusion model.

            Implements:
                R(f) = (β - γ)f - β f²

            This models logistic-type growth with saturation.

            Parameters:
            -----------
            f : torch.Tensor
                Current state vector, shape (N,)
            beta : float
                Reaction rate coefficient (birth rate)
            gamma : float
                Decay or death rate coefficient

            Returns:
            --------
            torch.Tensor
                Nonlinear reaction term R(f), shape (N,)
        """
        R = (beta - gamma) * f - beta * f ** 2
        return R


    def dRdf_diag(self, f, beta, gamma):
        """
            Diagonal Jacobian of the nonlinear reaction function R(f), used in Newton iteration.

            Computes:
                dR/df = (β - γ) - 2βf

            Returns a diagonal matrix with the partial derivatives for each node.

            Parameters:
            -----------
            f : torch.Tensor
                Current state vector, shape (N,)
            beta : float
                Reaction rate coefficient
            gamma : float
                Decay or death rate coefficient

            Returns:
            --------
            torch.Tensor
                Diagonal matrix of dR/df, shape (N, N)
        """
        dR_diag = torch.diag((beta - gamma) - 2 * beta * f)
        return dR_diag


    def sample_heat_with_linearreaction(
        self, v0, nu = 0, T = 5.0, dt = 0.01, w_noise = None,
        beta = 10.0, gamma = 1.0,
        clamp01 = True
    ):
        """
        Linearized SIS reaction-diffusion dynamics on a graph:

            f_dot = -L f + a f + b   (+ noise)

        with a = R'(f0), b = R(f0) - a f0 (constant term).
        Implicit Euler:
            (I + dt L - dt a I) f_{n+1} = f_n + dt b + sqrt(dt) xi_n
        """
        device = self.L.device
        dtype = self.dtype
        N = self.num_nodes

        L = self.L.to(device=device, dtype=dtype)
        I = torch.eye(N, device=device, dtype=dtype)

        # choose linearization point
        f0 = (beta - gamma) / beta  # endemic equilibrium for SIS logistic

        # SIS reaction and derivative
        # R(f) = (beta-gamma) f - beta f^2
        # R'(f) = (beta-gamma) - 2 beta f
        if torch.is_tensor(f0):
            f0_vec = f0.to(device=device, dtype=dtype).reshape(-1)
            a_vec = (beta - gamma) - 2.0 * beta * f0_vec  # length N
            b_vec = ((beta - gamma) * f0_vec - beta * f0_vec ** 2) - a_vec * f0_vec
        else:
            f0s = float(f0)
            a = (beta - gamma) - 2.0 * beta * f0s
            b = ((beta - gamma) * f0s - beta * f0s ** 2) - a * f0s
            a_vec = a * torch.ones(N, device=device, dtype=dtype)
            b_vec = b * torch.ones(N, device=device, dtype=dtype)

        dt_t = torch.tensor(dt, device=device, dtype=dtype)
        steps = int(T / dt)

        f = v0.to(device=device, dtype=dtype)
        sol = [f]

        for n in range(steps):
            # noise term
            rhs = f + dt_t * b_vec
            if w_noise is not None:
                xi = w_noise[n].to(device=device, dtype=dtype)
                rhs = rhs + torch.sqrt(dt_t) * xi

            # system matrix: I + dt L - dt a I
            A = I + dt_t * L - dt_t * torch.diag(a_vec)

            f = torch.linalg.solve(A, rhs)

            if clamp01:
                f = f.clamp(0.0, 1.0)

            sol.append(f)

        return torch.stack(sol)

    # def sample_heat_with_nonlinreaction(self, v0, nu=0, T=5.0, dt=0.01, w_noise=None):
    #     """
    #     Simulate the time evolution of a nonlinear reaction-diffusion system using an implicit Euler scheme.
    #
    #     Solves:
    #         f_{n+1} = (I + dt * L)^(-1) [f_n + dt * R(f_n) + sqrt(dt) * ξ_n]
    #
    #     where L is the graph Laplacian, R(f) is a nonlinear reaction term, and ξ_n is noise.
    #     The Laplacian is assumed to already be scaled appropriately by population or spatial weights.
    #
    #     Parameters:
    #     -----------
    #     v0 : torch.Tensor
    #         Initial condition, shape (N,)
    #     nu : int, optional
    #         Smoothness parameter for spatial noise sampling (default: 0, i.e., white noise)
    #     T : float, optional
    #         Final simulation time (default: 5.0)
    #     dt : float, optional
    #         Time step size (default: 0.01)
    #     w_noise : torch.Tensor or None, optional
    #         Optional precomputed noise tensor of shape (T/dt, N); if None, noise is not added.
    #
    #     Returns:
    #     --------
    #     torch.Tensor
    #         Time evolution of the state, shape (T/dt + 1, N)
    #     """
    #     beta = 10.
    #     gamma = 1.0
    #     #I = torch.eye( self.N, dtype=self.L.dtype )
    #     #temp = -( self.tau * I + self.L )
    #     #K_nu_tensor = torch.linalg.matrix_power(temp, nu)
    #
    #     K_nu_tensor = -self.L
    #
    #
    #     MAX_ITER = int(T / dt)
    #     sol = [v0]
    #
    #     for i in range(MAX_ITER):
    #         f_n = v0
    #
    #         dt_tensor = torch.tensor(dt, dtype=self.dtype)
    #
    #         A = torch.eye(self.num_nodes, dtype=self.dtype) + dt_tensor * K_nu_tensor
    #         Rf_n = self.Rf(f_n, beta, gamma).to(self.dtype)
    #         rhs = f_n + dt_tensor * Rf_n
    #
    #         if w_noise is not None:
    #             noise = self.sample_stationary(w_noise[i], nu=nu).to(self.dtype)
    #             rhs += torch.sqrt(dt_tensor) * noise
    #
    #         # Now A and rhs are both of dtype self.dtype (e.g., torch.float64)
    #         f_next = torch.linalg.solve(A, rhs)
    #
    #         v0 = f_next
    #         sol.append(f_next)
    #
    #     return torch.stack(sol)


    def sample_heat_with_nonlinreaction(
        self, v0, nu=0, T=5.0, dt=0.01, w_noise=None,
        beta=10.0, gamma=1.0,
        newton_tol=1e-8, newton_max_iter=20,
        ls_max=10, clamp01=True
        ):
        """
        Fully implicit Euler for diffusion + reaction with damped Newton:

            (I + dt*L) f_{n+1} - dt*R(f_{n+1}) = f_n + sqrt(dt)*xi_n

        Newton on: G(f) = (I + dt*L)f - dt*R(f) - b = 0
        J(f) = (I + dt*L) - dt*R'(f)
        """
        device = self.L.device
        dtype = self.dtype
        N = self.num_nodes

        L = self.L.to(device=device, dtype=dtype)

        # If your diffusion sign convention is different, adjust here.
        # For PDE: c_t = div(...) + R(c), the discrete form is usually c_t = -L c + R(c)
        # In that case you'd use (I + dt*Laplacian_operator) accordingly.
        # I'll follow your current "K_nu_tensor = -L" approach:
        K = (-L)

        I = torch.eye(N, device=device, dtype=dtype)
        dt_t = torch.tensor(dt, device=device, dtype=dtype)

        MAX_STEPS = int(T / dt)
        sol = [v0.to(device=device, dtype=dtype)]

        for n in range(MAX_STEPS):
            f_n = sol[-1]

            # Build b = f_n + sqrt(dt)*noise
            b = f_n.clone()

            if w_noise is not None:
                noise = self.sample_stationary(w_noise[n], nu=nu).to(device=device, dtype=dtype)
                b = b + torch.sqrt(dt_t) * noise

            # Nonlinear solve for f_{n+1}
            # Start guess: previous state (good for small dt)
            f = f_n.clone()

            # Constant part of Jacobian: A = I + dt*K
            A = I + dt_t * K

            for it in range(newton_max_iter):
                Rf = self.Rf(f, beta, gamma).to(device=device, dtype=dtype)

                # Residual: G(f) = A f - dt*R(f) - b
                G = A @ f - dt_t * Rf - b
                Gnorm = torch.linalg.norm(G)

                if Gnorm < newton_tol:
                    break

                dRdf = self.dRdf_diag(f, beta, gamma)
                if dRdf.ndim == 2:
                    dRdf_vec = torch.diagonal(dRdf)
                else:
                    dRdf_vec = dRdf.reshape(-1)

                if dRdf_vec.numel() != N:
                    raise ValueError(f"dRdf has wrong size: got {dRdf_vec.numel()}, expected {N}. "
                                     f"Original shape was {tuple(dRdf.shape)}")

                # Jacobian: J = A - dt*diag(dR/df)
                J = A.clone()
                idx = torch.arange(N, device=device)
                J[idx, idx] -= dt_t * dRdf_vec

                delta = torch.linalg.solve(J, G)

                # Damped Newton / backtracking
                step = 1.0
                accepted = False
                for _ in range(ls_max):
                    cand = f - step * delta
                    if clamp01:
                        cand = cand.clamp(0.0, 1.0)

                    Rc = self.Rf(cand, beta, gamma).to(device=device, dtype=dtype)
                    Gc = A @ cand - dt_t * Rc - b

                    if torch.linalg.norm(Gc) < (1.0 - 1e-4 * step) * Gnorm:
                        f = cand
                        accepted = True
                        break

                    step *= 0.5

                if not accepted:
                    # If line search fails, take a tiny step to avoid stalling
                    f = (f - 1e-3 * delta).clamp(0.0, 1.0) if clamp01 else (f - 1e-3 * delta)

            sol.append(f)

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

    def sample(self, num_samples):
        return self.dist.sample(num_samples,)


    def find_dist(self, i1,i2,i3,i4):
        d1 = nx.shortest_path_length(self.graph, source=self.nodes[i1], target=self.nodes[i3])
        d2 = nx.shortest_path_length(self.graph, source=self.nodes[i1], target=self.nodes[i4])
        d3 = nx.shortest_path_length(self.graph, source=self.nodes[i2], target=self.nodes[i3])
        d4 = nx.shortest_path_length(self.graph, source=self.nodes[i2], target=self.nodes[i4])
        return np.min( np.array([d1,d2,d3,d4]) )
