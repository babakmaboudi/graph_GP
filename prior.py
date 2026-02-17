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


    def sample_stationary_with_linearreaction(self, w, nu=2, beta=2.5, gamma=1.0, alpha=1):
        """
        Sample from the stationary solution of a linear reaction-diffusion system:

            Linearization: R(f) ≈ c_const + R'(f0) f
              R'(f0) = (beta-gamma) - 2*beta*f0
              c_const = beta*f0^2
            with f0 chosen as the endemic equilibrium (beta>gamma):
              f0 = (beta-gamma)/beta

            We solve: (K + alpha*R'(f0) I) f = w - alpha*c_const*1,
            where K = (tau I + L)^nu.


        where L is the graph Laplacian (possibly scaled), tau is a diffusion scaling parameter,
        R is a linear reaction operator, and ν controls the smoothness (via inverse powers).

        Parameters:
        -----------
        w : torch.Tensor
            Forcing or input term (e.g., white noise sample), shape (N,)
        nu : int, optional
            Power of the inverse operator to apply (default is 2)
        beta : float
               Reaction rate coefficient (birth rate)
        gamma : float
                Decay or death rate coefficient
        alpha: float
               weight of reaction term

        Returns:
        --------
        torch.Tensor
            Solution f of the modified system incorporating linear reaction
    """

        device = self.L.device
        dtype = self.dtype
        N = self.num_nodes

        I = torch.eye(N, device=device, dtype=dtype)
        w = w.to(device=device, dtype=dtype)

        # linearization point (you chose endemic equilibrium for SIS logistic)
        # endemic equilibrium (beta > gamma)
        f0 = float((beta - gamma) / beta)

        # Linearized SIS slope a = R'(f0)
        # For SIS: R(f) = (beta-gamma)f - beta f^2  -> R'(f) = (beta-gamma) - 2 beta f
        # Linearized: R(f) = c_const + R'(f0) * f, R'(f0) = (beta -gamma) -2*beta*f0
        Rprime_f0 = (beta - gamma) - 2.0 * beta * f0
        c_const = beta * f0 * f0

        # K = (tau I + L)^nu, (swap 1.0 for self.tau if needed)
        A = 1 * I + self.L.to(device=device, dtype=dtype)
        K = torch.linalg.matrix_power(A, nu)

        # M = K - alpha R_lin
        M = K - alpha * Rprime_f0 * torch.eye(N, device=device, dtype=dtype)

        # Solve M f = w
        # (K - alpha R'(f0) I) f = w + alpha c_const * 1
        rhs = w + alpha*c_const*torch.ones(N, device=device, dtype=dtype) #constant part of R
        f = torch.linalg.solve(M, rhs)
        return f


    def sample_stationary_with_nonlinearreaction(
            self, w, nu=2, tol=1e-6, max_iter=20,
            beta=2.5, gamma=1.0,
            alpha=1e-2,  # reaction strength
            clamp01=False,  # enforce prevalence bounds
            use_linesearch=False, #activate linesearch
            ls_max=10  # line-search steps
    ):
        """
        Solve: ( (tau I + L)^nu ) f = w - alpha * R(f)

        Newton on: G(f) = K f - alpha R(f) - w = 0
        with:      J(f) = K + alpha R'(f)
        Newton iteration:
            J(f_k) delta_k = G(f_k),
            f_{k+1} = f_k + delta_k,
        with optional damped Newton (backtracking line search).
        L is the graph Laplacian (possibly scaled),
        tau is a diffusion scaling parameter controlling correlation length /smoothing,
        R is a linear reaction operator, and ν controls the smoothness (via inverse powers),
        alpha controls reaction strength.
        What to do when you detect instability
        •	turn on damped Newton / line search
        •	lower alpha (×0.1)
        •	clamp f to [0,1] if it’s a ratio
        •	start closer to equilibrium instead of zeros

        Parameters:
        -----------
        w : torch.Tensor
            Forcing or input term (e.g., white noise sample),
            shape (N,).
        nu : int, optional
            Power of the smoothing operator (tau I + L),
            controls spatial smoothness / correlation length.
            Default is 2.
        tol : float, optional
            Absolute tolerance for Newton convergence.
            Iteration stops when ||G(f)|| < tol.
            Default is 1e-6.
        max_iter : int, optional
            Maximum number of Newton iterations.
            Default is 20.
        beta : float
            Reaction rate coefficient (infection/contact rate).
        gamma : float
            Recovery or decay rate coefficient.
        alpha : float
            Weight (strength) of the reaction term R(f)
            relative to the diffusion/smoothing operator.
        clamp01 : bool, optional
            If True, clamp the solution f to the interval [0, 1]
            after each Newton update. Useful when f represents
            a population ratio or probability.
            Default is False.
        use_linesearch : bool, optional
            If True, use a damped Newton method with backtracking
            line search (Armijo condition) to improve robustness
            for stiff or unstable problems.
            Default is False.
        ls_max : int, optional
            Maximum number of backtracking steps in the line search.
            Default is 10.

        Returns:
        --------
        torch.Tensor
            Solution f of the modified system incorporating nonlinear reaction

        """
        device = self.L.device
        dtype = self.dtype
        N = self.num_nodes

        I = torch.eye(N, device=device, dtype=dtype)
        w = w.to(device=device, dtype=dtype)

        # Build K = (A)^nu with A = tau I + L
        # (swap 1.0 for self.tau if needed)
        A = 1  * I + self.L.to(device=device, dtype=dtype)
        K = torch.linalg.matrix_power(A, nu)

        # initial guess
        f = torch.zeros_like(w, dtype=dtype, device=device)

        for i in range(max_iter):
            # Residual at current f
            Rf = self.Rf(f, beta, gamma)  # shape (N,)
            G = K @ f - alpha * Rf - w

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

            # check if J spd for debugging:
            # try:
            #     torch.linalg.cholesky(J)
            #     spd = True
            # except RuntimeError:
            #     spd = False
            #     print(i, "J SPD?", spd)

            # Newton direction
            delta = torch.linalg.solve(J, G)

            if not use_linesearch:
                # plain Newton
                f = f - delta
                if clamp01:
                    f = f.clamp(0.0, 1.0)
                continue

            # optional damped Newton / backtracking line search
            step = 1.0
            accepted = False
            for _ in range(ls_max):
                f_cand = f - step * delta
                if clamp01:
                    f_cand = f_cand.clamp(0.0, 1.0)

                Rc = self.Rf(f_cand, beta, gamma).to(device=device, dtype=dtype)
                Gc = K @ f_cand - alpha * Rc - w

                # Armijo (sufficient decrease) condition:
                # 1e-4 small constant = how much decrease is enough to accept Newton step
                if torch.linalg.norm(Gc) < (1.0 - 1e-4 * step) * Gnorm:
                    f = f_cand
                    accepted = True
                    break

                step *= 0.5

            if not accepted:
                # fallback: tiny step to avoid stalling
                f = f - 1e-3 * delta
                if clamp01:
                    f = f.clamp(0.0, 1.0)

        return f


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


    def sample_heat_with_linearreaction(self, v0, nu=2, T=5.0, dt=0.01, w_noise=None,
                                        alpha=1, beta=2.5, gamma=1.0, f0=None, clamp01=False):
        """
       Simulate the time evolution of a linearized reaction–diffusion system on a graph
       using an explicit Euler scheme.

       This implements a linear (affine) approximation of the SIS reaction term
       R(f) = (beta-gamma) f - beta f^2 around a reference value f0:

           R(f) ≈ c_const + R'(f0) * f,

       where
           R'(f0) = (beta-gamma) - 2*beta*f0,
           c_const = R(f0) - R'(f0)*f0.
       For the SIS logistic reaction, this constant simplifies to c_const = beta*f0^2.

       The dynamics simulated are (in discrete time, explicit Euler):

           f_{n+1} = f_n
                    - dt * (K_eff f_n)
                    + dt * alpha * c_const * 1
                    + sqrt(dt) * omega_n,

       with
           K = (I + L)^nu                (replace I with tau I if desired),
           K_eff = K - alpha * R'(f0) I,

       and omega_n is Gaussian noise (white, or optionally smoothed via sample_stationary).

       Parameters
       ----------
       v0 : torch.Tensor
           Initial condition, shape (N,).
       nu : int, optional
           Power used to form the smoothing/diffusion operator K = (I + L)^nu.
           Controls spatial smoothing / correlation structure. Default is 2.
       T : float, optional
           Final simulation time. Default is 5.0.
       dt : float, optional
           Time step size for explicit Euler. Default is 0.01.
       w_noise : torch.Tensor or None, optional
           If provided, should have shape (T/dt, N) and contain Gaussian samples to drive
           the noise. In this implementation, each sample is optionally mapped through
           self.sample_stationary(...) to produce spatially smoothed noise.
           If None, standard Gaussian noise is generated internally. Default is None.
       alpha : float, optional
           Weight (strength) of the reaction term relative to diffusion/smoothing.
           Default is 1.0.
       beta : float, optional
           SIS infection/contact rate parameter in the reaction term. Default is 2.5.
       gamma : float, optional
           SIS recovery/decay rate parameter in the reaction term. Default is 1.0.
       f0 : float or None, optional
           Linearization point for the reaction term.
           If None, uses the endemic equilibrium f0 = (beta-gamma)/beta (for beta > gamma).
           Default is None.
       clamp01 : bool, optional
           If True, clamp the solution f to the interval [0, 1]
           after each Newton update. Useful when f represents
           a population ratio or probability.
           Default is True.

       Returns
       -------
       torch.Tensor
           Time series of the simulated state, shape (steps + 1, N),
           where steps = int(T/dt). The first entry is the initial condition v0.

       Notes
       -----
       - This is an explicit Euler scheme. Stability may require small dt,
         especially if K_eff has large eigenvalues or alpha is large.
       - If f represents a prevalence/ratio, you may want to clamp the output to [0, 1]
         after each step (not done here).
       - If you want white-in-space noise, replace `term = self.sample_stationary(w_noise[i])`
         with `term = w_noise[i]` (or torch.randn).
        """

        device = self.L.device
        dtype = torch.float64  # keep your choice consistent

        N = v0.shape[0] #51
        I = torch.eye(N, device=device, dtype=self.L.dtype)

        # Base operator (you currently use I + L; replace 1. with self.tau if you want)
        temp = 1.0 * (1.0 * I + self.L.to(device=device))
        K = torch.linalg.matrix_power(temp, nu).to(dtype)

        # Linearized SIS slope a = R'(f0)
        # For SIS: R(f) = (beta-gamma)f - beta f^2  -> R'(f) = (beta-gamma) - 2 beta f
        if f0 is None:
            f0 = (beta - gamma) / beta  # endemic equilibrium of logistic SIS
        f0 = float(f0)

        # SIS reaction: R(f)=(beta-gamma)f - beta f^2
        #R_f0 = (beta - gamma) * f0 - beta * (f0 ** 2)
        Rprime_f0 = (beta - gamma) - 2.0 * beta * f0

        # constant term in affine linearization: c_const = R(f0) - R'(f0)*f0
        c_const = beta*f0*f0 #=R_f0 - Rprime_f0 * f0

        # effective drift operator
        K_eff = K - alpha * Rprime_f0 * torch.eye(N, device=device, dtype=dtype)

        MAX_ITER = int(T / dt)
        sol = [v0.to(device=device, dtype=dtype)]

        dt_t = torch.tensor(dt, device=device, dtype=dtype)
        sqrt_dt = torch.sqrt(dt_t)

        ones = torch.ones(N, device=device, dtype=dtype)

        #explicit Euler:
        for i in range(MAX_ITER):
            v = sol[-1]

            if w_noise is None:
                term = torch.randn(N, device=device, dtype=dtype)
            else:
                term = self.sample_stationary(w_noise[i]).to(device=device, dtype=dtype)

            v_next = v - dt_t * (K_eff @ v) + sqrt_dt * term

            #include constant part of reaction term
            v_next = v_next + dt_t * alpha * c_const * ones

            # clamp to physical bounds if requested
            if clamp01:
                v_next = v_next.clamp(0.0, 1.0)

            sol.append(v_next)

        return torch.stack(sol)


    def sample_heat_with_linearreaction_implicit(self, v0, nu=2, T=5.0, dt=0.01, w_noise=None,
                                                 alpha=1.0, beta=2.5, gamma=1.0, f0=None, clamp01=False):
        """
        Simulate the time evolution of a linearized (affine) reaction–diffusion system on a graph
        using an implicit Euler scheme.

        This uses an affine linearization of the SIS reaction term
            R(f) = (beta-gamma) f - beta f^2
        around a reference value f0:

            R(f) ≈ c_const + R'(f0) * f,

        where
            R'(f0) = (beta-gamma) - 2*beta*f0,
            c_const = R(f0) - R'(f0)*f0.
        For the SIS logistic reaction, this constant simplifies to:
            c_const = beta*f0^2.

        The (semi-discrete) model being stepped is:

            df/dt = -K_eff f + alpha*c_const*1 + noise,

        where
            K = (I + L)^nu              (replace I with tau I if desired),
            K_eff = K - alpha*R'(f0) I.

        Implicit Euler step:

            (I + dt*K_eff) f_{n+1} = f_n + dt*alpha*c_const*1 + sqrt(dt)*omega_n,

        so each time step requires solving one linear system with the (time-independent) matrix
            M = I + dt*K_eff.

        Parameters
        ----------
        v0 : torch.Tensor
            Initial condition, shape (N,).
        nu : int, optional
            Power used to form the smoothing/diffusion operator K = (I + L)^nu.
            Controls spatial smoothing / correlation structure. Default is 2.
        T : float, optional
            Final simulation time. Default is 5.0.
        dt : float, optional
            Time step size for implicit Euler. Default is 0.01.
        w_noise : torch.Tensor or None, optional
            If provided, should have shape (T/dt, N) and contain Gaussian samples omega_n.
            In this implementation, each sample may be mapped through self.sample_stationary(...)
            to produce spatially smoothed noise (depending on your implementation).
            If None, standard Gaussian noise is generated internally. Default is None.
        alpha : float, optional
            Weight (strength) of the reaction term relative to diffusion/smoothing.
            Default is 1.0.
        beta : float, optional
            SIS infection/contact rate parameter. Default is 2.5.
        gamma : float, optional
            SIS recovery/decay rate parameter. Default is 1.0.
        f0 : float or None, optional
            Linearization point for the reaction term.
            If None, uses the endemic equilibrium f0 = (beta-gamma)/beta (for beta > gamma).
            Default is None.
        clamp01:bool, optional
            If True, clamp the solution f to the interval [0, 1]
            after each Newton update. Useful when f represents
            a population ratio or probability.
            Default is True.

        Returns
        -------
        torch.Tensor
            Time series of the simulated state, shape (steps + 1, N),
            where steps = int(T/dt). The first entry is the initial condition v0.

        Notes
        -----
        - Implicit Euler is typically more stable than explicit Euler for stiff diffusion
          or strong damping (large eigenvalues in K_eff), allowing larger dt.
        - If f represents a prevalence/ratio, you may want to clamp the output to [0, 1]
             after each step (not done here).
        - If you want white-in-space noise, replace `term = self.sample_stationary(w_noise[i])`
             with `term = w_noise[i]` (or torch.randn).
        """

        device = self.L.device
        dtype = torch.float64

        N = v0.shape[0]
        I = torch.eye(N, device=device, dtype=dtype)

        # K = (I + L)^nu  (swap 1.0 for self.tau if needed)
        B = 1.0 * I + self.L.to(device=device, dtype=dtype)
        K = torch.linalg.matrix_power(B, nu).to(dtype)

        if f0 is None:
            f0 = (beta - gamma) / beta
        f0 = float(f0)

        # SIS: R(f)=(beta-gamma)f - beta f^2
        Rprime_f0 = (beta - gamma) - 2.0 * beta * f0

        # affine constant term in linearization:
        # c_const = R(f0) - R'(f0)*f0 = beta f0^2
        c_const = beta * f0 * f0

        K_eff = (K - alpha * Rprime_f0 * I)

        dt_t = torch.tensor(dt, device=device, dtype=dtype)
        sqrt_dt = torch.sqrt(dt_t)
        ones = torch.ones(N, device=device, dtype=dtype)

        steps = int(T / dt)
        sol = [v0.to(device=device, dtype=dtype)]

        # Precompute the implicit system matrix: M = I + dt K_eff
        M = I + dt_t * K_eff

        for i in range(steps):
            v = sol[-1]

            if w_noise is None:
                term = torch.randn(N, device=device, dtype=dtype)
            else:
                term = self.sample_stationary(w_noise[i]).to(device=device, dtype=dtype)

            rhs = v + dt_t * alpha * c_const * ones + sqrt_dt * term

            v_next = torch.linalg.solve(M, rhs)

            #clamp to physical bounds if requested
            if clamp01:
                v_next = v_next.clamp(0.0, 1.0)

            sol.append(v_next)

        return torch.stack(sol)


    def sample_heat_with_nonlinreaction(
            self, v0, nu=2, T=5.0, dt=0.01, w_noise=None,
            beta=2.5, gamma=1.0,
            alpha=1e-2,
            newton_tol=1e-6, newton_max_iter=20,
            clamp01=False,
            use_linesearch=True, ls_max=10
    ):
        """
        Time-dependent nonlinear reaction-diffusion (implicit Euler):

            f_{n+1} = f_n + dt*(-K f_{n+1} + alpha R(f_{n+1})) + sqrt(dt)*omega_n
            where K = (tau I + L)^nu

        Per step, solve:
            G(f) = (I + dt K) f - dt alpha R(f) - b = 0
            b = f_n + sqrt(dt)*omega_n
        Newton:
            J(f) = (I + dt K) - dt alpha d/df(R(f))
                w : torch.Tensor
            Forcing or input term (e.g., white noise sample),
            shape (N,).

        Parameters
        ----------
        v0 : torch.Tensor
            Initial condition for the state variable f,
            shape (N,).
        nu : int, optional
            Power of the smoothing operator (tau I + L),
            controlling spatial smoothness / correlation length.
            Default is 2.
        T : float, optional
            Final simulation time.
            Default is 5.0.
        dt : float, optional
            Time step size for the implicit Euler scheme.
            Default is 0.01.
        w_noise : torch.Tensor or None, optional
            Stochastic driving noise. If provided, should have
            shape (T/dt, N) and represent independent standard
            Gaussian samples omega_n. If None, white noise
            is generated internally.
            Default is None.
        beta : float
            Reaction rate coefficient (infection/contact rate).
        gamma : float
            Recovery or decay rate coefficient.
        alpha : float
            Weight (strength) of the reaction term R(f)
            relative to the diffusion/smoothing operator.
        newton_tol : float, optional
            Absolute tolerance for Newton convergence at each
            time step. Iteration stops when ||G(f)|| < newton_tol.
            Default is 1e-6.
        newton_max_iter : int, optional
            Maximum number of Newton iterations per time step.
            Default is 20.
        clamp01 : bool, optional
            If True, clamp the solution f to the interval [0, 1]
            after each Newton update. Useful when f represents
            a population ratio or probability.
            Default is True.
        use_linesearch : bool, optional
            If True, use damped Newton with backtracking line search
            (Armijo condition) to improve robustness for stiff or
            strongly nonlinear reactions.
            Default is True.
        ls_max : int, optional
            Maximum number of backtracking steps in the line search.
            Default is 10.

        Returns
        -------
        torch.Tensor
            Time evolution of the solution f with nonlinear reaction term, with shape
            (T/dt + 1, N), including the initial condition.
        """
        device = self.L.device
        dtype = self.dtype
        N = self.num_nodes

        I = torch.eye(N, device=device, dtype=dtype)

        # Build K = (tau I + L)^nu once (time-independent)(swap 1.0 for self.tau if needed)
        A = 1.0 * I + self.L.to(device=device, dtype=dtype)
        K = torch.linalg.matrix_power(A, nu)

        steps = int(T / dt)
        dt_t = torch.tensor(dt, device=device, dtype=dtype)
        sqrt_dt = torch.sqrt(dt_t)

        f = v0.to(device=device, dtype=dtype)
        sol = [f]

        # Constant part in Jacobian: B = I + dt K
        B = I + dt_t * K

        for i in range(steps):
            # noise for this step
            if w_noise is None:
                omega = torch.randn(N, device=device, dtype=dtype)
            else:
                omega = self.sample_stationary(w_noise[i]).to(device=device, dtype=dtype)

            b = f + sqrt_dt * omega  # RHS for the implicit step

            # initial guess for Newton: previous state
            x = f.clone()

            for it in range(newton_max_iter):
                Rx = self.Rf(x, beta, gamma).to(device=device, dtype=dtype)

                # residual: G(x) = (I + dt K)x - dt alpha R(x) - b
                G = B @ x - dt_t * alpha * Rx - b
                Gnorm = torch.linalg.norm(G)

                if Gnorm < newton_tol:
                    break

                dRdf = self.dRdf_diag(x, beta, gamma)
                if dRdf.ndim == 2:
                    dRdf_vec = torch.diagonal(dRdf)
                else:
                    dRdf_vec = dRdf.reshape(-1)

                if dRdf_vec.numel() != N:
                    raise ValueError(f"dRdf has wrong size: got {dRdf_vec.numel()}, expected {N}. "
                                     f"Original shape was {tuple(dRdf.shape)}")

                # Jacobian: J = B - dt alpha diag(dRdf_vec)
                J = B.clone()
                idx = torch.arange(N, device=device)
                J[idx, idx] -= dt_t * alpha * dRdf_vec

                delta = torch.linalg.solve(J, G)

                if not use_linesearch:
                    x = x - delta
                    if clamp01:
                        x = x.clamp(0.0, 1.0)
                    continue

                # line search (same as stationary, just with this step's G)
                step_size = 1.0
                accepted = False
                for _ in range(ls_max):
                    cand = x - step_size * delta
                    if clamp01:
                        cand = cand.clamp(0.0, 1.0)

                    Rc = self.Rf(cand, beta, gamma).to(device=device, dtype=dtype)
                    Gc = B @ cand - dt_t * alpha * Rc - b

                    # Armijo (sufficient decrease) condition:
                    # 1e-4 small constant = how much decrease is enough to accept Newton step
                    if torch.linalg.norm(Gc) < (1.0 - 1e-4 * step_size) * Gnorm:
                        x = cand
                        accepted = True
                        break

                    step_size *= 0.5

                if not accepted:
                    x = x - 1e-3 * delta
                    if clamp01:
                        x = x.clamp(0.0, 1.0)

            # accept timestep
            f = x
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
