import torch
import matplotlib.pyplot as plt
import math

class heat_PDE():
    # Fininite difference class for the heat eqution with left boundary homogeneous Dirichlet and right boundary with homogeneous Neumann
    def __init__(self, N, dx=1, dtype=torch.float64):
        self.device = torch.device('cpu')
        self.dtype = dtype
        self.N = N
        self.dx = dx
        
        inv_dx2 = 1.0 / (self.dx * self.dx)
        diag_main = (-2.0 * torch.ones(self.N, device=self.device, dtype=self.dtype)) * inv_dx2
        diag_off = ( 1.0 * torch.ones(self.N - 1, device=self.device, dtype=self.dtype)) * inv_dx2

        self.L = torch.diag(diag_main) + torch.diag(diag_off, 1) + torch.diag(diag_off, -1)

        self.L[0, :].zero_()
        self.L[0, 0] = -2.0 * inv_dx2
        self.L[0, 1] =  2.0 * inv_dx2

        self.L[-1, :].zero_()
        self.L[-1, -2] =  2.0 * inv_dx2
        self.L[-1, -1] = -2.0 * inv_dx2
        self.L.to(dtype)

    def weighted_laplacian_1d_NN_arith(self, rho, dx=1.0):
        w = 0.5 * (rho[:-1] + rho[1:]) / (dx * dx)  # w[i] = rho_{i+1/2}/dx^2

        diag = torch.zeros_like(rho)
        diag[0] = w[0]
        diag[1:-1] = w[:-1] + w[1:]
        diag[-1] = w[-1]

        self.L = torch.diag(diag) + torch.diag(-w, 1) + torch.diag(-w, -1)
        print(self.L )

    def fd_laplacian_from_edge_weights_NN(self, w, dx=1):
        N = w.numel() + 1
        diag = torch.zeros(N, device=w.device, dtype=w.dtype)
        diag[0] = w[0]
        diag[1:-1] = w[:-1] + w[1:]
        diag[-1] = w[-1]
        self.L = torch.diag(diag) + torch.diag(-w, 1) + torch.diag(-w, -1)
        self.L = self.L/(dx*dx)

    def sample_stationary(self, w, nu=2):
        I = torch.eye( self.L.shape[0], dtype=self.L.dtype )
        temp = 1 * ( 1 * I + self.L ).to(self.L.dtype)
        K_nu_tensor = torch.linalg.matrix_power(temp, nu ).to(torch.float64)
        return torch.linalg.solve( K_nu_tensor, w )

    def sample_heat(self, v0, nu=2, T=5.0, dt=0.01, w_noise=None):
        I = torch.eye( self.L.shape[0], dtype=self.L.dtype )
        temp = 1. * ( 1. * I + self.L )
        K_nu_tensor = torch.linalg.matrix_power(temp, nu ).to(torch.float64)

        MAX_ITER = int(T / dt)
        sol = [ v0 ]

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

def create_signal_random():
    N = 5
    dx = 1#math.sqrt(2)
    dtype = torch.float64
    problem = heat_PDE(N, dx=dx)

    # parameters used in the prior
    nu_prior = 1 # regularity
    dt_prior = 0.01 # time-step size
    T_prior = 5. # maximum time for simulation
    MAX_ITER_prior = int(T_prior / dt_prior) # number of iterations to hit maximum time

    x = torch.linspace(0,1,N)
    v0 = torch.exp(-0.5*((x-0.5)/(0.5/8))**2).to(dtype)

    prior_map_scale = .1 # output variance parameter
    prior_map_mean = .0
    prior_map_text = 'lambda x:  prior_map_mean + prior_map_scale*prior_map_parameter*torch.exp(x)' # the copy of the mapping for future reference
    prior_map = lambda x:  prior_map_mean + prior_map_scale*torch.exp(x) # the actual mapping
    
    # create a signal
    #x_true = torch.randn(G.num_edges).to(dtype) # the unknown for the inverse problem
    torch.manual_seed(0)
    x_true = torch.randn(N-1).to(dtype)
    #x_true = torch.ones(N-1).to(dtype)
    w_noise_true = torch.randn(MAX_ITER_prior, v0.shape[0]).to(dtype)
    problem.fd_laplacian_from_edge_weights_NN( x_true, dx=dx) # updating the graph weights accroding to the unknown
    print(problem.L)
    y_true = problem.sample_heat(v0, nu=nu_prior, dt=dt_prior, T=T_prior, w_noise=w_noise_true ) # creating a noise free signal
    #print(y_true)


if __name__ == '__main__':
    create_signal_random()    
