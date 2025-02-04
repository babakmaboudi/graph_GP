import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import scipy as scp
import matplotlib.animation as animation


# class for creating a Laplacian operator on a circular graph
# The code is inspired by the work "Non-separable Spatio-temporal Graph Kernels via 
# SPDEs" (2022) by Nikitin et al.
class graph_circle_mesh():
    # the constructor function creates a graph consists of nodes on the circumference of 
    # a circle. The edges of the graph connects only the neighboring nodes wieghted by
    # the distance. Since the nodes are chosen uniformly, the weights become a constant
    # value.
    def __init__(self, N=128): # N is the number of nodes on the circumference of the circle
        self.N = N
        self.theta = np.linspace(0,2*np.pi,N,endpoint=False) # uniform placement of the nodes
        points = np.exp( 1j*self.theta ) # defining nodes on the unit circle in the complex plane
        
        # the following line tranform complex plane to real plane
        self.x = np.real(points)
        self.y = np.imag(points)

        # initiating an empty adjacecy matrix
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
        Dw = np.diag( np.sum( self.W, axis = 1 ) ) # diagonal matrix of accumulated sums
        self.L = (Dw - self.W)/dx/dx # The graph Laplacian operator (differs from the paper by the nomalaization constant 1/dx/dx)

        #self.tau = 1/(0.5*2*np.pi)/(0.5*2*np.pi)
        self.tau = 0.1 # The length scale: the distantce to which nodes are correlated. In the paper tau = 2nu/kappa^2

    # sampling the stationary Gaussian process by solving (K^nu) v = w, at this point nu = 1 or 2
    def sample_stationary(self, w, nu=2):
        return np.linalg.solve(np.linalg.matrix_power(self.tau*np.eye(self.N)+self.L, nu),w)

    # sampling the non-stationary Gaussian process with pure white noise
    def sample_heat(self, nu=2):
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


    # routine for plotting a sample in a plt axis handler
    def plot_sample(self, u, ax):
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


    # routine to plot graph samples as normal displacements (currently has a bug)
    def plot_sample_displacemnet(self, u, ax):
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

# sampling and plotting the stationary Gaussian process for nu = 1 and nu = 2
def plot_stationary():
    N = 256 # discretization size
    graph = graph_circle_mesh(N=N) # graph class holding the graph Laplacian operator
    f1,axes1 = plt.subplots(2,4) # figures for 8 samples
    #f2,axes2 = plt.subplots(2,4)

    nus = [1,2] # smoothness levels
    for i in range(axes1.shape[0]): # loop over the nu
        for j in range(axes1.shape[1]): # loop over number of samples
            w = np.random.standard_normal(N) # random input
            v = graph.sample_stationary(w, nus[i]) # sample
            graph.plot_sample(v,axes1[i,j]) # plotting the sample
            axes1[i,j].set_xticklabels([])
            axes1[i,j].set_yticklabels([])
            axes1[i,j].set_title(r'$\nu = {}$'.format(nus[i]))

            #graph.plot_sample_displacemnet(v,axes2[i,j])
            #axes2[i,j].set_xticklabels([])
            #axes2[i,j].set_yticklabels([])
            #axes2[i,j].set_title(r'$\nu = {}$'.format(nus[i]))

    plt.subplots_adjust(wspace=0) # removing space between plots
    plt.show()

# sampling and plotting a non-stationary Gaussian process on the graph
# plotting on the real line
def plot_heat():
    N = 256 # discretization size
    graph = graph_circle_mesh(N=N) # graph class holding the graph Laplacian operator 
    sol_vec = graph.sample_heat(nu=1) # sampling the Gaussian process

    # The rest of this code plots the sample of the Gaussian process on the real line
    f,ax = plt.subplots(1) # initiating the figure
    
    line = ax.plot( graph.theta, sol_vec[0] )[0] # plotting the initial lines
    ax.set( xlim=[0,2*np.pi], ylim=[-3,3] ) 
    
    # This class updates the frame content according to the sample of the Gaussian process
    # and pass the content to plt.animation
    class heat_animation:
        def __init__(self, sol):
            self.sol = sol
        def update(self, frame):
            line.set_ydata(self.sol[frame]) # this updates the frame content
            return line
    
    heat_anim = heat_animation(sol_vec) # instance of the frame updator class
    ani = animation.FuncAnimation(fig=f, func=heat_anim.update, frames=sol_vec.shape[0], interval=30) # plt animation
    plt.show()

# sampling and plotting a non-stationary Gaussian process on the graph
# plotting on the circular graph
def plot_heat_graph():
    N = 256 # discretization size
    graph = graph_circle_mesh(N=N) # graph class holding the graph Laplacian operator 
    sol_vec = graph.sample_heat(nu=1) # sampling the Gaussian process

    # The rest of this code plots the sample of the Gaussian process on the graph
    f,ax = plt.subplots(1)

    nbin = 64
    cmap = matplotlib.colormaps['plasma']
    colors = cmap(np.linspace(0, 1, nbin))

    dist = np.max(sol_vec) - np.min(sol_vec)
    plot_items = []
    for i in range(graph.x.shape[0]-1):
        color_id = int( (sol_vec[0,i]-np.min(sol_vec) )/dist * (nbin-1) )
        x = graph.x[i:i+2]
        y = graph.y[i:i+2]
        line = ax.plot( x, y , color=colors[color_id], linewidth=10 )
        plot_items.append( line )
    ax.set_aspect('equal')

    class heat_animation:
        def __init__(self, sol):
            self.sol = sol
        def update(self, frame):
            for i in range(graph.x.shape[0]-1):
                color_id = int( (self.sol[frame,i]-np.min(sol_vec) )/dist * (nbin-1) )
                plot_items[i][0]._color = colors[color_id]

            #return plot_items

    heat_anim = heat_animation(sol_vec)
    ani = animation.FuncAnimation(fig=f, func=heat_anim.update, frames=sol_vec.shape[0], interval=100)
    plt.show()





if __name__ == '__main__':
    # uncomment to visualize a stationary sample
    #plot_stationary()

    # uncomment to visualize a non-stationary Gaussian process visualized on the real line
    #plot_heat()

    # uncomment to visualize a non-stationary Gaussian process visualized on the graph
    plot_heat_graph()
