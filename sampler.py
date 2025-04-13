import numpy as np
import progressbar
import pickle

class Gibbs():
    def __init__(self, inits, target, scales):
        self.target = target
        self.num_components = len(inits)

        self.samplers = []

        t0 = lambda x: self.target(x, inits[1])
        self.samplers.append( pCN( inits[0], t0, scales[0] ) )

        t1 = lambda x: self.target(inits[0], x)
        self.samplers.append( MH( inits[1], t1, scales[1] ) )

    def warm_up(self, N_outer=1, N_inner=1):
        if(N_inner > 10):
            skip_len = int(N_inner/10)
        else:
            skip_len = 1
        for i in progressbar.ProgressBar()( range(N_outer) ):
            t0 = lambda x: self.target(x, self.samplers[1].current_sample)
            self.samplers[0].set_target( t0 )
            self.samplers[0].warm_up(N_inner, skip_len=skip_len)
            self.samplers[0].clear()

            t1 = lambda x: self.target(self.samplers[0].current_sample, x)
            self.samplers[1].set_target( t1 )
            self.samplers[1].warm_up(N_inner, skip_len=skip_len)
            self.samplers[1].clear()

    def sample(self, N_outer=1, N_inner=1, checkpoint=False, batch_size=None):
        if(checkpoint):
            self.batch_size = batch_size
            self.batch_idx = 0

        for i in progressbar.ProgressBar()( range(N_outer) ):
            t0 = lambda x: self.target(x, self.samplers[1].current_sample)
            self.samplers[0].set_target( t0 )
            self.samplers[0].multi_step(N_inner)

            t1 = lambda x: self.target(self.samplers[0].current_sample, x)
            self.samplers[1].set_target( t1 )
            self.samplers[1].multi_step(N_inner)

            if( checkpoint and ((i+1)%self.batch_size==0) ):
                self.save_checkpoint( path='./__samples_cache/checkpoint_{}.npz'.format(self.batch_idx) )
                self.dump()
                self.batch_idx += 1

    def get_samples(self):
        samples = []
        for i in range(self.num_components):
            samples.append( self.samplers[i].get_samples() )

        return samples

    def save_checkpoint(self, path='checkpoint.pickle'):
        states = []
        for i in range(self.num_components):
            states.append( self.samplers[i].get_state() )

        with open(path, 'wb') as handle:
            pickle.dump(states, handle, protocol=pickle.HIGHEST_PROTOCOL)

    def load_checkpoint(self, path):
        with open(path, 'rb') as handle:
            states = pickle.load(handle)

        for i in range(self.num_components):
            self.samplers[i].load_state( states[i] )

    def clear(self):
        for i in range(self.num_components):
            self.samplers[i].clear()

    def dump(self, path='./__samples_cache/'):
        if( path.endswith('/') ):
            pass
        else:
            path += '/'

        batch = []
        for i in range(self.num_components):
            batch.append( self.samplers[i].get_batch(self.batch_size) )

        np.savez(path+'_{}.npz'.format(self.batch_idx), batch=np.array(batch) )

class sampler():
    """
    The class is a parent sampler class that is used to handle samples and log densities
    """
    def __init__(self, x0, target, scale=1):
        """
        the initializer sets the starting parameters for the sampler
        """
        self.dim = len(x0) # dimension of the target distribution
        self.target = target # log of the target density function

        self.current_sample = x0 # current sample in the sampling algorithm
        self.current_target = self.target( x0 ) # evaluated log of the density function

        self.samples = [ x0 ] # list containing the samples
        self.acc = [1] # a boolean list containing the history of acceptance and rejected samples

        self.scale = scale # the scale of the proposal step

    def sample(self, N, batch_size=None):
        """
        the sampling loop

        batch_size: the number of samples before checkpointing
        """
        if(batch_size):
            batch = 0

        # the main sampling loop
        for i in progressbar.ProgressBar()( range(N) ):
            self.step() # perfomring a signle acceptance/rejection step

            # checkpointing step
            if(batch_size):
                if((i+1)%batch_size == 1):
                    self.save_checkpoint('./checkpoints/check_{}.npz'.format(batch))
                    np.savez('__sampler_cache/samples_{}.npz'.format(batch), self.samples[-1-batch_size])
                    batch += 1

    def warm_up(self, N, skip_len=None):
        """
        the warm up loop to adjust the step-size of the sampling method

        skip_len: Number of samples before adjusting the step size
        """
        if(skip_len == None):
            self.skip_len = int( N/10 )
        else:
            self.skip_len = skip_len

        update_count = 0
        # main warm up loop
        for i in progressbar.ProgressBar()( range(N) ):
            self.step()
            if(  (i+1)%self.skip_len == 0 ):
                self.tune(update_count)
                update_count += 1

    def step(self):
        """
        Performing one step of the sampling method
        """
        acc = self.update()

        self.acc.append(acc)
        self.samples.append(self.current_sample)

    def multi_step(self, N):
        """
        Performing multiple steps of the sampling method

        N: number of steps
        """
        acc = 0
        for i in range(N):
            self.update()

        self.samples.append(self.current_sample)

    def tune(self, update_count):
        """
        Performing one step of the warm-up method

        N: number of steps
        update_count: frequency of the step size adjustment
        """
        hat_acc = np.mean(self.acc[-1-self.skip_len:])

        # d. compute new scaling parameter
        zeta = 1/np.sqrt(update_count+1)   # ensures that the variation of lambda(i) vanishes
        scale_temp = np.exp(np.log(self.scale) + zeta*(hat_acc-0.234))

        # update parameters
        self.scale = min(scale_temp, 1)

    def set_target(self, target):
        """
        sets the log density of the target density
        """
        self.target = target

    def get_samples(self):
        """
        returns the samples after sampling method
        """
        return np.array(self.samples)

    def get_batch(self, batch_size):
        """
        returns the last batch of samples
        """
        return np.array( self.samples[-batch_size:] )

    def load_checkpoint(self, path):
        """
        resumes the state of the sampler from a path to a checkpoint file
        """
        self.clear()
        checkpoint = np.load(path)
        self.current_sample = checkpoint['current_sample']
        self.current_target = checkpoint['current_target']
        self.scale = checkpoint['scale']

    def load_state(self, state):
        """
        resumes the state of the sampler from a state variable
        """
        self.clear()
        self.current_sample = state['current_sample']
        self.current_target = state['current_target']
        self.scale = state['scale']

    def clear(self):
        """
        deletes the samples in the sampler
        """
        self.samples.clear()
        self.acc.clear()

class pCN(sampler):
    def __init__(self, x0, target, scale=1):
        super().__init__(x0, target, scale=1)

    def update(self):
        xi = np.random.standard_normal(self.dim)
        x_star = np.sqrt( 1 - self.scale**2 ) * self.current_sample + self.scale*xi

        target_eval_star = self.target(x_star)

        ratio = target_eval_star - self.current_target
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        acc = 0
        if (u_theta <= alpha):
            self.current_sample = x_star
            self.current_target = target_eval_star
            acc = 1

        return acc

    def save_checkpoint(self, path='checkpoint.npz'):
        np.savez(path, type='pCN', current_sample=self.current_sample, current_target=self.current_target, scale=self.scale)

    def get_state(self):
        return {'type': 'pCN', 'current_sample': self.current_sample, 'current_target': self.current_target, 'scale': self.scale}

class MH(sampler):
    """
    The class that performs Metropolis-Hastings (MH) algorithm to sample from a (posterior) distribution
    """
    def __init__(self, x0, target, scale=1):
        """
        This function calls the parent method using the following input:

        x0 : the initial guess for th Metropolis-Hastings alogirthm
        target: a function that computes the log of the target's probability density (posterior)
        scale (optional): This is the proposal scale for the Metropolis-Hastings method
        """
        super().__init__(x0, target, scale=1)

    def update(self):
        """
        This function performs 1 step of the Metropolis-Hastings algorithm
        """
        # proposing a point x_i using the scale
        xi = np.random.standard_normal(self.dim)
        x_star = self.current_sample + self.scale*xi

        # evaluating the log of the target density for the proposal
        target_eval_star = self.target(x_star)

        # computing the log of the acceptance ratio
        ratio = target_eval_star - self.current_target
        alpha = min(0, ratio)

        # accept/reject
        u_theta = np.log(np.random.rand())
        acc = 0
        if (u_theta <= alpha):
            self.current_sample = x_star
            self.current_target = target_eval_star
            acc = 1

        return acc

    def save_checkpoint(self, path='checkpoint.npz'):
        """
        This function saves checkpoints in order to resume sampling
        """
        np.savez(path, type='MH', current_sample=self.current_sample, current_target=self.current_target, scale=self.scale)

    def get_state(self):
        """
        This function returns the state of the sampler
        """
        return {'type': 'MH', 'current_sample': self.current_sample, 'current_target': self.current_target, 'scale': self.scale}
