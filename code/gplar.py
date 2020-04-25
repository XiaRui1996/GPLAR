import pdb
import numpy as np
import tensorflow as tf
import gpflow
from gpflow.models import BayesianModel
from gpflow.mean_functions import Linear, Identity, Zero
from gpflow.kernels import SquaredExponential, RationalQuadratic, Periodic, White
from gpflow.kernels import Linear as LinearKernel
from gpflow.likelihoods import Gaussian, MultiClass
from gpflow.config import default_float, default_jitter
from layers import SVGPLayer
from utilities import BroadcastingLikelihood
from gpflow.utilities import set_trainable
from gpflow.models.util import inducingpoint_wrapper

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)

class GPLARBase(BayesianModel):
    """Base class for GPLAR."""

    def __init__(self, likelihoods, layers, 
                 num_samples=10, num_data=None, 
                 minibatch_size=None,
            **kwargs):
        super().__init__(**kwargs)

        self.likelihoods = likelihoods
        self.layers = layers
        self.num_samples = num_samples
        self.num_data = num_data
        self.minibatch_size = minibatch_size
        

    def propagate(self, X, full_cov=False, S=1, zs=None):
        """Propagate input X through layers of the GPLAR S times. 

        :X: A tensor, the input to the GPLAR.
        :full_cov: A bool, indicates whether or not to use the full
        covariance matrix.
        :S: An int, the number of samples to draw.
        :zs: A tensor, samples from N(0,1) to use in the reparameterisation
        trick.
        
        Every time before propagate need to update inducing points value
        using updated value of q_mu for every layer"""
        
        #for layer, next_layer in zip(self.layers[:-1],self.layers[1:]):
                #next_layer.update_inducing_points(layer.q_mu)
        
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1]) # [S,N,D]
        Hs, Hmeans, Hvars = [], [], []
        H = sX
        zs = zs or [None, ] * len(self.layers) # [None, None, ..., None]
        for layer, z in zip(self.layers, zs):
            Hy, Hmean, Hvar = layer.sample_from_conditional(H, z=z,
                    full_cov=full_cov)

            Hs.append(Hy)
            Hmeans.append(Hmean)
            Hvars.append(Hvar)
            
            H = tf.concat([H,Hy], axis=-1)

        return Hs, Hmeans, Hvars

    def _predict(self, X, full_cov=False, S=1):
        Hs, Hmeans, Hvars = self.propagate(X, full_cov=full_cov, S=S)
        return Hmeans, Hvars

    def E_log_p_Y(self, X, Y, full_cov=False):
        """Computes Monte Carlo estimate of the expected log density of the
        data, given a Gaussian distribution for the function values.

        if 

            q(f) = N(Hmu, Hvar)
            
        this method approximates

            \int (\log p(y|f)) q(f) df"""
        num_output = Y.shape[1]
        Hmean, Hvar = self._predict(X, full_cov=full_cov, S=self.num_samples)
        if full_cov:
            for i in range(num_output):
                Hvar[i] = tf.linalg.trace(Hvar[i])[:,:,None] #[S,N,1]
               
        result = tf.cast(0., dtype=np.float64)
        for i in range(num_output):
            var_exp = self.likelihoods[i].variational_expectations(Hmean[i], #[S,N,1]
                                                                   Hvar[i], #[S,N,1]
                                                                   tf.expand_dims(Y[:,i],-1)) #[N,1]
            result += tf.reduce_mean(var_exp,0)
        return result

    def prior_kl(self):
        return tf.reduce_sum([layer.KL() for layer in self.layers])

    def log_likelihood(self, X, Y, full_cov=False):
        """Gives a variational bound on the model likelihood."""
        L = tf.reduce_sum(self.E_log_p_Y(X, Y, full_cov))
        KL = self.prior_kl()
        if self.minibatch_size is not None:
            num_data = tf.cast(self.num_data, KL.dtype)
            minibatch_size = tf.cast(self.minibatch_size, KL.dtype)
            scale = num_data / minibatch_size
        else:
            scale = tf.cast(1.0, KL.dtype)

        return L * scale - KL

    def maximum_log_likelihood_objective(self, X, Y, full_cov=False):
        """ This returns the evidence lower bound (ELBO) of the log 
        marginal likelihood. """
        return self.log_likelihood(X, Y, full_cov=full_cov) 

    def predict_f(self, Xnew, num_samples, full_cov=False):
        """Returns mean and variance of each output."""
        return self._predict(Xnew, full_cov=full_cov, S=num_samples)
    
    def predict_y(self, Xnew, num_samples, full_cov=False):
        Hmean, Hvar = self._predict(Xnew, full_cov=full_cov, S=num_samples)
        f = lambda i: self.likelihoods[i].predict_mean_and_var(Hmean[i], Fvar[i])
        mean, var = tf.map_fn(f, list(range(len(self.layers))), dtype=(tf.float64, tf.float64))    
        return np.stack(mean), np.stack(var)
    
    #def predict_density(self, Xnew, Ynew, num_samples, full_cov=False):
        #Hmean, Hvar = self._predict(Xnew, full_cov=full_cov, S=num_samples)
        #l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        #log_num_samples = tf.math.log(tf.cast(num_samples, l.dtype))
        #return tf.reduce_logsumexp(l - log_num_samples, axis=0)

class GPLAR(GPLARBase):
    """The GPLAR model with zero mean function at each layer"""
    
    def __init__(self, X, Y, Z, kernels, likelihoods,
                 mean_function=Zero(), white=False, **kwargs):
        
        layers = self._init_layers(X, Y, Z, kernels, 
                        mean_function=mean_function, white=white)

        super().__init__(likelihoods, layers, **kwargs)
        
    def _init_layers(self, X, Y, Z, kernels,
                     mean_function=Zero(), Layer=SVGPLayer, white=False):
        """
        The first layer only models between input and output_1,
        The second layer models between input and output_2, output_1 and output_2,
        The inducing point for each layer for input dimension should be shared?
        The induing point for output dimension should be calculated instead of changing?"""
        
        layers = []
        num_inputs = X.shape[1]
        num_outputs = Y.shape[1]
        
        for i in range(num_outputs):
            layer = Layer(kernels[i], Z[:,:num_inputs+i], Z[:,num_inputs+i], mean_function, white=white)
            layers.append(layer)
            #Z = tf.concate([Z,layer.q_mu], axis=1)
            
        return layers

class GPLARegressor(GPLAR):
    def __init__(self, X, Y, M, minibatch_size=None,
                 mean_function=Zero(), white=False,
                 impute=True, 
                 scale=1.0,scale_tie=False,
                 per=False, per_period=1.0, per_scale=1.0, per_decay=10.0,
                 input_linear=False, input_linear_scale=100.0,
                 linear=True, linear_scale=100.0,
                 nonlinear=True, nonlinear_scale=0.1,
                 rq=False,
                 markov=None,
                 noise_inner=1e-05, noise_obs=0.01,
                 normalise_y=True, transform_y=(lambda x:x, lambda x:x),**kwargs):
    
        self.impute = impute
        self.model_config = {
            'scale': scale, 'scale_tie': scale_tie,
            'per': per, 'per_period': per_period, 'per_scale': per_scale, 'per_decay': per_decay,
            'input_linear': input_linear, 'input_linear_scale': input_linear_scale,
            'linear': linear, 'linear_scale': linear_scale, 
            'nonlinear': nonlinear, 'nonlinear_scale': nonlinear_scale,
            'rq': rq,
            'markov': markov,
            'noise_inner': noise_inner}
        self.m = X.shape[1]
        self.num_outputs = Y.shape[1]
        kernels = self._kernels_generator()
        likelihoods = [Gaussian(variance=noise_obs)]*self.num_outputs
        
        # Todo: normalize y
        # Todo: impute, handle missing data, make closed down
        # Todo: initialize inducing locations Z
        Z = self._initialize_inducing_locations(X,Y,M)
        super().__init__(X,Y,Z, kernels, likelihoods,
                         mean_function=mean_function,white=white,
                         num_data=X.shape[0],
                         minibatch_size=minibatch_size,**kwargs)
        
    def _initialize_inducing_locations(self, X, Y, M):
        r = np.random.choice(X.shape[0],M,replace=False)
        return np.concatenate((X[r,:],Y[r,:]),axis=1)
        
        
    def _kernels_generator(self):
        
        def _determine_indicies(m,pi,markov):
            # Build in the Markov structure: juggle with the indices of the outputs.
            p_last = pi - 1  # Index of last output that is given as input.
            p_start = 0 if markov is None else max(p_last - (markov - 1), 0)
            p_num = p_last - p_start + 1

            # Determine the indices corresponding to the outputs and inputs.
            m_inds = list(range(m))
            p_inds = list(range(m + p_start, m + p_last + 1))

            return m_inds, p_inds, p_num
        
        kernels = []
        for pi in range(self.num_outputs):
            m_inds, p_inds, p_num = _determine_indicies(self.m, pi, self.model_config['markov'])
            
            # Initialize a non-linear kernels over inputs
            scales = [self.model_config['scale']]*self.m if self.model_config['scale_tie'] else self.model_config['scale']
            if self.model_config['rq']:
                kernel = RationalQuadratic(active_dims=m_inds, 
                                                 variance=1.0,
                                                 lengthscales=scales,
                                                 alpha=1e-2)
            else:
                kernel = SquaredExponential(active_dims=m_inds, 
                                                 variance=1.0,
                                                 lengthscales=scales)
            # Add a periodic kernel over inputs
            # Decay?????
            if self.model_config['per']:
                scales = [self.model_config['per_scale']]*self.m
                periods = [self.model_config['per_period']]*self.m
                base_kernel = SquaredExponential(active_dims=m_inds, 
                                                 variance=1.0,
                                                 lengthscales=scales)
                kernel += Periodic(base_kernel, period=periods)
                
            # Add a linear kernel over inputs
            if self.model_config['input_linear']:
                variances = [self.model_config['input_linear_scale']]*self.m
                kernel += LinearKernel(active_dims=m_inds, 
                                              variance=variances)
            # Add a linear kernel over outputs
            if self.model_config['linear'] and pi>0:
                variances = [self.model_config['linear_scale']]*p_num
                kernel += LinearKernel(active_dims=p_inds, 
                                              variance=variances)
                
            # Add a non-linear kernel over outputs
            if self.model_config['nonlinear'] and pi>0:
                scales = [self.model_config['nonlinear_scale']]*p_num
                if self.model_config['rq']:
                    kernel += RationalQuadratic(active_dims=p_inds, 
                                                 variance=1.0,
                                                 lengthscales=scales,
                                                 alpha=1e-2)
                else:
                    kernel += SquaredExponential(active_dims=p_inds, 
                                                 variance=1.0,
                                                 lengthscales=scales)
                
            # Construct inner-layers noise kernel
            kernel += White(variance=self.model_config['noise_inner'])
            
            kernels.append(kernel)
            
        return kernels
            
            
            
            
            
