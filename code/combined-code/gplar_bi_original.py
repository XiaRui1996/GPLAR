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
from layers_bi import SVGPLayer
from utilities import BroadcastingLikelihood
from gpflow.utilities import set_trainable
from gpflow.models.util import inducingpoint_wrapper
from gpar.regression import GPARRegressor
from scipy.cluster.vq import kmeans2

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)

class GPLARBase(BayesianModel):
    """Base class for GPLAR."""

    def __init__(self, likelihoods, layers, backwards_layers,
                 num_samples=10, num_data=None, 
                 minibatch_size=None, missing=False,
            **kwargs):
        super().__init__(**kwargs)

        self.likelihoods = likelihoods
        self.layers = layers
        self.backwards_layers = backwards_layers
        self.num_samples = num_samples
        self.num_data = num_data
        self.minibatch_size = minibatch_size
        self.missing = missing
        

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
        BHs, BHmeans, BHvars = [], [], []
        H, BH = sX, sX
        zs = zs or [None, ] * len(self.layers) # [None, None, ..., None]
        for layer,backlayer, z in zip(self.layers,self.backwards_layers, zs):
            Hy, Hmean, Hvar = layer.sample_from_conditional(H, z=z,
                    full_cov=full_cov)
            BHy, BHmean, BHvar = backlayer.sample_from_conditional(BH, z=z,
                    full_cov=full_cov)

            Hs.append(Hy)
            Hmeans.append(Hmean)
            Hvars.append(Hvar)
            
            BHs = [BH,] + BHs
            BHmeans = [BHmean,] + BHmeans
            BHvars = [BHvar,] + BHvars
            
            H = tf.concat([H,Hy], axis=-1)
            BH = tf.concat([BH, BHy], axis=-1)

        return Hs, Hmeans, Hvars, BHs, BHmeans, BHvars

    def _predict(self, X, full_cov=False, S=1):
        Hs, Hmeans, Hvars, BHs, BHmeans, BHvars = self.propagate(X, full_cov=full_cov, S=S)
        return Hmeans, Hvars, BHmeans, BHvars

    def E_log_p_Y(self, X, Y, full_cov=False):
        """Computes Monte Carlo estimate of the expected log density of the
        data, given a Gaussian distribution for the function values.
        if 
            q(f) = N(Hmu, Hvar)
            
        this method approximates
            \int (\log p(y|f)) q(f) df"""
        num_output = Y.shape[1]
        Hmean, Hvar, BHmean, BHvar = self._predict(X, full_cov=full_cov, S=self.num_samples)
        if full_cov:
            for i in range(num_output):
                Hvar[i] = tf.linalg.trace(Hvar[i])[:,:,None] #[S,N,1]
               
        result = tf.cast(0., dtype=np.float64)
        for i in range(num_output):
            if self.missing:
                available = ~tf.math.is_nan(Y[:,i])
                y = tf.where(available, Y[:,i], 0.)
            else:
                y = Y[:,i]
                
            var_exp = self.likelihoods[i].variational_expectations((Hmean[i]+BHmean[i])/2., #[S,N,1]
                                                (Hvar[i]+BHvar[i])/4, #[S,N,1]
                                                y[:,None]) #[N,1]
            if self.missing:
                mask = tf.cast(tf.where(available,1.,0.),dtype=var_exp.dtype)
                var_exp = var_exp * tf.tile(mask[None,:],[self.num_samples,1])
            result += tf.reduce_mean(var_exp,0)
        return result

    def prior_kl(self):
        return tf.reduce_sum([layer.KL() for layer in self.layers]) \
             + tf.reduce_sum([layer.KL() for layer in self.backwards_layers])

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
        Hmean, Hvar, BHmean, BHvar = self._predict(Xnew, full_cov=full_cov, S=num_samples)
        ms, vs = [],[]
        for i in range(len(self.likelihoods)):
            mean, var = self.likelihoods[i].predict_mean_and_var((Hmean[i]+BHmean[i])/2.,
                                                (Hvar[i]+BHvar[i])/4,)
        ms.append(mean)
        vs.append(var)
        return np.stack(ms), np.stack(vs)
    
    #def predict_log_density(self, Xnew, Ynew, num_samples, full_cov=False):
        #Hmean, Hvar = self._predict(Xnew, full_cov=full_cov, S=num_samples)
        #l = self.likelihood.predict_density(Fmean, Fvar, Ynew)
        #log_num_samples = tf.math.log(tf.cast(num_samples, l.dtype))
        #return tf.reduce_logsumexp(l - log_num_samples, axis=0)

class GPLAR(GPLARBase):
    """The GPLAR model with zero mean function at each layer"""
    
    def __init__(self, X, Y, Z, q_sqrt_initial, Z_back, q_sqrt_initial_back, kernels, back_kernels, likelihoods,
                 mean_function=Zero(), white=False, **kwargs):
        
        layers = self._init_layers(X, Y, Z, q_sqrt_initial, kernels, 
                        mean_function=mean_function, white=white)
        backwards_layers = self._init_layers(X, Y, Z_back, q_sqrt_initial_back, back_kernels,
                                    mean_function=mean_function, white=white)

        super().__init__(likelihoods, layers, backwards_layers, **kwargs)
        
    def _init_layers(self, X, Y, Z, q_sqrt_initial, kernels,
                     mean_function=Zero(), Layer=SVGPLayer, white=False):
        """
        The first layer only models between input and output_1,
        The second layer models between input and output_2, output_1 and output_2,
        The inducing point for each layer for input dimension should be shared?
        The induing point for output dimension should be calculated instead of changing?"""
        
        layers = []
        num_inputs = X.shape[1]
        num_outputs = Y.shape[1]
        self.inducing_inputs = inducingpoint_wrapper(Z[:, :num_inputs])
        inducing_inputs = self.inducing_inputs.Z
        
        for i in range(num_outputs):
            layer = Layer(kernels[i], inducing_inputs, Z[:,num_inputs+i], q_sqrt_initial[:,i], mean_function, white=white)
            #layer = Layer(kernels[i], Z[:,:num_inputs+i], Z[:,num_inputs+i], q_sqrt_initial[:,i], mean_function, white=white)
            layers.append(layer)
            inducing_inputs = tf.concat([inducing_inputs,layer.q_mu], axis=1)
            
        return layers
    

class GPLARegressor(GPLAR):
    def __init__(self, X, Y, M, gpar, gpar_back, #missing_data, begin, end, training_columns,
                 likelihoods = None,
                 reorder = None, minibatch_size=None, missing=False,
                 mean_function=Zero(), white=False,
                 impute=True, 
                 input_nonlinear=True, scale=1.0,scale_tie=False,
                 per=False, per_period=1.0, per_scale=1.0, per_decay=10.0,
                 input_linear=False, input_linear_scale=100.0,
                 linear=True, linear_scale=100.0,
                 nonlinear=True, nonlinear_scale=0.1, nonlinear_dependent=False,nonlinear_additive=False,
                 rq=False,
                 markov=None,
                 noise_inner=1e-05, noise_obs=0.01,
                 normalise_y=True, transform_y=(lambda x:x, lambda x:x),**kwargs):
    
        self.impute = impute
        self.model_config = {
            'input_nonlinear':input_nonlinear, 'scale': scale, 'scale_tie': scale_tie,
            'per': per, 'per_period': per_period, 'per_scale': per_scale, 'per_decay': per_decay,
            'input_linear': input_linear, 'input_linear_scale': input_linear_scale,
            'linear': linear, 'linear_scale': linear_scale, 
            'nonlinear': nonlinear, 'nonlinear_scale': nonlinear_scale, 'nonlinear_dependent': nonlinear_dependent,
            'nonlinear_additive':nonlinear_additive,
            'rq': rq,
            'markov': markov,
            'noise_inner': noise_inner}
        self.m = X.shape[1]
        self.num_outputs = Y.shape[1]
        self.reorder = reorder
        kernels = self._kernels_generator()
        back_kernels = self._kernels_generator()
        if not likelihoods:
            likelihoods = []
            for i in range(self.num_outputs):
                likelihoods.append(Gaussian(variance=noise_obs))
                
        Z, q_sqrt_initial = self._initialize_inducing_locations_from_post_GPAR(gpar,X,Y,M)  
        Y_back = Y[:,::-1].copy()
        Z_back, q_sqrt_initial_back = self._initialize_inducing_locations_from_post_GPAR(gpar_back,X,Y_back,M)
        
        self.initial_inducing_points = Z
        if np.any(np.isnan(Y)): missing = True
        super().__init__(X,Y,Z, q_sqrt_initial, Z_back, q_sqrt_initial_back, 
                         kernels, back_kernels, likelihoods,
                         mean_function=mean_function,white=white,
                         num_data=X.shape[0],
                         minibatch_size=minibatch_size,
                         missing = missing, **kwargs)
        
    # choose datapoint that are closed downwards 
    def _initialize_inducing_locations(self, X, Y, M): #M is number of inducing points per layer
        N, inducing_points = X.shape[0], []
        notnan, idx = np.array([True]*N), np.array(list(range(N)))
        for i in range(self.num_outputs):
            notnan = np.logical_and(notnan, ~np.isnan(Y[:,i]))
            r = np.random.choice(idx[notnan],M[i],replace=False)
            inducing_points.append(np.concatenate((X[r,:],Y[r,:i+1]),axis=1))
        return inducing_points
    
    def _initialize_inducing_locations_from_post_GPAR(self, gpar, X, Y, M):
                            #, missing_data, begin, end, training_columns):
        gpar.fit(X,Y)
        Z = np.linspace(np.min(X),np.max(X),M).reshape(M,1)
        samples = gpar.sample(Z, num_samples=50, latent=True, posterior=True)
        
        means, std = np.mean(samples,axis=0), np.std(samples,axis=0)
            
        inducing_points = np.concatenate((Z, means), axis=1)
        return inducing_points, std
    
        
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
            # Construct inner-layers noise kernel
            kernel = White(variance=self.model_config['noise_inner'])
            # Initialize a non-linear kernels over inputs
            #if pi==0:
            if self.model_config['input_nonlinear']:
                scales = [self.model_config['scale']]*self.m if self.model_config['scale_tie'] else self.model_config['scale']
                if self.model_config['rq']:
                    kernel += RationalQuadratic(active_dims=m_inds, 
                                                 variance=1.0,
                                                 lengthscales=scales,
                                                 alpha=1e-2)
                else:
                    kernel += SquaredExponential(active_dims=m_inds, 
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
                if self.model_config['nonlinear_additive']:
                    if self.model_config['rq']:
                        for i in range(pi):
                            kernel += RationalQuadratic(active_dims=[self.m+i], 
                                                 variance=1.0,
                                                 lengthscales=self.model_config['nonlinear_scale'],
                                                 alpha=1e-2)
                    else:
                        for i in range(pi):
                            kernel += SquaredExponential(active_dims=[self.m+i],
                                                variance=1.0,
                                                lengthscales=self.model_config['nonlinear_scale'])
                else:  
                    if self.model_config['nonlinear_dependent']: 
                        active_dims = m_inds.extend(p_inds)
                        scales = [self.model_config['scale']]*self.m
                        scales.extend([self.model_config['nonlinear_scale']]*p_num)
                    else: 
                        active_dims = p_inds
                        scales = [self.model_config['nonlinear_scale']]*p_num
                    if self.model_config['rq']:
                        kernel += RationalQuadratic(active_dims=active_dims, 
                                                 variance=1.0,
                                                 lengthscales=scales,
                                                 alpha=1e-2)
                    else:
                        kernel += SquaredExponential(active_dims=active_dims, 
                                                 variance=1.0,
                                                 lengthscales=scales)
            
            kernels.append(kernel)
            
        return kernels