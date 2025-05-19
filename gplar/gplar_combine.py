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
from layers_pascal import SVGPLayer
from gpflow.base import Parameter
from utilities import BroadcastingLikelihood
from gpflow.utilities import set_trainable
from gpflow.models.util import inducingpoint_wrapper
from scipy.cluster.vq import kmeans2
from gpflow.ci_utils import ci_niter

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)

class GPLARBase(BayesianModel):
    """Base class for GPLAR."""

    def __init__(self, 
                 num_samples=10, num_data=None, 
                 minibatch_size=None, missing=False,
            **kwargs):
        super().__init__(**kwargs)

        self.num_samples = num_samples
        self.num_data = num_data
        self.minibatch_size = minibatch_size
        self.missing = missing
        

    def propagate(self, X, full_cov=False, S=1, zs=None):
        
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1]) # [S,N,1]
        Hg, Hmeang, Hvarg = [], [], []
        Hmeanb, Hvarb = [],[]
        zs = zs or [None, ] * len(self.temporal_layers) # [None, None, ..., None]
        for temporal_layer,z in zip(self.temporal_layers, zs):
            Hmean, Hvar = [],[]
            
            Ht, Hmeant, Hvart = temporal_layer.sample_from_conditional(sX, z=z, full_cov=full_cov)
            Hg.append(Ht)
            Hmeang.append(Hmeant)
            Hvarg.append(Hvart)
        
        Hg = tf.matmul(tf.concat(Hg,axis=2), self.A)
            
        for boosting_layer, i, z in zip(self.boosting_layers, range(len(self.boosting_layers)),zs):
            Hc, Hmeanc, Hvarc = boosting_layer.sample_from_conditional(Hg[:,:,i:i+1], z=z, full_cov=full_cov)
            Hmeanb.append(Hmeanc)
            Hvarb.append(Hvarc)

        return tf.concat(Hmeang, axis=2), tf.concat(Hvarg, axis=2),\
             tf.concat(Hmeanb, axis=2), tf.concat(Hvarb, axis=2)


    def E_log_p_Y(self, X, Y, full_cov=False):
        num_output = Y.shape[1]
        Hmeang, Hvarg, Hmeanb, Hvarb = self.propagate(X, full_cov=full_cov, S=self.num_samples)
        Hmean, Hvar = Hmeang + Hvarb, Hvarg + Hvarb
        
        result = tf.cast(0., dtype=np.float64)
        for i in range(num_output):
            if self.missing:
                available = ~tf.math.is_nan(Y[:,i])
                y = tf.where(available, Y[:,i], 0.)
            else:
                y = Y[:,i]
                
            var_exp = self.likelihoods[i].variational_expectations(Hmean[:,:,i:i+1],Hvar[:,:,i:i+1],y[:,None])
            if self.missing:
                mask = tf.cast(tf.where(available,1.,0.),dtype=var_exp.dtype)
                var_exp = var_exp * tf.tile(mask[None,:],[self.num_samples,1])
            result += tf.reduce_mean(var_exp,0)
        return result

    def prior_kl(self):
        all_layers = []
        for layer in self.temporal_layers:
            all_layers.append(layer)
        for layer in self.boosting_layers:
            all_layers.append(layer)
        return tf.reduce_sum([layer.KL() for layer in all_layers])

    def maximum_log_likelihood_objective(self, X, Y, full_cov=False):
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

    def predict_f(self, Xnew, num_samples, full_cov=False):
        """Returns mean and variance of each output."""
        return self.propagate(Xnew, full_cov=full_cov, S=num_samples)
    
    def predict_y(self, Xnew, num_samples, full_cov=False):
        Hmean, Hvar = self._predict(Xnew, full_cov=full_cov, S=num_samples)
        ms, vs = [],[]
        for i in range(len(self.likelihoods)):
            mean, var = self.likelihoods[i].predict_mean_and_var(Hmean[i],Fvar[i])
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
    
    def __init__(self, X, Y, M,
                 mean_function=Zero(), white=False, Layer=SVGPLayer, **kwargs):
        
        self.temporal_layers = []
        
        for i in range(self.num_outputs):
            kerneli = self.temporal_kernel()
            inducing_inputs = inducingpoint_wrapper(kmeans2(X,M,minit='points')[0])
            layer = Layer(kerneli, inducing_inputs.Z, mean_function, white=white)
            self.temporal_layers.append(layer)

        super().__init__(**kwargs)   
    
    def train_temporal(self, X, Y, iteration):
        
        def temporal_elbo(X, Y, full_cov=False):
            var_exp, kl_priors = [],[]
            for i, layer, likelihood in zip(list(range(self.num_outputs)), self.temporal_layers, self.likelihoods):
                meani, vari = layer.conditional(X, full_cov=full_cov)
                available = ~tf.math.is_nan(Y[:,i])
                y = tf.where(available, Y[:,i], 0.)
                var_expi = likelihood.variational_expectations(meani, vari[:,None], y[:,None])
                var_expi = var_expi * tf.cast(tf.where(available,1.,0.),dtype=var_expi.dtype)
                var_exp.append(tf.reduce_sum(var_expi))
                kl_priors.append(layer.KL())
            L, KL = tf.reduce_sum(var_exp), tf.reduce_sum(kl_priors)
            if self.minibatch_size is not None:
                num_data = tf.cast(self.num_data, KL.dtype)
                minibatch_size = tf.cast(self.minibatch_size, KL.dtype)
                scale = num_data / minibatch_size
            else:
                scale = tf.cast(1.0, KL.dtype)
            return L * scale - KL
        
        @tf.function(autograph=False)
        def optimization_step(optimizer, data):
            with tf.GradientTape(watch_accessed_variables=False) as tape:
                tape.watch(self.trainable_variables)
                objective = -temporal_elbo(*data)
                grads = tape.gradient(objective, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return objective

        def run_adam(data, iterations):
            adam = tf.optimizers.Adam(0.001)
            for step in range(iterations):
                neg_elbo= optimization_step(adam, data)
                elbo = -neg_elbo
                if step%1000 == 0:
                    print(elbo.numpy())
        
        print("Start initial temporal training.")
        maxiter = ci_niter(iteration)
        run_adam((X, Y), maxiter)
        print("Done initial temporal training.")
              
    def boosting_initialization(self, Y, M, mean_function=Zero(), Layer=SVGPLayer, white=False):
        self.boosting_layers = []
        inducing_inputs = []
        #for i in range(self.num_outputs):
            #y_i = Y[~np.isnan(Y[:,i]),i]
            #inducing_inputs.append(np.linspace(np.min(y_i), np.max(y_i), M))
        #inducing_inputs = np.stack(inducing_inputs).T
        for layer in self.temporal_layers:
            inducing_inputs.append(layer.q_mu.value())
        inducing_inputs = tf.concat(inducing_inputs,axis=1)
        
        A = np.random.normal(0.,1.,size=(self.num_outputs, self.num_outputs))
        self.A = Parameter(A, dtype=default_float())
        
        inducing_inputs = tf.matmul(inducing_inputs, self.A)
        
        for i in range(self.num_outputs):
            kerneli = self.boosting_kernel()
            inducing_input = inducingpoint_wrapper(inducing_inputs[:,i:i+1])
            layer = Layer(kerneli, inducing_input.Z, mean_function, white=white)
            self.boosting_layers.append(layer)
        

class GPLARmodel(GPLAR):
    def __init__(self, X, Y, M, 
                 minibatch_size=None, missing=False,
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
        self.likelihoods = []
        for i in range(self.num_outputs):
            self.likelihoods.append(Gaussian(variance=noise_obs))
            
        if np.any(np.isnan(Y)): missing = True
        super().__init__(X, Y, M,
                       mean_function=mean_function,white=white,
                       num_data=X.shape[0], minibatch_size=minibatch_size,
                       missing = missing, **kwargs)
        
    def temporal_kernel(self):
        kernel = White(variance=self.model_config['noise_inner'])
        m_inds = list(range(self.m))
        # Initialize a non-linear kernels over inputs
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
        return kernel
    
    def boosting_kernel(self):
        kernels = []
        if self.model_config['nonlinear']:
            if self.model_config['rq']:
                kernels.append(RationalQuadratic(variance=1.0,
                                      lengthscales=self.model_config['nonlinear_scale'],
                                      alpha=1e-2))
            else:
                kernels.append(SquaredExponential(variance=1.0,
                                      lengthscales=self.model_config['nonlinear_scale']))
        
        if self.model_config['linear']:
            kernels.append(LinearKernel(variance=self.model_config['linear_scale']))
        
        kernel = kernels[0]
        for kerneli in kernels[1:]:
            kernel += kerneli
        return kernel