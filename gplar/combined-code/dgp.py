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
from dgp_layers import SVGPLayer
from utilities import BroadcastingLikelihood
from gpflow.utilities import set_trainable
from gpflow.models.util import inducingpoint_wrapper
from scipy.cluster.vq import kmeans2

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)

class DGPBase(BayesianModel):
    """Base class for GPLAR."""

    def __init__(self, likelihoods, layers, 
                 num_samples=10, num_data=None, 
                 minibatch_size=None, missing=False,
            **kwargs):
        super().__init__(**kwargs)

        self.likelihoods = likelihoods
        self.layers = layers
        self.num_samples = num_samples
        self.num_data = num_data
        self.minibatch_size = minibatch_size
        self.missing = missing
        

    def propagate(self, X, full_cov=False, S=1, zs=None):
        
        sX = tf.tile(tf.expand_dims(X, 0), [S, 1, 1]) # [S,N,D]
        Hs, Hmeans, Hvars = [], [], []
        H = sX
        zs = zs or [None, ] * len(self.layers) # [None, None, ..., None]
        for layer, z in zip(self.layers, zs):
            H, Hmean, Hvar = layer.sample_from_conditional(H, z=z,
                    full_cov=full_cov)

            Hs.append(H)
            Hmeans.append(Hmean)
            Hvars.append(Hvar)

        return Hs, Hmeans, Hvars

    def _predict(self, X, full_cov=False, S=1):
        Hs, Hmeans, Hvars = self.propagate(X, full_cov=full_cov, S=S)
        return Hmeans[-1], Hvars[-1]

    def E_log_p_Y(self, X, Y, full_cov=False):
        """Computes Monte Carlo estimate of the expected log density of the
        data, given a Gaussian distribution for the function values.
        if 
            q(f) = N(Hmu, Hvar)
            
        this method approximates
            \int (\log p(y|f)) q(f) df"""
        num_output = Y.shape[1]
        Hmean, Hvar = self._predict(X, full_cov=full_cov, S=self.num_samples)
               
        result = tf.cast(0., dtype=np.float64)
        for i in range(num_output):
            if self.missing:
                available = ~tf.math.is_nan(Y[:,i])
                y = tf.where(available, Y[:,i], 0.)
            else:
                y = Y[:,i]
                
            var_exp = self.likelihoods[i].variational_expectations(Hmean[:,:,i][:,:,None], #[S,N,1]
                                                                   Hvar[:,:,i][:,:,None], #[S,N,1]
                                                                   y[:,None]) #[N,1]
            if self.missing:
                mask = tf.cast(tf.where(available,1.,0.),dtype=var_exp.dtype)
                var_exp = var_exp * tf.tile(mask[None,:],[self.num_samples,1])
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



class DGP(DGPBase):
    """The GPLAR model with zero mean function at each layer"""
    
    def __init__(self, X, Y, Z, dims, kernels, likelihoods, input_prop_dim=None, 
                 mean_function=Zero(), white=False, **kwargs):
        
        layers = self._init_layers(X, Y, Z, dims, kernels,
                        mean_function=mean_function, white=white)

        super().__init__(likelihoods, layers, **kwargs)
        
    def _init_layers(self, X, Y, Z, dims, kernels, num_outputs=None, 
                     mean_function=Zero(), Layer=SVGPLayer, white=False):
        num_outputs = num_outputs or Y.shape[1]
        D = X.shape[1]
        M = Z.shape[0]

        layers = []

        for dim_in, dim_out, kern in zip(dims[:-1], dims[1:], kernels):
            dim_out = dim_out - D
            std_in = kern.variance.read_value()**0.5
            pad = np.random.randn(M, dim_in - D) * 2. * std_in
            Z_padded = np.concatenate([Z, pad], 1)
            layers.append(Layer(kern, Z_padded, dim_out, Zero(), white=white, input_prop_dim=D))

        dim_in = dims[-1]
        std_in = kernels[-2].variance.read_value()**0.5 if dim_in > D else 1.
        pad = np.random.randn(M, dim_in - D) * 2. * std_in
        Z_padded = np.concatenate([Z, pad], 1)
        layers.append(Layer(kernels[-1], Z_padded, num_outputs, mean_function, white=white))
        return layers

