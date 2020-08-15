import pdb
import numpy as np
import tensorflow as tf
import gpflow
from gpflow import kullback_leiblers
from gpflow.base import Module, Parameter
from gpflow.covariances import Kuf, Kuu
from gpflow.utilities import positive, triangular
from gpflow.models.util import inducingpoint_wrapper
from gpflow.config import default_float, default_jitter
from utilities import reparameterise
import tensorflow_probability as tfp

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-5)

class Layer(Module):
    """A base glass for GPLAR layers. Basic functionality for multisample 
    conditional and input propagation.
    :inputs_prop_dim: An int, the first dimensions of X to propagate."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

    def conditional(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        return tf.cast(0., dtype=float_type)

    def conditional_SND(self, X, full_cov=False):
        """A multisample conditional, where X has shape [S,N,D], 
        independent over samples.
        :X: A tensor, the input locations [S,N,D].
        :full_cov: A boolean, whether to use the full covariance or not."""
        if full_cov is True:
            f = lambda a: self.conditional(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            # Inputs can be considered independently as diagonal covariance
            X_flat = tf.reshape(X, [S * N, D])
            mean, var = self.conditional(X_flat)
            return [tf.reshape(m, [S, N, 1]) for m in [mean, var]]

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """Computes self.conditional and draws a sample using the 
        reparameterisation trick, adding input propagation if necessary.
        :X: A tensor, input points [S,N,D_in].
        :full_cov: A boolean, whether to calculate full covariance or not.
        :z: A tensor or None, used in reparameterisation trick."""
        mean, var = self.conditional_SND(X, full_cov=full_cov)

        S, N = tf.shape(X)[0], tf.shape(X)[1]

        if z is None:
            z = tf.random.normal(tf.shape(mean), dtype=default_float())

        samples = reparameterise(mean, var, z, full_cov=full_cov)

        return samples, mean, var

class SVGPLayer(Layer):
    """A sparse variational GP layer in whitened representation. This layer 
    holds the kernel, variational parameters, inducing point and mean
    function.
    The underlying model at inputs X is:
    f = Lv + mean_function(X), where v ~ N(0,I) and LL^T = kernel.K(X).
    The variational distribution over the inducing points is:
    q(u) = N(u; q_mu, L_qL_q^T), where L_qL_q^T = q_var.
    The layer holds only one GP.
    :kernel: A gpflow.kernel, the kernel for the layer.
    :inducing_variables: A tensor, the inducing points. [M+i,D_in]
    :index_output i: A scalr number, indicating index of current output.
    :mean_function: A gpflow.mean_function, the mean function for the layer.
    """

    def __init__(self, kernel, inducing_variables, q_mu_initial, q_sqrt_initial,
                 mean_function,white=False, **kwargs):
        super().__init__(**kwargs)

        self.inducing_points = inducing_variables
        
        self.num_inducing = inducing_variables.shape[0]
        m = inducing_variables.shape[1]
        
         # Initialise q_mu to y^2_pi(i)
        q_mu = q_mu_initial[:,None]
        #q_mu = np.zeros((self.num_inducing, 1))
        self.q_mu = Parameter(q_mu, dtype=default_float())

        # Initialise q_sqrt to near deterministic. Store as lower triangular matrix L.
        q_sqrt = 1e-4*np.eye(self.num_inducing, dtype=default_float())
        #q_sqrt = np.diag(q_sqrt_initial)
        self.q_sqrt = Parameter(q_sqrt, transform=triangular())

        self.kernel = kernel
        self.mean_function = mean_function
        self.white = white

        # Initialise to prior (Ku) + jitter.
        #if not self.white:
        #    Ku = self.kernel(self.inducing_points)
        #    Ku += default_jitter()*tf.eye(self.num_inducing, dtype=Ku.dtype)
        #    Lu = tf.linalg.cholesky(Ku)
        #    q_sqrt = Lu
        #    self.q_sqrt = Parameter(q_sqrt, transform=triangular())

    def conditional(self, X, full_cov=False):
        # X is [N,D] or [S*N,D]
        
        #Kmm = Kuu(self.inducing_points, self.kernel, jitter=default_jitter()) #[M,M]
        Kmm = self.kernel(self.inducing_points)
        Kmm += default_jitter()* tf.eye(self.num_inducing, dtype=Kmm.dtype)
        Lmm = tf.linalg.cholesky(Kmm)
        #Kmn = Kuf(self.inducing_points, self.kernel, X) #[M,N]
        Kmn = self.kernel(self.inducing_points, X)
        
        # alpha(X) = k(Z,Z)^{-1}k(Z,X), = L^{-T}L^{-1}k(Z,X)
        A = tf.linalg.triangular_solve(Lmm, Kmn, lower=True) # L^{-1}k(Z,X)
        if not self.white:
            # L^{-T}L^{-1}K(Z,X) is [M,N]
            A = tf.linalg.triangular_solve(tf.transpose(Lmm), A, lower=False)
        
        # m = alpha(X)^T(q_mu - m(Z))
        mean = tf.matmul(A, self.q_mu-self.mean_function(self.inducing_points), 
                         transpose_a=True) # [N,1]
        
        I = tf.eye(self.num_inducing, dtype=default_float())
       
        # var = k(X,X) - alpha(X)^T(k(Z,Z)-q_sqrtq_sqrt^T)alpha(X)
        if self.white: SK = -I
        else: SK = -Kmm 

        if self.q_sqrt is not None: # SK = -k(Z,Z) + q_sqrtq_sqrt^T
            SK += tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True) 
        
        # B = -(k(Z,Z) - q_sqrtq_sqrt^T)alpha(X)
        B = tf.matmul(SK, A) #[M,N]

        if full_cov:
            # delta_cov = -alpha(X)^T(k(Z,Z) - q_sqrtq_sqrt^T)alpha(X)
            delta_cov = tf.matmul(A, B, transpose_a=True) # [N,N]
            Knn = self.kernel(X, full_cov=True, presliced=False)
        else:
            delta_cov = tf.reduce_sum(A * B, 0)
            Knn = self.kernel(X, full_cov=False, presliced=False)
       
        var = Knn + delta_cov
        var = tf.transpose(var)
        
        return mean + self.mean_function(X), var
        # mean is [N,], var is [N,] or [N,N]

    def KL(self):
        """The KL divergence from variational distribution to the prior."""
        if self.white:
            return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt[None,:,:], None)
        else:
            K = self.kernel(self.inducing_points)
            K += default_jitter()* tf.eye(self.num_inducing, dtype=K.dtype)
            return kullback_leiblers.gauss_kl(self.q_mu, self.q_sqrt[None,:,:], K)
    