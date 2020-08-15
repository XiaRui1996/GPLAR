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

gpflow.config.set_default_float(np.float64)
gpflow.config.set_default_jitter(1e-6)

class Layer(Module):
    """A base glass for DGP layers. Basic functionality for multisample 
    conditional and input propagation.
    :inputs_prop_dim: An int, the first dimensions of X to propagate."""

    def __init__(self, input_prop_dim=None, **kwargs):
        super().__init__(**kwargs)
        self.input_prop_dim = input_prop_dim

    def conditional_ND(self, X, full_cov=False):
        raise NotImplementedError

    def KL(self):
        return tf.cast(0., dtype=float_type)

    def conditional_SND(self, X, full_cov=False):
        """A multisample conditional, where X has shape [S,N,D], 
        independent over samples.
        :X: A tensor, the input locations [S,N,D].
        :full_cov: A boolean, whether to use the full covariance or not."""
        if full_cov is True:
            f = lambda a: self.conditional_ND(a, full_cov=full_cov)
            mean, var = tf.map_fn(f, X, dtype=(tf.float64, tf.float64))
            return tf.stack(mean), tf.stack(var)
        else:
            S, N, D = tf.shape(X)[0], tf.shape(X)[1], tf.shape(X)[2]
            # Inputs can be considered independently as diagonal covariance
            X_flat = tf.reshape(X, [S * N, D])
            mean, var = self.conditional_ND(X_flat)
            return [tf.reshape(m, [S, N, self.num_outputs]) for m in [mean, var]]

    def sample_from_conditional(self, X, z=None, full_cov=False):
        """Computes self.conditional and draws a sample using the 
        reparameterisation trick, adding input propagation if necessary.
        :X: A tensor, input points [S,N,D_in].
        :full_cov: A boolean, whether to calculate full covariance or not.
        :z: A tensor or None, used in reparameterisation trick."""
        mean, var = self.conditional_SND(X, full_cov=full_cov)

        S, N, D = tf.shape(X)[0], tf.shape(X)[1], self.num_outputs

        if z is None:
            z = tf.random.normal(tf.shape(mean), dtype=default_float())

        samples = reparameterise(mean, var, z, full_cov=full_cov)

        if self.input_prop_dim:
            shape = [S, N, self.input_prop_dim]
            # Get first self.input_prop_dim dimensions of X to propagate
            X_prop = tf.reshape(X[:, :, :self.input_prop_dim], shape)

            samples = tf.concat([X_prop, samples], axis=2) 
            #mean = tf.concat([X_prop, mean], axis=2)

            #if full_cov:
            #    shape = [S, N, N, self.num_outputs]
                # Zero variance for retained dimensions of X
            #    zeros = tf.zeros(shape, dtype=default_float())
            #    var = tf.concat([zeros, var], axis=3)
            #else:
            #    var = tf.concat([tf.zeros_like(X_prop), var], axis=2)
        return samples, mean, var

class SVGPLayer(Layer):
    """A sparse variational GP layer in whitened representation. This layer 
    holds the kernel, variational parameters, inducing point and mean
    function.
    The underlying model at inputs X is:
    f = Lv + mean_function(X), where v ~ N(0,I) and LL^T = kernel.K(X).
    The variational distribution over the inducing points is:
    q(u) = N(u; q_mu, L_qL_q^T), where L_qL_q^T = q_var.
    The layer holds D_out independent GPs with the same kernel and inducing
    points.
    :kernel: A gpflow.kernel, the kernel for the layer.
    :inducing_variables: A tensor, the inducing points. [M,D_in]
    :num_outputs: The number of GP outputs.
    :mean_function: A gpflow.mean_function, the mean function for the layer.
    """

    def __init__(self, kernel, inducing_variables, num_outputs, mean_function,
            input_prop_dim=None, white=False, **kwargs):
        super().__init__(input_prop_dim, **kwargs)

        self.num_inducing = inducing_variables.shape[0]

        # Initialise q_mu to all zeros
        q_mu = np.zeros((self.num_inducing, num_outputs))
        self.q_mu = Parameter(q_mu, dtype=default_float())

        # Initialise q_sqrt to identity function
        #q_sqrt = tf.tile(tf.expand_dims(tf.eye(self.num_inducing, 
        #    dtype=default_float()), 0), (num_outputs, 1, 1))
        q_sqrt = [np.eye(self.num_inducing, dtype=default_float()) for _ in 
                range(num_outputs)]
        q_sqrt = np.array(q_sqrt)
        # Store as lower triangular matrix L.
        self.q_sqrt = Parameter(q_sqrt, transform=triangular())

        self.inducing_points = inducingpoint_wrapper(inducing_variables)

        self.kernel = kernel
        self.mean_function = mean_function
        self.num_outputs = num_outputs
        self.white = white

        # Initialise to prior (Ku) + jitter.
        if not self.white:
            Ku = self.kernel.K(inducing_variables)
            Lu = np.linalg.cholesky(Ku + np.eye(self.num_inducing) * 
                    default_jitter())
            #q_sqrt = tf.tile(tf.expand_dims(Lu, 0), (num_outputs, 1, 1))
            q_sqrt = [Lu for _ in range(num_outputs)]
            q_sqrt = np.array(q_sqrt)
            self.q_sqrt = Parameter(q_sqrt, transform=triangular())

    def conditional_ND(self, X, full_cov=False):
        # X is [S,N,D]
        Kmm = Kuu(self.inducing_points, self.kernel, jitter=default_jitter())
        Lmm = tf.linalg.cholesky(Kmm)
        Kmm_tiled = tf.tile(tf.expand_dims(Kmm, 0), (self.num_outputs, 1, 1))
        Lmm_tiled = tf.tile(tf.expand_dims(Lmm, 0), (self.num_outputs, 1, 1))

        Kmn = Kuf(self.inducing_points, self.kernel, X) # K(Z,X)
        # alpha(X) = k(Z,Z)^{-1}k(Z,X), = L^{-T}L^{-1}k(Z,X)
        A = tf.linalg.triangular_solve(Lmm, Kmn, lower=True) # L^{-1}k(Z,X)
        if not self.white:
            # L^{-T}L^{-1}K(Z,X) is [M,N]
            A = tf.linalg.triangular_solve(tf.transpose(Lmm), A, lower=False)
        
        # m = alpha(X)^T(q_mu - m(Z)) = alpha(X)^T(q_mu) if zero mean function.
        mean = tf.matmul(A, self.q_mu, transpose_a=True) # [N]
        
        # [D_out,M,N]
        A_tiled = tf.tile(A[None, :, :], [self.num_outputs, 1, 1]) 
        I = tf.eye(self.num_inducing, dtype=default_float())[None, :, :]
       
        # var = k(X,X) - alpha(X)^T(k(Z,Z)-q_sqrtq_sqrt^T)alpha(X)
        if self.white:
            SK = -I
        else:
            # -k(Z,Z)
            SK = -Kmm_tiled # [D_out,M,M]

        if self.q_sqrt is not None:
            # SK = -k(Z,Z) + q_sqrtq_sqrt^T
            # [D_out,M,M]
            SK += tf.matmul(self.q_sqrt, self.q_sqrt, transpose_b=True) 
        
        # B = -(k(Z,Z) - q_sqrtq_sqrt^T)alpha(X)
        B = tf.matmul(SK, A_tiled) # [D_out,M,N]

        if full_cov:
            # delta_cov = -alpha(X)^T(k(Z,Z) - q_sqrtq_sqrt^T)alpha(X)
            delta_cov = tf.matmul(A_tiled, B, transpose_a=True) # [D_out,N,N]
            # Knn = k(X,X)
            Knn = self.kernel.K(X)
        else:
            # Summing over dimension 1 --> sum variances due to other.
            # Is this legit?
            delta_cov = tf.reduce_sum(A_tiled * B, 1)
            #delta_cov = tf.linalg.diag_part(tf.matmul(A_tiled, B, 
            #    transpose_a=True)) # [D_out,N]
            Knn = self.kernel.K_diag(X) # [N]
       
        var = tf.expand_dims(Knn, 0) + delta_cov # [D_out,N]
        var = tf.transpose(var)
        
        return mean + self.mean_function(X), var

    def KL(self):
        """The KL divergence from variational distribution to the prior."""
        return kullback_leiblers.prior_kl(self.inducing_points, self.kernel,
                self.q_mu, self.q_sqrt, whiten=self.white)