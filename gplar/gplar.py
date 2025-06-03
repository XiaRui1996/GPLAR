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
from .layers import SVGPLayer
from gpflow.base import Parameter
from .utilities import BroadcastingLikelihood
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
            BH, BHmean, BHvar = backlayer.sample_from_conditional(BH, z=z,
                    full_cov=full_cov)

            Hs.append(Hy)
            Hmeans.append(Hmean)
            Hvars.append(Hvar)

            BHs = [BH,] + BHs
            BHmeans = [BHmean,] + BHmeans
            BHvars = [BHvar,] + BHvars

            H = tf.concat([H,Hy], axis=-1)

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
            y = tf.convert_to_tensor(y, dtype=tf.float64)
            y = tf.reshape(y, [-1, 1])

            var_exp = self.likelihoods[i].variational_expectations(X=X,
                                          Fmu=(Hmean[i] + BHmean[i]) / 2.,
                                          Fvar=(Hvar[i] + BHvar[i]) / 4,
                                          Y=y)
            if self.missing:
                mask = tf.cast(tf.where(available,1.,0.),dtype=var_exp.dtype)
                var_exp = var_exp * tf.tile(mask[None,:],[self.num_samples,1])
            result += tf.reduce_mean(var_exp,0)
        return result

    def prior_kl(self):
        return tf.reduce_sum([layer.KL() for layer in self.layers]) + tf.reduce_sum([layer.KL() for layer in self.backwards_layers])

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
                                          (Hvar[i]+BHvar[i])/4)
        ms.append(mean)
        vs.append(var)
        return np.stack(ms), np.stack(vs)


class GPLAR(GPLARBase):
    """The GPLAR model with zero mean function at each layer"""

    def __init__(self, X, Y, Z, q_sqrt_initial, kernels, likelihoods,
                 mean_function=Zero(), white=False, **kwargs):

        layers = self._init_layers(X, Y, Z, q_sqrt_initial, kernels,
                        mean_function=mean_function, white=white)
        backwards_layers = self._init_backwards_layers(X,Y,Z,mean_function=mean_function, white=white)

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

        for i in range(num_outputs):
            layer = Layer(kernels[i], Z[:,:num_inputs+i], Z[:,num_inputs+i], q_sqrt_initial[:,i], mean_function, white=white)
            layers.append(layer)
            #Z = tf.concate([Z,layer.q_mu], axis=1)

        return layers

    def _init_backwards_layers(self, X, Y, Z, mean_function=Zero(),
                    optimize_inducing_location=True, Layer=SVGPLayer, white=False):
        backlayers = []
        num_inputs=X.shape[1]
        num_outputs=Y.shape[1]
        num_inducing=Z.shape[0]

        for i in range(num_outputs):
            if i==0: inducing_points = Z[:, :num_inputs]
            else: inducing_points = Z[:,num_inputs+num_outputs-i][:,None]
            layer = Layer(SquaredExponential(), inducing_points,
                          Z[:,num_inputs+num_outputs-i-1],
                          [default_jitter()]*num_inducing, mean_function,
                          optimize_inducing_location=optimize_inducing_location,
                          white=white)
            backlayers.append(layer)
        return backlayers

class DeepSMKernel(gpflow.kernels.Kernel):
    """
    k(x,x') = k_RBF( φ(x), φ(x') ) + σ²_white δ
    ϕ = MLP feature extractor
    """
    def __init__(self, in_dim, latent_dim=16, Q=6, white_var=1e-5, name=None):
        super().__init__(name=name)
        # --- feature extractor ---
        self.phi = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", dtype=tf.float64),
            tf.keras.layers.Dense(64, activation="relu", dtype=tf.float64),
            tf.keras.layers.Dense(latent_dim, dtype=tf.float64)                # ϕ(x) ∈ ℝ^{latent_dim}
        ])

        # --- spectral mixture kernel ---
        self.smk = SpectralMixture(num_mixtures=Q, input_dim=latent_dim)

        self.white = White(variance=white_var)

    def K(self, X, X2=None):
        Z1 = self.phi(X)
        Z2 = self.phi(X2) if X2 is not None else None
        return self.smk.K(Z1, Z2) + self.white.K(Z1, Z2)

    def K_diag(self, X):
        Z = self.phi(X)
        return self.smk.K_diag(Z) + self.white.K_diag(Z)

class DeepRBFKernel(Kernel):
    """
    DeepRBFKernel: k(x, x') = k_RBF(ϕ(x), ϕ(x')) + σ²_white δ
    """

    def __init__(self, in_dim, latent_dim=16, white_var=1e-5, name=None):
        super().__init__(name=name)

        # MLP feature extractor: ϕ(x) ∈ ℝ^latent_dim
        self.phi = tf.keras.Sequential([
            tf.keras.layers.Dense(64, activation="relu", dtype=tf.float64),
            tf.keras.layers.Dense(64, activation="relu", dtype=tf.float64),
            tf.keras.layers.Dense(latent_dim, dtype=tf.float64)
        ])

        # Learnable RBF kernel over ϕ(x)
        self.rbf = SquaredExponential(
            variance=Parameter(1.0, transform=positive(), dtype=tf.float64),
            lengthscales=Parameter(tf.ones(latent_dim, dtype=tf.float64), transform=positive())
        )

        # White noise kernel (optional, improves numerical stability)
        self.white = White(variance=white_var)

    def K(self, X, X2=None):
        Z1 = self.phi(X)
        Z2 = self.phi(X2) if X2 is not None else None
        return self.rbf.K(Z1, Z2) + self.white.K(Z1, Z2)

    def K_diag(self, X):
        Z = self.phi(X)
        return self.rbf.K_diag(Z) + self.white.K_diag(Z)



class GPLARegressor(GPLAR):
    def __init__(self, X, Y, M, gpar= None, #missing_data, begin, end, training_columns,
                 deep_kernel = False, latent_dim=16, num_mixtures=6,### DKL
                 deep_rbfkernel = False,
                 reorder = None, minibatch_size=None, missing=False,
                 mean_function=Zero(), white=False, time_ar=True,
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
        self.deep_kernel = deep_kernel
        self.deep_rbfkernel = deep_rbfkernel
        self.latent_dim  = latent_dim
        self.num_mixtures = num_mixtures
        if 'deep_kernel' in kwargs: del kwargs['deep_kernel']
        if 'latent_dim' in kwargs: del kwargs['latent_dim']
        if 'num_mixtures' in kwargs: del kwargs['num_mixtures']
        for k in ['deep_kernel','deep_rbfkernel','latent_dim','deep_rbf_white_var']:
            if k in kwargs: del kwargs[k]
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
        likelihoods = []
        for i in range(self.num_outputs):
            likelihoods.append(Gaussian(variance=noise_obs))

        # Todo: normalize y
        # Todo: impute, handle missing data, make closed down
        # Todo: initialize inducing locations Z
        # Z, q_sqrt_initial = self._initialize_inducing_locations_from_post_GPAR(gpar,X,Y,M)
        Z, q_sqrt_initial = self._initialize_inducing_locations_from_kmeans(X, Y, M)  ########Kmean 选点看这里
        self.initial_inducing_points = Z
        if time_ar:
            self.model_config['time_ar'] = True
        if np.any(np.isnan(Y)): missing = True
        super().__init__(X,Y,Z, q_sqrt_initial, kernels, likelihoods,
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

    def _initialize_inducing_locations_from_kmeans(self, X, Y, M):
        valid_idx = ~np.isnan(Y).any(axis=1)
        X_valid = X[valid_idx]
        Y_valid = Y[valid_idx]
        np.random.seed(42)
        Z_y, labels = kmeans2(Y_valid, M, minit='++')

        Z_x = np.vstack([
            X_valid[labels == i].mean(axis=0)
            for i in range(M)
        ])

        Z = np.hstack([Z_x, Z_y])
        q_sqrt_initial = np.vstack([
            (Y_valid[labels == i].std(axis=0)
            if np.sum(labels==i) > 1
            else np.ones(Y.shape[1]) * 1e-3)
            for i in range(M)
        ])

        return Z, q_sqrt_initial

    def _initialize_inducing_locations_from_post_GPAR(self, gpar, X, Y, M):
                            #, missing_data, begin, end, training_columns):
        t = X[:, :1]
        gpar.fit(t,Y)
        t_min, t_max = t.min(), t.max()
        Z_time = np.linspace(t_min, t_max, M).reshape(M, 1)
        D_in = X.shape[1]
        col_means = np.nanmean(X, axis=0)
        Z_x = np.tile(col_means, (M, 1))
        Z_x[:, 0:1] = Z_time
        samples = gpar.sample(Z_time, num_samples=100, latent=True, posterior=True)

        means, std = np.mean(samples,axis=0), np.std(samples,axis=0)
        # replace missing area inducing points with true observations.
        #Z[begin:end,0] = missing_data.index
        #for name in missing_data.columns:
        #    p = training_columns.index(name)
        #    means[begin:end, p] = missing_data[name]

        Z = np.hstack([Z_x, means])
        return Z, std


    def _kernels_generator(self):
        if self.deep_rbfkernel and self.deep_kernel:
          raise ValueError("Only one of `deep_rbfkernel` or `deep_smkernel` can be True.")

        if getattr(self, "deep_kernel", False):
            dk_list = [
                copy.deepcopy(
                    DeepSMKernel(
                        in_dim     = self.m + self.num_outputs,
                        latent_dim = getattr(self, "latent_dim", 16),
                        white_var  = self.model_config["noise_inner"],
                    )
                )
                for _ in range(self.num_outputs)
            ]
            return dk_list

        if getattr(self, "deep_rbfkernel", False):
            dk_list= [
                copy.deepcopy(
                    DeepRBFKernel(
                        in_dim     = self.m + self.num_outputs,
                        latent_dim = self.latent_dim,
                        white_var  = self.model_config["noise_inner"],
                    )
                )
                for _ in range(self.num_outputs)
            ]


        def get_active_dims_per_layer(num_inputs, num_outputs, markov_order=1):
            active_dims_list = []
            for i in range(num_outputs):
                active_dims = [0]  
                active_dims.append(1 + i)
                active_dims += list(range(num_inputs + i))
                active_dims_list.append(active_dims)
            return active_dims_list
        def _determine_indicies(m,pi,markov):
            # Build in the Markov structure: juggle with the indices of the outputs.
            p_last = pi - 1  # Index of last output that is given as input.
            p_start = 0 if markov is None else max(p_last - (markov - 1), 0)
            p_num = p_last - p_start + 1

            # Determine the indices corresponding to the outputs and inputs.
            m_inds = list(range(m))
            p_inds = list(range(m + p_start, m + p_last + 1))

            return m_inds, p_inds, p_num

        num_inputs = 1 + self.num_outputs
        active_dims_list = get_active_dims_per_layer(
            num_inputs=num_inputs,
            num_outputs=self.num_outputs)

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
                        active_dims = m_inds + p_inds
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

            if self.model_config.get('time_ar', False):
                time_ar_dim = 1 + pi
                time_ar_kernel = Linear(active_dims=[time_ar_dim])
                kernel += time_ar_kernel

            kernels.append(kernel)
        return kernels
