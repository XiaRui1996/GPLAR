# GPLAR
## Completed: Simple initialisation of inducing inputs and outputs

Initialization of fixed inducing inputs and outputs, i.e. Z=[z,u_{1},...,u_{l-1}]. Firstly, randomly sample M(<<N) data points from training sets. The corresponding observations y_{l} will be inducing outputs which is concatenated to be inputs to next layer.

The mean of the variational distributions are intializaed to be the corresponding observations of the inducing points, the variances are intialized to identity matrix * very small numbers, i.e. start with near deterministic.
