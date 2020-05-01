# GPLAR
#### 25/04/2020: Simple initialisation of inducing inputs and outputs

Initialization of fixed inducing inputs and outputs, i.e. Z=[z,u_{1},...,u_{l-1}]. Firstly, randomly sample M(<<N) data points from training sets. The corresponding observations y_{l} will be inducing outputs which is concatenated to be inputs to next layer.

The mean of the variational distributions are intializaed to be the corresponding observations of the inducing points, the variances are intialized to identity matrix * very small numbers, i.e. start with near deterministic.

#### 29/04/2020: Handle missing data
Have missing data in say the first / second output, but subsequent outputs are observed.
This can be implemented fairly simply by removing the reconstruction term in the ELBO E( log p(y_1|z_1 )) for the missing outputs.
Comparing with GPAR with impute, the square sum of error for GPLAR is lower. But GPLAR is also easy to be trapped into local minimum. 

###### Problems: 
1. GPLAR is not producing correct uncertainty, it still seems over-confident in second and thrid output
2. GPLAR is not certain in first output, while it should. Because the current pseudo-points strategies, only close-downwards observations can be chosen as pseudo-points. 
