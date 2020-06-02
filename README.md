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

#### 12/05/2020: 
1. Use different numbers of inducing points per layer. It did reduce the unnecessary uncertainty in first layer.
2. Use GPAR posterior predictive mean as initial value for q_mu. I tried normal long time series and longer time series, and also only with 50 inducing points. It is observed in all cases that even when GPAR posterior predictive mean fail, GPLAR can correct them.
3. Try GPLAR using method in above 2, on real dataset, EEG and Exchange rate. Although the smse metrics are similar for the two models, we can see GPLAR has better calibrated uncertainty in EEG F1 and Exchange USD/AUD.

#### 22/05/2020:
1. Increase observation noise on synthetic data, and compare smse and log-density of GPAR and GPLAR. Line graphs are shown. Log-density of GPLAR is very steady for all three outputs. smse performance are similar.

2. I try to make observations that are close-downwards or close-upwards. GPLAR still fails in "close-upwards"  area missing observations, and the uncertainty estimates can be terrible. I suspect the reason is because, 'F4,F5,FZ,F1,F2' (later five) outputs can perform perfectly only given input x(time), such that the q_mu and the q_sqrt of the first two outputs are not updated. Hence, I try to remove the kernel over inputs for those later outputs without missing data, i.e., 'F4, F5, FZ, F1, F2' only has kernels over preceding outputs. and GPLAR seems to work. However, uncertainty in the subsequent outputs will be unnecessarily high.
   Two method can make GPLAR work on close-upwards observations: 
   1) Let some outputs only use kernels over preceding outputs, i.e. only learn through "cross-channel"
   2) Initialize "q_mu" on missing areas from zeros.

#### 02/06/2020:
1.	Generate synthetic data directly from probabilistic model. I have tried different kernels, such as non-linear over inputs + non-linear over outputs, or + linear over outputs. Both GPAR and GPLAR can fit to those data. GPLAR performs better when a). there is large noise in the observations, b). different noise level in different outputs.

2. I have looked into weird behavior of GPAR over the synthetic third output last time in the log-density vs observation noise, such that log-density of the third output can be extremely high sometimes. I have made a mistake that I didn’t calculate the log-density over held-out datasets, but over training datasets and GPAR can sometimes overfit servely. Instead, I run the held-out log-likelihood vs noise using the synthetic data generated from first task. And this time log-likelihood is calculated over test dataset never observed during training. 100 trials of different noise seed are run and np.percentile are used to produce 2.5-97.5%bounds. Again, the third output held-out loglikelihoods for GPAR are weird when noise variance is close to zero. I checked why, and it turned out that GPAR’s predictive variance for third output are all near zero, (1e-7 level), making the log-likelihood extremely negative if the true value is slightly away from the predictive mean. But GPLAR’s predictive variance is at an appropriate level. 

3.	To make GPLAR work in close-upwards observations. 
   a)	I have tried i). completely remove temporal kernels, ii). using additive kernels over outputs. Both methods cannot work. 
   b)	I also sanity check that when q_mu is initialized over missing areas using the true observations, “q_mu” would not run away from them during optimization. 
   c)	Hence, I implemented the "bi-directional" version of GPLAR, such that a DGP run in reverse is added. And the model can produce good results as shown below, without losing its good performance on close-downwards observations. (I first initialized q_mu to be the corresponding output posterior mean of GPAR, but it trained rather slowly, and then I realized in normal DGP, all q_mu is initialized to zero, which will train much faster with similar time as when reverse DGP are not added. Waste some time on realizing this).
