import numpy as np
import sklearn
from matplotlib import pyplot as plt
import pandas as pd
from math import *
from numpy import random
from itertools import product
from sklearn.preprocessing import MaxAbsScaler
import time
from sklearn.gaussian_process import GaussianProcessRegressor as gpr
from sklearn.gaussian_process.kernels import RBF
#from sklearn.gaussian_process.kernels import ConstantKernel
#from sklearn.gaussian_process.kernels import WhiteKernel
from sklearn.gaussian_process.kernels import Matern
#from sklearn.gaussian_process.kernels import DotProduct
#from sklearn.gaussian_process.kernels import RationalQuadratic
import skopt
from skopt import gp_minimize
import joblib
from joblib import parallel_backend
from ray.util.joblib import register_ray
# For parallelizing gpr training
register_ray() # added line.

#
# Written by Truman Russell
# Modefied/Commented by Jacob Rusch
# Marquette University departement of Mechanical Engineering
# 25 May 2023
#
# Text file where data is stored. Data should be organized column-wise with no 
# headers. If there are column titles or any other type of header, need to
# change the impor finction below.
#
# The data should be organized such that the input parameters are the first
# rows and the last row is the RMSE or any other outputs from the simulations
#
# For this specific code, the high- and low- fidelity data are stored in two
# different text files.
#
# Properties in the files are ordereed the following way:
# (Note: Each number represents the column of the property file)
#   1) C11A     |  11) theta_t          |  21) SS
#   2) C12A     |  12) theta_ref_low    |  22) A_hard
#   3) C44A     |  13) lambda_t         |  23) QL
#   4) C11M     |  14) C11A             |  24) theta_ref_high
#   5) C12M     |  15) C12A             |  25) B_sat
#   6) C44M     |  16) C44A             |  26) b1
#   7) alpha_a  |  17) gamma_dot_0      |  27) N_GEN_F
#   8) alpha_m  |  18) Am               |  28) d_ir
#   9) f_c      |  19) S0               |  29) Dir
#  10) gamma_0  |  20) H0               |  30) mu
#
#%% Number of variables that were varried for this analysis
# For the first cycle analysis, the only properties that were varried were:
#   9) f_c
#  11) theta_t
#  13) lambda_t 
#  19) S0
#  20) H0
#  24) theta_ref_high
#  27) N_GEN_F
#  29) Dir
#
#%% Time when code was started
print('Time code started: ',(time.ctime()))
# print("--- %s seconds ---" % (time.time())

#%% HF Data
# HF Data is from Abaqus simulations compared with experimental data
data_high = './GPML_High_Fidelity.txt'
HF_Data=np.array(pd.read_csv(data_high,header=None),dtype="float32")

X_hf_Train=np.array([HF_Data[:,8],HF_Data[:,10],HF_Data[:,12],HF_Data[:,18],\
           HF_Data[:,19],HF_Data[:,23],HF_Data[:,26],HF_Data[:,28]],\
                    dtype="float32").T

y_hf=HF_Data[:,-1:]

#%% LF Data
# Low fidelity data is from mpDriver simulations  compared with experimental 
# data
data_low = './GPML_Low_Fidelity.txt'
LF_Data=np.array(pd.read_csv(data_low,header=None),dtype="float32")

X_lf_Train=np.array([LF_Data[:,8],LF_Data[:,10],LF_Data[:,12],LF_Data[:,18],\
           LF_Data[:,19],LF_Data[:,23],LF_Data[:,26],LF_Data[:,28]]\
                    ,dtype="float32").T
    
y_lf=LF_Data[:,-1:]


#%% Combined data
X_c_unscaled=np.concatenate((X_hf_Train,X_lf_Train))
y_c=np.concatenate((y_hf,y_lf))

# Need to adjust the data so that the inputs and output are the same order of 
# magnitude. GPML struggles when the inputs, output, or both have large
# differences in orders of magnitude.

X_c_scaled = MaxAbsScaler().fit(X_c_unscaled)
X_c = X_c_scaled.transform(X_c_unscaled)

#%% Multi-Fidelity Gaussian Process Regression Set-Up
#
# See the scikit-learn page on GaussianProcessRegressor for details. Many of 
# the comments being made below are either directly coppied from the website
# or they are simplified comments based on the information provided on the
# website.
#
# gpr(kernel=None, *, alpha=1e-10, optimizer='fmin_l_bfgs_b', 
#     n_restarts_optimizer=0, normalize_y=False, copy_X_train=True, 
#     random_state=None)
# 
# kernel : kernel instance, default=None
#  Chosen kernel. See scikit-learn website for available kernels. If value is
#  None the default is: ConstantKernel(1.0, constant_value_bounds="fixed") * 
#  RBF(1.0, length_scale_bounds="fixed"). Note that the kernel hyperparameters 
#  are optimized during fitting unless the bounds are marked as “fixed”.
#
# alpha : float or ndarray of shape (n_samples,), default=1e-10
#  Interpreted as the variance of additional Gaussian measurement noise on the 
#  training observations. The specified value(s) is added to the diagonal of 
#  the kernel matrix during fitting. Note that this is different from using a 
#  WhiteKernel. If an array is passed, it must have the same number of entries 
#  as the data used for fitting and is used as datapoint-dependent noise level.
#
# optimizer“fmin_l_bfgs_b”, callable or None, default=”fmin_l_bfgs_b”
#  Can either be one of the internally supported optimizers for optimizing the 
#  kernel's parameters, specified by a string, or an externally defined 
#  optimizer passed as a callable. If a callable is passed, it must have the 
#  signature:
#
#    def optimizer(obj_func, initial_theta, bounds):
#         # * 'obj_func': the objective function to be minimized, which
#         #   takes the hyperparameters theta as a parameter and an
#         #   optional flag eval_gradient, which determines if the
#         #   gradient is returned additionally to the function value
#         # * 'initial_theta': the initial value for theta, which can be
#         #   used by local optimizers
#         # * 'bounds': the bounds on the values of theta
#         #
#         # Returned are the best found hyperparameters theta and
#         # the corresponding value of the target function.
#         return theta_opt, func_min
#
#  Per default, the L-BFGS-B algorithm from scipy.optimize.minimize is used. 
#  If None is passed, the kernel’s parameters are kept fixed. Available 
#  internal optimizers are: {'fmin_l_bfgs_b'}.
#
# n_restarts_optimizerint, default=0
#  The number of restarts of the optimizer for finding the kernel’s parameters 
#  which maximize the log-marginal likelihood. If greater than 0, all bounds 
#  must be finite. Note that n_restarts_optimizer == 0 implies that one run 
#  is performed.
#
# normalize_ybool, default=False
#  Whether or not to normalize the target values y by removing the mean and 
#  scaling to unit-variance. This is recommended for cases where zero-mean, 
#  unit-variance priors are used. Note that, in this implementation, the 
#  normalisation is reversed before the GP predictions are reported.
#
# copy_X_trainbool, default=True
#  If True, a persistent copy of the training data is stored in the object. 
#  Otherwise, just a reference to the training data is stored, which might 
#  cause predictions to change if the data is modified externally.
#
# random_stateint, RandomState instance or None, default=None
#  Determines random number generation used to initialize the centers. Pass an
#  int for reproducible results across multiple function calls.
#
#
# Methods
# fit(X, y)
#  - Fit Gaussian process regression model.
# get_params([deep])
#  - Get parameters for this estimator.
# log_marginal_likelihood([theta, ...])
#  - Return log-marginal likelihood of theta for training data.
# predict(X[, return_std, return_cov])
#  - Predict using the Gaussian process regression model.
# sample_y(X[, n_samples, random_state])
#  - Draw samples from Gaussian process and evaluate at X.
# score(X, y[, sample_weight])
#  - Return the coefficient of determination of the prediction.
# set_params(**params)
#  - Set the parameters of this estimator.

# Make inputs to explore space
# Exclude where theta_t > theta_ref_high
# These values vreak the physics of the code.
# 
# Multi-Fidelity Gaussian Process Regression Model Training
#kernel=ConstantKernel()*Matern(length_scale=100,length_scale_bounds=("fixed"))
#kernel=RBF(length_scale=100,length_scale_bounds=("fixed"))
#
#kernel=RBF(length_scale=6,length_scale_bounds=(0.001,1000))
#kernel=RBF(length_scale=6,length_scale_bounds=("fixed"))
#kernel = Matern(length_scale=1.0, nu=0.5)
#kernel = RationalQuadratic(length_scale=6.0, alpha=1)
kernel=RBF()
# kernel = Matern()

# Variable a is the noise for input in same position
a=np.ones(len(y_c))
for i in range(len(y_c)):
    if i<len(y_hf):
        # set noise close to zero. Documentation on scikit-learn states that
        # setting to zero might cause stability issues when using the
        # optimization functions.
        a[i]=1e-10
    else:
        a[i]=150

# Calculate how long it takes to train model
start_time = time.time()

# Calling Gaussian Process Regression
with joblib.parallel_backend('ray'):
    reg_c=gpr(kernel=kernel,                # Kernel function
              alpha=a,                      # Noise                       
              optimizer='fmin_l_bfgs_b',    # Optimizer
              n_restarts_optimizer = 500,   # Number of optimizer restarts
              normalize_y=True,             # Scale parameters by removing mean and scaling to unit variance
              copy_X_train=False            # Persistent copy of training data not stored in the object
              ).fit(X_c, y_c)

end = time.time()
print('Time to calibrate in seconds is: ')
print("--- %s seconds ---" % (time.time() - start_time))

##########################
# SAVE-LOAD using joblib #
##########################

# save
joblib.dump(reg_c,'GPR_RBF_Alpha_150_MaxAbsScale.pkl') 

