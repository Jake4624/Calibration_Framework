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

#%% Range of inputs
f_c_range = np.array([4,10],dtype="float32")
theta_t_range = np.array([253,318],dtype="float32")
lambda_t_range = np.array([110,240],dtype="float32")
S0_range = np.array([300,800],dtype="float32")
H0_range = np.array([300,1500],dtype="float32")
theta_ref_high_range = np.array([318,383],dtype="float32")
N_GEN_F_range = np.array([2,10],dtype="float32")
Dir_range = np.array([0.0001,0.1],dtype="float32")

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

# Need to predict points in the space to find minimum
N_Pts=8

f_c_explore = np.linspace(f_c_range[0],f_c_range[1],N_Pts,dtype="float32")
theta_t_explore = np.linspace(theta_t_range[0],theta_t_range[1],N_Pts,\
                              dtype="float32")
lambda_t_explore = np.linspace(lambda_t_range[0],lambda_t_range[1],N_Pts,\
                               dtype="float32")
S0_explore = np.linspace(S0_range[0],S0_range[1],N_Pts,dtype="float32")
H0_explore = np.linspace(H0_range[0],H0_range[1],N_Pts,dtype="float32")
theta_ref_high_explore = np.linspace(theta_ref_high_range[0],\
                                     theta_ref_high_range[1],N_Pts,\
                                         dtype="float32")
N_GEN_F_explore = np.linspace(N_GEN_F_range[0],N_GEN_F_range[1],N_Pts,\
                              dtype="float32")
Dir_explore = np.linspace(Dir_range[0],Dir_range[1],N_Pts,dtype="float32")

# Make inputs to explore space
# Exclude where theta_t > theta_ref_high
x_explore_unscaled = np.zeros((N_Pts**8,8),dtype="float32")
counter = 0
for a in range(len(f_c_explore)):
    for b in range(len(theta_t_explore)):
        for c in range(len(lambda_t_explore)):
            for d in range(len(S0_explore)):
                    for e in range(len(H0_explore)):
                            for f in range(len(theta_ref_high_explore)):
                                    for g in range(len(N_GEN_F_explore)):
                                            for h in range(len(Dir_explore)):
                                                if(theta_t_explore[b]>\
                                                   theta_ref_high_explore[f]):
                                                    continue
                                                else:
                                                    x_explore_unscaled[counter,0] = \
                                                    f_c_explore[a]
                                                    x_explore_unscaled[counter,1] = \
                                                    theta_t_explore[b]
                                                    x_explore_unscaled[counter,2] = \
                                                    lambda_t_explore[c]
                                                    x_explore_unscaled[counter,3] = \
                                                    S0_explore[d]
                                                    x_explore_unscaled[counter,4] = \
                                                    H0_explore[e]
                                                    x_explore_unscaled[counter,5] = \
                                                    theta_ref_high_explore[f]
                                                    x_explore_unscaled[counter,6] = \
                                                    N_GEN_F_explore[g]
                                                    x_explore_unscaled[counter,7] = \
                                                    Dir_explore[h]
                                                    counter = counter + 1

# Need to scale this data as well to match the training data
# x_explore_scaled = MaxAbsScaler().fit(x_explore_unscaled)
# x_explore = x_explore_scaled.transform(x_explore_unscaled)

x_explore_scaled = np.float32(MaxAbsScaler().fit(x_explore_unscaled).\
    transform(x_explore_unscaled))

# Due to space limitations, need to split the explore array into smaller sized
# arrays. This allows us to explore more points in the space. For a computer
# with 64 Gb of ram, arrays of length (5^8)x8 is the max size for the code
# to work. smaller laptops and PCs with 16Gb of ram should be able to handle
# arrays of (4^8)x8 and still be able to run.

num_rows = 300

x_explore = np.array_split(x_explore_scaled,num_rows,axis=0)

    
#%% Load trained model in
# Model that has been trained and saved using the other script can be loaded
# in here. This is broken up to reduce memory usage
# If the model has been trained previously and you wish to explore different
# points without retraining, can load in the model with the fuction below.
reg_c = joblib.load("GPR_RBF_Alpha_150_MaxAbsScale.pkl")

        
#%% Multi-Fidelity Gaussian Process Regression Prediction        
# Controls number of standard deviations to fill the space
# 1.96 is the z value for 95% confidence interval
n_explore=1.96
#
y_explore=np.zeros((N_Pts**8,1),dtype="float32")
v_explore=np.zeros((N_Pts**8,1),dtype="float32")

y_explore_split = np.array_split(y_explore,num_rows,axis=0)
v_explore_split = np.array_split(v_explore,num_rows,axis=0)

# Calculate how long it takes to explore/predict space
start_time = time.time()

for i in range(num_rows):
    print(i)
    y_explore_split[i],v_explore_split[i]=reg_c.predict(x_explore[i],\
                                                        return_std=True)

end = time.time()
print('Time to explore/predict in seconds is: ')
print("--- %s seconds ---" % (time.time() - start_time))

#%%

y_explore = np.array([])
v_explore = np.array([])

for i in range(len(y_explore_split)):
    y_explore = np.append(y_explore,y_explore_split[i])
    v_explore = np.append(v_explore,v_explore_split[i])

#%%
# UNSURE OF WHY Y IS RETURNED AS SIZE (X,1) INSTEAD OF (X,). NEED TO FLATTEN
# IT IN ORDER TO CALCULATE YH AND YL
y_explore = y_explore.flatten()
yh_explore = y_explore+n_explore*v_explore
yl_explore = y_explore-n_explore*v_explore


# Find minimum as predicted by model and its position in the output array
# Can use the position to find the inputs that produced the minimum.
min_GPR_mean_func = min(y_explore)
position_min_GPR_mean_func = np.asarray(np.where(y_explore ==\
                                                 min_GPR_mean_func)).T
    
    
# Finding values of inputs at the minimum output position
# Need to rescale these values to their original order of magnitude

min_inputs_mean_func = \
    np.array([x_explore_unscaled[position_min_GPR_mean_func[0,0],0],\
              x_explore_unscaled[position_min_GPR_mean_func[0,0],1],\
              x_explore_unscaled[position_min_GPR_mean_func[0,0],2],\
              x_explore_unscaled[position_min_GPR_mean_func[0,0],3],\
              x_explore_unscaled[position_min_GPR_mean_func[0,0],4],\
              x_explore_unscaled[position_min_GPR_mean_func[0,0],5],\
              x_explore_unscaled[position_min_GPR_mean_func[0,0],6],\
              x_explore_unscaled[position_min_GPR_mean_func[0,0],7]])

print('Min Predicted error is using mean function: ', min_GPR_mean_func)
print('Confidence Interval using mean function: ', min_GPR_mean_func)
print('Location of predicted min value using mean function: ', \
      min_inputs_mean_func)

min_GPR_lower_bound_func = min(yl_explore)
position_min_GPR_lower_bound_func = np.asarray(np.where(yl_explore ==\
                                                 min_GPR_lower_bound_func)).T

# Finding values of inputs at the minimum output position
# Need to rescale these values to their original order of magnitude


min_inputs_lower_bound_func = \
    np.array([x_explore_unscaled[position_min_GPR_lower_bound_func[0,0],0],\
              x_explore_unscaled[position_min_GPR_lower_bound_func[0,0],1],\
              x_explore_unscaled[position_min_GPR_lower_bound_func[0,0],2],\
              x_explore_unscaled[position_min_GPR_lower_bound_func[0,0],3],\
              x_explore_unscaled[position_min_GPR_lower_bound_func[0,0],4],\
              x_explore_unscaled[position_min_GPR_lower_bound_func[0,0],5],\
              x_explore_unscaled[position_min_GPR_lower_bound_func[0,0],6],\
              x_explore_unscaled[position_min_GPR_lower_bound_func[0,0],7]])

print('Min Predicted error is using lower bound function: ', \
      min_GPR_lower_bound_func)
print('Location of predicted min value using lower bound function: ', \
      min_inputs_lower_bound_func)

# With the new minimum values, need to go back to Abaqus to run a new 
# simulation and compare to experimental data, find the error, add it to the
# training dataset, and redo this analysis. Repeat until difference between 
# HF simulations and experimental data is small (within whatever tolerance
# is deemed acceptible for the particular application)

#%% Check if Min is Same or Not
# Check to make sure the minimum predicted points are not the same between
# the two different acquisition functions

if np.array_equal(min_inputs_mean_func,min_inputs_lower_bound_func) == True:
    input_props = min_inputs_mean_func
    numsim = 1
else:
    input_props = np.stack((min_inputs_mean_func,\
                                  min_inputs_lower_bound_func))
    numsim = 2

#%% Write New Input File to Run 

numElsets = 64
DEPVAR = 234

# Orientation file. Make sure this is consistent with the number of grains. 
# Need to have the same number of lines in the texture file as there are 
# grains. Euler angles for texture are made in a separate file called
# writeTexture.py
orientFile = 'texture.inc'
orient = np.loadtxt(orientFile,delimiter=',')

for i in range(numsim):
    mat_props = np.array([130000,98000,34000,130000,98000,34000,1.1e-05,\
                          6.6e-06,input_props[i,0],0.1308,input_props[i,1],\
                          253,input_props[i,2],130000,98000,34000,0.002,\
                          0.02,input_props[i,3],input_props[i,4],900,\
                          0.125,1.4,input_props[i,5],150,2,input_props[i,6],\
                          0.001,input_props[i,7],0.2])
    f = open(str(i+1)+'.inp','w')
    f.write("""**
**  Input file for a cube of unit dimension in all directions
** 
**   3D Visualization of cube
** 
**               ----------------
**              /|             /|
**             / |            / |
**            ---+------------  |
**            |  |           |  |          Y
**            |  |           |  |          |
**            |  |           |  |          |
**            |  ------------+---          +------X
**            | /            | /          /
**            |/             |/          /
**            ----------------          Z
** 

*Heading
*Preprint, echo=NO, model=NO, history=NO, contact=NO
**
** PARTS
** 
*Part, name=Part-1
*Node
1	,0.0	,0.0	,0.0
2	,0.166666666667	,0.0	,0.0
3	,0.333333333333	,0.0	,0.0
4	,0.5	,0.0	,0.0
5	,0.666666666667	,0.0	,0.0
6	,0.833333333333	,0.0	,0.0
7	,1.0	,0.0	,0.0
8	,0.0	,0.166666666667	,0.0
9	,0.166666666667	,0.166666666667	,0.0
10	,0.333333333333	,0.166666666667	,0.0
11	,0.5	,0.166666666667	,0.0
12	,0.666666666667	,0.166666666667	,0.0
13	,0.833333333333	,0.166666666667	,0.0
14	,1.0	,0.166666666667	,0.0
15	,0.0	,0.333333333333	,0.0
16	,0.166666666667	,0.333333333333	,0.0
17	,0.333333333333	,0.333333333333	,0.0
18	,0.5	,0.333333333333	,0.0
19	,0.666666666667	,0.333333333333	,0.0
20	,0.833333333333	,0.333333333333	,0.0
21	,1.0	,0.333333333333	,0.0
22	,0.0	,0.5	,0.0
23	,0.166666666667	,0.5	,0.0
24	,0.333333333333	,0.5	,0.0
25	,0.5	,0.5	,0.0
26	,0.666666666667	,0.5	,0.0
27	,0.833333333333	,0.5	,0.0
28	,1.0	,0.5	,0.0
29	,0.0	,0.666666666667	,0.0
30	,0.166666666667	,0.666666666667	,0.0
31	,0.333333333333	,0.666666666667	,0.0
32	,0.5	,0.666666666667	,0.0
33	,0.666666666667	,0.666666666667	,0.0
34	,0.833333333333	,0.666666666667	,0.0
35	,1.0	,0.666666666667	,0.0
36	,0.0	,0.833333333333	,0.0
37	,0.166666666667	,0.833333333333	,0.0
38	,0.333333333333	,0.833333333333	,0.0
39	,0.5	,0.833333333333	,0.0
40	,0.666666666667	,0.833333333333	,0.0
41	,0.833333333333	,0.833333333333	,0.0
42	,1.0	,0.833333333333	,0.0
43	,0.0	,1.0	,0.0
44	,0.166666666667	,1.0	,0.0
45	,0.333333333333	,1.0	,0.0
46	,0.5	,1.0	,0.0
47	,0.666666666667	,1.0	,0.0
48	,0.833333333333	,1.0	,0.0
49	,1.0	,1.0	,0.0
50	,0.0	,0.0	,0.166666666667
51	,0.166666666667	,0.0	,0.166666666667
52	,0.333333333333	,0.0	,0.166666666667
53	,0.5	,0.0	,0.166666666667
54	,0.666666666667	,0.0	,0.166666666667
55	,0.833333333333	,0.0	,0.166666666667
56	,1.0	,0.0	,0.166666666667
57	,0.0	,0.166666666667	,0.166666666667
58	,0.166666666667	,0.166666666667	,0.166666666667
59	,0.333333333333	,0.166666666667	,0.166666666667
60	,0.5	,0.166666666667	,0.166666666667
61	,0.666666666667	,0.166666666667	,0.166666666667
62	,0.833333333333	,0.166666666667	,0.166666666667
63	,1.0	,0.166666666667	,0.166666666667
64	,0.0	,0.333333333333	,0.166666666667
65	,0.166666666667	,0.333333333333	,0.166666666667
66	,0.333333333333	,0.333333333333	,0.166666666667
67	,0.5	,0.333333333333	,0.166666666667
68	,0.666666666667	,0.333333333333	,0.166666666667
69	,0.833333333333	,0.333333333333	,0.166666666667
70	,1.0	,0.333333333333	,0.166666666667
71	,0.0	,0.5	,0.166666666667
72	,0.166666666667	,0.5	,0.166666666667
73	,0.333333333333	,0.5	,0.166666666667
74	,0.5	,0.5	,0.166666666667
75	,0.666666666667	,0.5	,0.166666666667
76	,0.833333333333	,0.5	,0.166666666667
77	,1.0	,0.5	,0.166666666667
78	,0.0	,0.666666666667	,0.166666666667
79	,0.166666666667	,0.666666666667	,0.166666666667
80	,0.333333333333	,0.666666666667	,0.166666666667
81	,0.5	,0.666666666667	,0.166666666667
82	,0.666666666667	,0.666666666667	,0.166666666667
83	,0.833333333333	,0.666666666667	,0.166666666667
84	,1.0	,0.666666666667	,0.166666666667
85	,0.0	,0.833333333333	,0.166666666667
86	,0.166666666667	,0.833333333333	,0.166666666667
87	,0.333333333333	,0.833333333333	,0.166666666667
88	,0.5	,0.833333333333	,0.166666666667
89	,0.666666666667	,0.833333333333	,0.166666666667
90	,0.833333333333	,0.833333333333	,0.166666666667
91	,1.0	,0.833333333333	,0.166666666667
92	,0.0	,1.0	,0.166666666667
93	,0.166666666667	,1.0	,0.166666666667
94	,0.333333333333	,1.0	,0.166666666667
95	,0.5	,1.0	,0.166666666667
96	,0.666666666667	,1.0	,0.166666666667
97	,0.833333333333	,1.0	,0.166666666667
98	,1.0	,1.0	,0.166666666667
99	,0.0	,0.0	,0.333333333333
100	,0.166666666667	,0.0	,0.333333333333
101	,0.333333333333	,0.0	,0.333333333333
102	,0.5	,0.0	,0.333333333333
103	,0.666666666667	,0.0	,0.333333333333
104	,0.833333333333	,0.0	,0.333333333333
105	,1.0	,0.0	,0.333333333333
106	,0.0	,0.166666666667	,0.333333333333
107	,0.166666666667	,0.166666666667	,0.333333333333
108	,0.333333333333	,0.166666666667	,0.333333333333
109	,0.5	,0.166666666667	,0.333333333333
110	,0.666666666667	,0.166666666667	,0.333333333333
111	,0.833333333333	,0.166666666667	,0.333333333333
112	,1.0	,0.166666666667	,0.333333333333
113	,0.0	,0.333333333333	,0.333333333333
114	,0.166666666667	,0.333333333333	,0.333333333333
115	,0.333333333333	,0.333333333333	,0.333333333333
116	,0.5	,0.333333333333	,0.333333333333
117	,0.666666666667	,0.333333333333	,0.333333333333
118	,0.833333333333	,0.333333333333	,0.333333333333
119	,1.0	,0.333333333333	,0.333333333333
120	,0.0	,0.5	,0.333333333333
121	,0.166666666667	,0.5	,0.333333333333
122	,0.333333333333	,0.5	,0.333333333333
123	,0.5	,0.5	,0.333333333333
124	,0.666666666667	,0.5	,0.333333333333
125	,0.833333333333	,0.5	,0.333333333333
126	,1.0	,0.5	,0.333333333333
127	,0.0	,0.666666666667	,0.333333333333
128	,0.166666666667	,0.666666666667	,0.333333333333
129	,0.333333333333	,0.666666666667	,0.333333333333
130	,0.5	,0.666666666667	,0.333333333333
131	,0.666666666667	,0.666666666667	,0.333333333333
132	,0.833333333333	,0.666666666667	,0.333333333333
133	,1.0	,0.666666666667	,0.333333333333
134	,0.0	,0.833333333333	,0.333333333333
135	,0.166666666667	,0.833333333333	,0.333333333333
136	,0.333333333333	,0.833333333333	,0.333333333333
137	,0.5	,0.833333333333	,0.333333333333
138	,0.666666666667	,0.833333333333	,0.333333333333
139	,0.833333333333	,0.833333333333	,0.333333333333
140	,1.0	,0.833333333333	,0.333333333333
141	,0.0	,1.0	,0.333333333333
142	,0.166666666667	,1.0	,0.333333333333
143	,0.333333333333	,1.0	,0.333333333333
144	,0.5	,1.0	,0.333333333333
145	,0.666666666667	,1.0	,0.333333333333
146	,0.833333333333	,1.0	,0.333333333333
147	,1.0	,1.0	,0.333333333333
148	,0.0	,0.0	,0.5
149	,0.166666666667	,0.0	,0.5
150	,0.333333333333	,0.0	,0.5
151	,0.5	,0.0	,0.5
152	,0.666666666667	,0.0	,0.5
153	,0.833333333333	,0.0	,0.5
154	,1.0	,0.0	,0.5
155	,0.0	,0.166666666667	,0.5
156	,0.166666666667	,0.166666666667	,0.5
157	,0.333333333333	,0.166666666667	,0.5
158	,0.5	,0.166666666667	,0.5
159	,0.666666666667	,0.166666666667	,0.5
160	,0.833333333333	,0.166666666667	,0.5
161	,1.0	,0.166666666667	,0.5
162	,0.0	,0.333333333333	,0.5
163	,0.166666666667	,0.333333333333	,0.5
164	,0.333333333333	,0.333333333333	,0.5
165	,0.5	,0.333333333333	,0.5
166	,0.666666666667	,0.333333333333	,0.5
167	,0.833333333333	,0.333333333333	,0.5
168	,1.0	,0.333333333333	,0.5
169	,0.0	,0.5	,0.5
170	,0.166666666667	,0.5	,0.5
171	,0.333333333333	,0.5	,0.5
172	,0.5	,0.5	,0.5
173	,0.666666666667	,0.5	,0.5
174	,0.833333333333	,0.5	,0.5
175	,1.0	,0.5	,0.5
176	,0.0	,0.666666666667	,0.5
177	,0.166666666667	,0.666666666667	,0.5
178	,0.333333333333	,0.666666666667	,0.5
179	,0.5	,0.666666666667	,0.5
180	,0.666666666667	,0.666666666667	,0.5
181	,0.833333333333	,0.666666666667	,0.5
182	,1.0	,0.666666666667	,0.5
183	,0.0	,0.833333333333	,0.5
184	,0.166666666667	,0.833333333333	,0.5
185	,0.333333333333	,0.833333333333	,0.5
186	,0.5	,0.833333333333	,0.5
187	,0.666666666667	,0.833333333333	,0.5
188	,0.833333333333	,0.833333333333	,0.5
189	,1.0	,0.833333333333	,0.5
190	,0.0	,1.0	,0.5
191	,0.166666666667	,1.0	,0.5
192	,0.333333333333	,1.0	,0.5
193	,0.5	,1.0	,0.5
194	,0.666666666667	,1.0	,0.5
195	,0.833333333333	,1.0	,0.5
196	,1.0	,1.0	,0.5
197	,0.0	,0.0	,0.666666666667
198	,0.166666666667	,0.0	,0.666666666667
199	,0.333333333333	,0.0	,0.666666666667
200	,0.5	,0.0	,0.666666666667
201	,0.666666666667	,0.0	,0.666666666667
202	,0.833333333333	,0.0	,0.666666666667
203	,1.0	,0.0	,0.666666666667
204	,0.0	,0.166666666667	,0.666666666667
205	,0.166666666667	,0.166666666667	,0.666666666667
206	,0.333333333333	,0.166666666667	,0.666666666667
207	,0.5	,0.166666666667	,0.666666666667
208	,0.666666666667	,0.166666666667	,0.666666666667
209	,0.833333333333	,0.166666666667	,0.666666666667
210	,1.0	,0.166666666667	,0.666666666667
211	,0.0	,0.333333333333	,0.666666666667
212	,0.166666666667	,0.333333333333	,0.666666666667
213	,0.333333333333	,0.333333333333	,0.666666666667
214	,0.5	,0.333333333333	,0.666666666667
215	,0.666666666667	,0.333333333333	,0.666666666667
216	,0.833333333333	,0.333333333333	,0.666666666667
217	,1.0	,0.333333333333	,0.666666666667
218	,0.0	,0.5	,0.666666666667
219	,0.166666666667	,0.5	,0.666666666667
220	,0.333333333333	,0.5	,0.666666666667
221	,0.5	,0.5	,0.666666666667
222	,0.666666666667	,0.5	,0.666666666667
223	,0.833333333333	,0.5	,0.666666666667
224	,1.0	,0.5	,0.666666666667
225	,0.0	,0.666666666667	,0.666666666667
226	,0.166666666667	,0.666666666667	,0.666666666667
227	,0.333333333333	,0.666666666667	,0.666666666667
228	,0.5	,0.666666666667	,0.666666666667
229	,0.666666666667	,0.666666666667	,0.666666666667
230	,0.833333333333	,0.666666666667	,0.666666666667
231	,1.0	,0.666666666667	,0.666666666667
232	,0.0	,0.833333333333	,0.666666666667
233	,0.166666666667	,0.833333333333	,0.666666666667
234	,0.333333333333	,0.833333333333	,0.666666666667
235	,0.5	,0.833333333333	,0.666666666667
236	,0.666666666667	,0.833333333333	,0.666666666667
237	,0.833333333333	,0.833333333333	,0.666666666667
238	,1.0	,0.833333333333	,0.666666666667
239	,0.0	,1.0	,0.666666666667
240	,0.166666666667	,1.0	,0.666666666667
241	,0.333333333333	,1.0	,0.666666666667
242	,0.5	,1.0	,0.666666666667
243	,0.666666666667	,1.0	,0.666666666667
244	,0.833333333333	,1.0	,0.666666666667
245	,1.0	,1.0	,0.666666666667
246	,0.0	,0.0	,0.833333333333
247	,0.166666666667	,0.0	,0.833333333333
248	,0.333333333333	,0.0	,0.833333333333
249	,0.5	,0.0	,0.833333333333
250	,0.666666666667	,0.0	,0.833333333333
251	,0.833333333333	,0.0	,0.833333333333
252	,1.0	,0.0	,0.833333333333
253	,0.0	,0.166666666667	,0.833333333333
254	,0.166666666667	,0.166666666667	,0.833333333333
255	,0.333333333333	,0.166666666667	,0.833333333333
256	,0.5	,0.166666666667	,0.833333333333
257	,0.666666666667	,0.166666666667	,0.833333333333
258	,0.833333333333	,0.166666666667	,0.833333333333
259	,1.0	,0.166666666667	,0.833333333333
260	,0.0	,0.333333333333	,0.833333333333
261	,0.166666666667	,0.333333333333	,0.833333333333
262	,0.333333333333	,0.333333333333	,0.833333333333
263	,0.5	,0.333333333333	,0.833333333333
264	,0.666666666667	,0.333333333333	,0.833333333333
265	,0.833333333333	,0.333333333333	,0.833333333333
266	,1.0	,0.333333333333	,0.833333333333
267	,0.0	,0.5	,0.833333333333
268	,0.166666666667	,0.5	,0.833333333333
269	,0.333333333333	,0.5	,0.833333333333
270	,0.5	,0.5	,0.833333333333
271	,0.666666666667	,0.5	,0.833333333333
272	,0.833333333333	,0.5	,0.833333333333
273	,1.0	,0.5	,0.833333333333
274	,0.0	,0.666666666667	,0.833333333333
275	,0.166666666667	,0.666666666667	,0.833333333333
276	,0.333333333333	,0.666666666667	,0.833333333333
277	,0.5	,0.666666666667	,0.833333333333
278	,0.666666666667	,0.666666666667	,0.833333333333
279	,0.833333333333	,0.666666666667	,0.833333333333
280	,1.0	,0.666666666667	,0.833333333333
281	,0.0	,0.833333333333	,0.833333333333
282	,0.166666666667	,0.833333333333	,0.833333333333
283	,0.333333333333	,0.833333333333	,0.833333333333
284	,0.5	,0.833333333333	,0.833333333333
285	,0.666666666667	,0.833333333333	,0.833333333333
286	,0.833333333333	,0.833333333333	,0.833333333333
287	,1.0	,0.833333333333	,0.833333333333
288	,0.0	,1.0	,0.833333333333
289	,0.166666666667	,1.0	,0.833333333333
290	,0.333333333333	,1.0	,0.833333333333
291	,0.5	,1.0	,0.833333333333
292	,0.666666666667	,1.0	,0.833333333333
293	,0.833333333333	,1.0	,0.833333333333
294	,1.0	,1.0	,0.833333333333
295	,0.0	,0.0	,1.0
296	,0.166666666667	,0.0	,1.0
297	,0.333333333333	,0.0	,1.0
298	,0.5	,0.0	,1.0
299	,0.666666666667	,0.0	,1.0
300	,0.833333333333	,0.0	,1.0
301	,1.0	,0.0	,1.0
302	,0.0	,0.166666666667	,1.0
303	,0.166666666667	,0.166666666667	,1.0
304	,0.333333333333	,0.166666666667	,1.0
305	,0.5	,0.166666666667	,1.0
306	,0.666666666667	,0.166666666667	,1.0
307	,0.833333333333	,0.166666666667	,1.0
308	,1.0	,0.166666666667	,1.0
309	,0.0	,0.333333333333	,1.0
310	,0.166666666667	,0.333333333333	,1.0
311	,0.333333333333	,0.333333333333	,1.0
312	,0.5	,0.333333333333	,1.0
313	,0.666666666667	,0.333333333333	,1.0
314	,0.833333333333	,0.333333333333	,1.0
315	,1.0	,0.333333333333	,1.0
316	,0.0	,0.5	,1.0
317	,0.166666666667	,0.5	,1.0
318	,0.333333333333	,0.5	,1.0
319	,0.5	,0.5	,1.0
320	,0.666666666667	,0.5	,1.0
321	,0.833333333333	,0.5	,1.0
322	,1.0	,0.5	,1.0
323	,0.0	,0.666666666667	,1.0
324	,0.166666666667	,0.666666666667	,1.0
325	,0.333333333333	,0.666666666667	,1.0
326	,0.5	,0.666666666667	,1.0
327	,0.666666666667	,0.666666666667	,1.0
328	,0.833333333333	,0.666666666667	,1.0
329	,1.0	,0.666666666667	,1.0
330	,0.0	,0.833333333333	,1.0
331	,0.166666666667	,0.833333333333	,1.0
332	,0.333333333333	,0.833333333333	,1.0
333	,0.5	,0.833333333333	,1.0
334	,0.666666666667	,0.833333333333	,1.0
335	,0.833333333333	,0.833333333333	,1.0
336	,1.0	,0.833333333333	,1.0
337	,0.0	,1.0	,1.0
338	,0.166666666667	,1.0	,1.0
339	,0.333333333333	,1.0	,1.0
340	,0.5	,1.0	,1.0
341	,0.666666666667	,1.0	,1.0
342	,0.833333333333	,1.0	,1.0
343	,1.0	,1.0	,1.0
*Element, type=C3D8R
1	,1	,2	,9	,8	,50	,51	,58	,57
2	,2	,3	,10	,9	,51	,52	,59	,58
3	,3	,4	,11	,10	,52	,53	,60	,59
4	,4	,5	,12	,11	,53	,54	,61	,60
5	,5	,6	,13	,12	,54	,55	,62	,61
6	,6	,7	,14	,13	,55	,56	,63	,62
7	,8	,9	,16	,15	,57	,58	,65	,64
8	,9	,10	,17	,16	,58	,59	,66	,65
9	,10	,11	,18	,17	,59	,60	,67	,66
10	,11	,12	,19	,18	,60	,61	,68	,67
11	,12	,13	,20	,19	,61	,62	,69	,68
12	,13	,14	,21	,20	,62	,63	,70	,69
13	,15	,16	,23	,22	,64	,65	,72	,71
14	,16	,17	,24	,23	,65	,66	,73	,72
15	,17	,18	,25	,24	,66	,67	,74	,73
16	,18	,19	,26	,25	,67	,68	,75	,74
17	,19	,20	,27	,26	,68	,69	,76	,75
18	,20	,21	,28	,27	,69	,70	,77	,76
19	,22	,23	,30	,29	,71	,72	,79	,78
20	,23	,24	,31	,30	,72	,73	,80	,79
21	,24	,25	,32	,31	,73	,74	,81	,80
22	,25	,26	,33	,32	,74	,75	,82	,81
23	,26	,27	,34	,33	,75	,76	,83	,82
24	,27	,28	,35	,34	,76	,77	,84	,83
25	,29	,30	,37	,36	,78	,79	,86	,85
26	,30	,31	,38	,37	,79	,80	,87	,86
27	,31	,32	,39	,38	,80	,81	,88	,87
28	,32	,33	,40	,39	,81	,82	,89	,88
29	,33	,34	,41	,40	,82	,83	,90	,89
30	,34	,35	,42	,41	,83	,84	,91	,90
31	,36	,37	,44	,43	,85	,86	,93	,92
32	,37	,38	,45	,44	,86	,87	,94	,93
33	,38	,39	,46	,45	,87	,88	,95	,94
34	,39	,40	,47	,46	,88	,89	,96	,95
35	,40	,41	,48	,47	,89	,90	,97	,96
36	,41	,42	,49	,48	,90	,91	,98	,97
37	,50	,51	,58	,57	,99	,100	,107	,106
38	,51	,52	,59	,58	,100	,101	,108	,107
39	,52	,53	,60	,59	,101	,102	,109	,108
40	,53	,54	,61	,60	,102	,103	,110	,109
41	,54	,55	,62	,61	,103	,104	,111	,110
42	,55	,56	,63	,62	,104	,105	,112	,111
43	,57	,58	,65	,64	,106	,107	,114	,113
44	,58	,59	,66	,65	,107	,108	,115	,114
45	,59	,60	,67	,66	,108	,109	,116	,115
46	,60	,61	,68	,67	,109	,110	,117	,116
47	,61	,62	,69	,68	,110	,111	,118	,117
48	,62	,63	,70	,69	,111	,112	,119	,118
49	,64	,65	,72	,71	,113	,114	,121	,120
50	,65	,66	,73	,72	,114	,115	,122	,121
51	,66	,67	,74	,73	,115	,116	,123	,122
52	,67	,68	,75	,74	,116	,117	,124	,123
53	,68	,69	,76	,75	,117	,118	,125	,124
54	,69	,70	,77	,76	,118	,119	,126	,125
55	,71	,72	,79	,78	,120	,121	,128	,127
56	,72	,73	,80	,79	,121	,122	,129	,128
57	,73	,74	,81	,80	,122	,123	,130	,129
58	,74	,75	,82	,81	,123	,124	,131	,130
59	,75	,76	,83	,82	,124	,125	,132	,131
60	,76	,77	,84	,83	,125	,126	,133	,132
61	,78	,79	,86	,85	,127	,128	,135	,134
62	,79	,80	,87	,86	,128	,129	,136	,135
63	,80	,81	,88	,87	,129	,130	,137	,136
64	,81	,82	,89	,88	,130	,131	,138	,137
65	,82	,83	,90	,89	,131	,132	,139	,138
66	,83	,84	,91	,90	,132	,133	,140	,139
67	,85	,86	,93	,92	,134	,135	,142	,141
68	,86	,87	,94	,93	,135	,136	,143	,142
69	,87	,88	,95	,94	,136	,137	,144	,143
70	,88	,89	,96	,95	,137	,138	,145	,144
71	,89	,90	,97	,96	,138	,139	,146	,145
72	,90	,91	,98	,97	,139	,140	,147	,146
73	,99	,100	,107	,106	,148	,149	,156	,155
74	,100	,101	,108	,107	,149	,150	,157	,156
75	,101	,102	,109	,108	,150	,151	,158	,157
76	,102	,103	,110	,109	,151	,152	,159	,158
77	,103	,104	,111	,110	,152	,153	,160	,159
78	,104	,105	,112	,111	,153	,154	,161	,160
79	,106	,107	,114	,113	,155	,156	,163	,162
80	,107	,108	,115	,114	,156	,157	,164	,163
81	,108	,109	,116	,115	,157	,158	,165	,164
82	,109	,110	,117	,116	,158	,159	,166	,165
83	,110	,111	,118	,117	,159	,160	,167	,166
84	,111	,112	,119	,118	,160	,161	,168	,167
85	,113	,114	,121	,120	,162	,163	,170	,169
86	,114	,115	,122	,121	,163	,164	,171	,170
87	,115	,116	,123	,122	,164	,165	,172	,171
88	,116	,117	,124	,123	,165	,166	,173	,172
89	,117	,118	,125	,124	,166	,167	,174	,173
90	,118	,119	,126	,125	,167	,168	,175	,174
91	,120	,121	,128	,127	,169	,170	,177	,176
92	,121	,122	,129	,128	,170	,171	,178	,177
93	,122	,123	,130	,129	,171	,172	,179	,178
94	,123	,124	,131	,130	,172	,173	,180	,179
95	,124	,125	,132	,131	,173	,174	,181	,180
96	,125	,126	,133	,132	,174	,175	,182	,181
97	,127	,128	,135	,134	,176	,177	,184	,183
98	,128	,129	,136	,135	,177	,178	,185	,184
99	,129	,130	,137	,136	,178	,179	,186	,185
100	,130	,131	,138	,137	,179	,180	,187	,186
101	,131	,132	,139	,138	,180	,181	,188	,187
102	,132	,133	,140	,139	,181	,182	,189	,188
103	,134	,135	,142	,141	,183	,184	,191	,190
104	,135	,136	,143	,142	,184	,185	,192	,191
105	,136	,137	,144	,143	,185	,186	,193	,192
106	,137	,138	,145	,144	,186	,187	,194	,193
107	,138	,139	,146	,145	,187	,188	,195	,194
108	,139	,140	,147	,146	,188	,189	,196	,195
109	,148	,149	,156	,155	,197	,198	,205	,204
110	,149	,150	,157	,156	,198	,199	,206	,205
111	,150	,151	,158	,157	,199	,200	,207	,206
112	,151	,152	,159	,158	,200	,201	,208	,207
113	,152	,153	,160	,159	,201	,202	,209	,208
114	,153	,154	,161	,160	,202	,203	,210	,209
115	,155	,156	,163	,162	,204	,205	,212	,211
116	,156	,157	,164	,163	,205	,206	,213	,212
117	,157	,158	,165	,164	,206	,207	,214	,213
118	,158	,159	,166	,165	,207	,208	,215	,214
119	,159	,160	,167	,166	,208	,209	,216	,215
120	,160	,161	,168	,167	,209	,210	,217	,216
121	,162	,163	,170	,169	,211	,212	,219	,218
122	,163	,164	,171	,170	,212	,213	,220	,219
123	,164	,165	,172	,171	,213	,214	,221	,220
124	,165	,166	,173	,172	,214	,215	,222	,221
125	,166	,167	,174	,173	,215	,216	,223	,222
126	,167	,168	,175	,174	,216	,217	,224	,223
127	,169	,170	,177	,176	,218	,219	,226	,225
128	,170	,171	,178	,177	,219	,220	,227	,226
129	,171	,172	,179	,178	,220	,221	,228	,227
130	,172	,173	,180	,179	,221	,222	,229	,228
131	,173	,174	,181	,180	,222	,223	,230	,229
132	,174	,175	,182	,181	,223	,224	,231	,230
133	,176	,177	,184	,183	,225	,226	,233	,232
134	,177	,178	,185	,184	,226	,227	,234	,233
135	,178	,179	,186	,185	,227	,228	,235	,234
136	,179	,180	,187	,186	,228	,229	,236	,235
137	,180	,181	,188	,187	,229	,230	,237	,236
138	,181	,182	,189	,188	,230	,231	,238	,237
139	,183	,184	,191	,190	,232	,233	,240	,239
140	,184	,185	,192	,191	,233	,234	,241	,240
141	,185	,186	,193	,192	,234	,235	,242	,241
142	,186	,187	,194	,193	,235	,236	,243	,242
143	,187	,188	,195	,194	,236	,237	,244	,243
144	,188	,189	,196	,195	,237	,238	,245	,244
145	,197	,198	,205	,204	,246	,247	,254	,253
146	,198	,199	,206	,205	,247	,248	,255	,254
147	,199	,200	,207	,206	,248	,249	,256	,255
148	,200	,201	,208	,207	,249	,250	,257	,256
149	,201	,202	,209	,208	,250	,251	,258	,257
150	,202	,203	,210	,209	,251	,252	,259	,258
151	,204	,205	,212	,211	,253	,254	,261	,260
152	,205	,206	,213	,212	,254	,255	,262	,261
153	,206	,207	,214	,213	,255	,256	,263	,262
154	,207	,208	,215	,214	,256	,257	,264	,263
155	,208	,209	,216	,215	,257	,258	,265	,264
156	,209	,210	,217	,216	,258	,259	,266	,265
157	,211	,212	,219	,218	,260	,261	,268	,267
158	,212	,213	,220	,219	,261	,262	,269	,268
159	,213	,214	,221	,220	,262	,263	,270	,269
160	,214	,215	,222	,221	,263	,264	,271	,270
161	,215	,216	,223	,222	,264	,265	,272	,271
162	,216	,217	,224	,223	,265	,266	,273	,272
163	,218	,219	,226	,225	,267	,268	,275	,274
164	,219	,220	,227	,226	,268	,269	,276	,275
165	,220	,221	,228	,227	,269	,270	,277	,276
166	,221	,222	,229	,228	,270	,271	,278	,277
167	,222	,223	,230	,229	,271	,272	,279	,278
168	,223	,224	,231	,230	,272	,273	,280	,279
169	,225	,226	,233	,232	,274	,275	,282	,281
170	,226	,227	,234	,233	,275	,276	,283	,282
171	,227	,228	,235	,234	,276	,277	,284	,283
172	,228	,229	,236	,235	,277	,278	,285	,284
173	,229	,230	,237	,236	,278	,279	,286	,285
174	,230	,231	,238	,237	,279	,280	,287	,286
175	,232	,233	,240	,239	,281	,282	,289	,288
176	,233	,234	,241	,240	,282	,283	,290	,289
177	,234	,235	,242	,241	,283	,284	,291	,290
178	,235	,236	,243	,242	,284	,285	,292	,291
179	,236	,237	,244	,243	,285	,286	,293	,292
180	,237	,238	,245	,244	,286	,287	,294	,293
181	,246	,247	,254	,253	,295	,296	,303	,302
182	,247	,248	,255	,254	,296	,297	,304	,303
183	,248	,249	,256	,255	,297	,298	,305	,304
184	,249	,250	,257	,256	,298	,299	,306	,305
185	,250	,251	,258	,257	,299	,300	,307	,306
186	,251	,252	,259	,258	,300	,301	,308	,307
187	,253	,254	,261	,260	,302	,303	,310	,309
188	,254	,255	,262	,261	,303	,304	,311	,310
189	,255	,256	,263	,262	,304	,305	,312	,311
190	,256	,257	,264	,263	,305	,306	,313	,312
191	,257	,258	,265	,264	,306	,307	,314	,313
192	,258	,259	,266	,265	,307	,308	,315	,314
193	,260	,261	,268	,267	,309	,310	,317	,316
194	,261	,262	,269	,268	,310	,311	,318	,317
195	,262	,263	,270	,269	,311	,312	,319	,318
196	,263	,264	,271	,270	,312	,313	,320	,319
197	,264	,265	,272	,271	,313	,314	,321	,320
198	,265	,266	,273	,272	,314	,315	,322	,321
199	,267	,268	,275	,274	,316	,317	,324	,323
200	,268	,269	,276	,275	,317	,318	,325	,324
201	,269	,270	,277	,276	,318	,319	,326	,325
202	,270	,271	,278	,277	,319	,320	,327	,326
203	,271	,272	,279	,278	,320	,321	,328	,327
204	,272	,273	,280	,279	,321	,322	,329	,328
205	,274	,275	,282	,281	,323	,324	,331	,330
206	,275	,276	,283	,282	,324	,325	,332	,331
207	,276	,277	,284	,283	,325	,326	,333	,332
208	,277	,278	,285	,284	,326	,327	,334	,333
209	,278	,279	,286	,285	,327	,328	,335	,334
210	,279	,280	,287	,286	,328	,329	,336	,335
211	,281	,282	,289	,288	,330	,331	,338	,337
212	,282	,283	,290	,289	,331	,332	,339	,338
213	,283	,284	,291	,290	,332	,333	,340	,339
214	,284	,285	,292	,291	,333	,334	,341	,340
215	,285	,286	,293	,292	,334	,335	,342	,341
216	,286	,287	,294	,293	,335	,336	,343	,342
*ELSET, elset=Set-All, generate
1,216
*Elset, elset=poly1
8, 14, 44, 45, 50, 51

*Elset, elset=poly2
120, 126

*Elset, elset=poly3
48, 84

*Elset, elset=poly4
185, 186, 190, 191, 192, 197

*Elset, elset=poly5
54, 89, 90

*Elset, elset=poly6
101, 102, 106, 137, 138, 143

*Elset, elset=poly7
17, 18

*Elset, elset=poly8
154, 155, 156, 161, 162

*Elset, elset=poly9
34, 35

*Elset, elset=poly10
23, 29, 64

*Elset, elset=poly11
28

*Elset, elset=poly12
36

*Elset, elset=poly13
98, 103, 104, 133, 134, 135, 139, 140, 141, 169, 176

*Elset, elset=poly14
164, 165, 170, 171

*Elset, elset=poly15
10, 46

*Elset, elset=poly16
152, 158, 187, 188, 194

*Elset, elset=poly17
118, 124

*Elset, elset=poly18
61, 91, 97

*Elset, elset=poly19
111, 112, 146, 147, 148, 182, 183, 184

*Elset, elset=poly20
13, 19, 20, 55, 56

*Elset, elset=poly21
153, 159, 189, 195, 196

*Elset, elset=poly22
41, 42, 47

*Elset, elset=poly23
136, 142, 166, 172, 173, 177, 178, 179

*Elset, elset=poly24
5, 11

*Elset, elset=poly25
174, 180, 209, 210, 215, 216

*Elset, elset=poly26
160

*Elset, elset=poly27
168, 198, 204

*Elset, elset=poly28
25, 31, 32, 67

*Elset, elset=poly29
93, 129, 130

*Elset, elset=poly30
95, 125, 131, 132

*Elset, elset=poly31

*Elset, elset=poly32
82, 88, 123

*Elset, elset=poly33
73, 109, 145

*Elset, elset=poly34
94, 100

*Elset, elset=poly35
75, 81, 117

*Elset, elset=poly36
33, 69, 70, 105

*Elset, elset=poly37
9, 15, 16

*Elset, elset=poly38
52, 53, 59

*Elset, elset=poly39
57, 99

*Elset, elset=poly40
76, 113

*Elset, elset=poly41
79, 80, 85, 86

*Elset, elset=poly42

*Elset, elset=poly43
121, 127, 157, 163

*Elset, elset=poly44
7, 43, 49

*Elset, elset=poly45
208

*Elset, elset=poly46
1, 2, 37, 38, 39, 74

*Elset, elset=poly47
110, 115, 116, 151, 181

*Elset, elset=poly48
96

*Elset, elset=poly49
167, 202, 203

*Elset, elset=poly50
24, 30, 65, 66

*Elset, elset=poly51
71, 72, 107, 108, 144

*Elset, elset=poly52
3, 4, 40

*Elset, elset=poly53
6, 12

*Elset, elset=poly54
114, 119, 149, 150

*Elset, elset=poly55
77, 78

*Elset, elset=poly56
87, 92, 122, 128

*Elset, elset=poly57
60

*Elset, elset=poly58
201, 207

*Elset, elset=poly59
83

*Elset, elset=poly60
193, 199, 200, 205, 206

*Elset, elset=poly61
175, 211, 212

*Elset, elset=poly62
21, 22, 58

*Elset, elset=poly63
26, 27, 62, 63, 68

*Elset, elset=poly64
213, 214


** Section: Section-1
*Solid Section, elset=poly1, material=Material-1-1
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly2, material=Material-1-2
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly3, material=Material-1-3
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly4, material=Material-1-4
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly5, material=Material-1-5
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly6, material=Material-1-6
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly7, material=Material-1-7
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly8, material=Material-1-8
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly9, material=Material-1-9
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly10, material=Material-1-10
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly11, material=Material-1-11
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly12, material=Material-1-12
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly13, material=Material-1-13
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly14, material=Material-1-14
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly15, material=Material-1-15
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly16, material=Material-1-16
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly17, material=Material-1-17
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly18, material=Material-1-18
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly19, material=Material-1-19
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly20, material=Material-1-20
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly21, material=Material-1-21
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly22, material=Material-1-22
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly23, material=Material-1-23
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly24, material=Material-1-24
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly25, material=Material-1-25
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly26, material=Material-1-26
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly27, material=Material-1-27
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly28, material=Material-1-28
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly29, material=Material-1-29
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly30, material=Material-1-30
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly31, material=Material-1-31
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly32, material=Material-1-32
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly33, material=Material-1-33
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly34, material=Material-1-34
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly35, material=Material-1-35
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly36, material=Material-1-36
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly37, material=Material-1-37
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly38, material=Material-1-38
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly39, material=Material-1-39
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly40, material=Material-1-40
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly41, material=Material-1-41
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly42, material=Material-1-42
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly43, material=Material-1-43
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly44, material=Material-1-44
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly45, material=Material-1-45
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly46, material=Material-1-46
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly47, material=Material-1-47
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly48, material=Material-1-48
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly49, material=Material-1-49
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly50, material=Material-1-50
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly51, material=Material-1-51
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly52, material=Material-1-52
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly53, material=Material-1-53
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly54, material=Material-1-54
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly55, material=Material-1-55
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly56, material=Material-1-56
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly57, material=Material-1-57
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly58, material=Material-1-58
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly59, material=Material-1-59
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly60, material=Material-1-60
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly61, material=Material-1-61
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly62, material=Material-1-62
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly63, material=Material-1-63
,
*Hourglass Stiffness
10., , 0., 0.
*Solid Section, elset=poly64, material=Material-1-64
,
*Hourglass Stiffness
10., , 0., 0.
*End Part
**  
**
** ASSEMBLY
**
*Assembly, name=Assembly
**  
*Instance, name=Part-1-1, part=Part-1
*End Instance
*ELSET, elset=Material-1, internal, instance=Part-1-1
8, 14, 44, 45, 50, 51

120, 126

48, 84

185, 186, 190, 191, 192, 197

54, 89, 90

101, 102, 106, 137, 138, 143

17, 18

154, 155, 156, 161, 162

34, 35

23, 29, 64

28

36

98, 103, 104, 133, 134, 135, 139, 140, 141, 169, 176

164, 165, 170, 171

10, 46

152, 158, 187, 188, 194

118, 124

61, 91, 97

111, 112, 146, 147, 148, 182, 183, 184

13, 19, 20, 55, 56

153, 159, 189, 195, 196

41, 42, 47

136, 142, 166, 172, 173, 177, 178, 179

5, 11

174, 180, 209, 210, 215, 216

160

168, 198, 204

25, 31, 32, 67

93, 129, 130

95, 125, 131, 132


82, 88, 123

73, 109, 145

94, 100

75, 81, 117

33, 69, 70, 105

9, 15, 16

52, 53, 59

57, 99

76, 113

79, 80, 85, 86


121, 127, 157, 163

7, 43, 49

208

1, 2, 37, 38, 39, 74

110, 115, 116, 151, 181

96

167, 202, 203

24, 30, 65, 66

71, 72, 107, 108, 144

3, 4, 40

6, 12

114, 119, 149, 150

77, 78

87, 92, 122, 128

60

201, 207

83

193, 199, 200, 205, 206

175, 211, 212

21, 22, 58

26, 27, 62, 63, 68

213, 214


*Nset, nset=refNode, internal, instance=Part-1-1
343
*Nset, nset=x0, internal, instance=Part-1-1

1, 8, 15, 22, 29, 36, 43, 50, 57, 64, 71, 78, 85, 92, 99, 106,

113, 120, 127, 134, 141, 148, 155, 162, 169, 176, 183, 190, 197, 204,

211, 218, 225, 232, 239, 246, 253, 260, 267, 274, 281, 288, 295, 302,

309, 316, 323, 330, 337



*Nset, nset=x1, internal, instance=Part-1-1

7, 14, 21, 28, 35, 42, 49, 56, 63, 70, 77, 84, 91, 98, 105, 112,

119, 126, 133, 140, 147, 154, 161, 168, 175, 182, 189, 196, 203, 210,

217, 224, 231, 238, 245, 252, 259, 266, 273, 280, 287, 294, 301, 308,

315, 322, 329, 336, 343



*Nset, nset=y0, internal, instance=Part-1-1

1, 2, 3, 4, 5, 6, 7, 50, 51, 52, 53, 54, 55, 56, 99, 100,

101, 102, 103, 104, 105, 148, 149, 150, 151, 152, 153, 154, 197, 198,

199, 200, 201, 202, 203, 246, 247, 248, 249, 250, 251, 252, 295, 296,

297, 298, 299, 300, 301



*Nset, nset=y1, internal, instance=Part-1-1 
43, 44, 45, 46, 47, 48, 49, 92, 93, 94, 95, 96, 97, 98, 141, 142, 
143, 144, 145, 146, 147, 190, 191, 192, 193, 194, 195, 196, 239, 240, 
241, 242, 243, 244, 245, 288, 289, 290, 291, 292, 293, 294, 337, 338, 
339, 340, 341, 342, 



*Nset, nset=z0, internal, instance=Part-1-1

1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,

17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32,

33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48,

49



*Nset, nset=z1, internal, instance=Part-1-1 
295, 296, 297, 298, 299, 300, 301, 302, 303, 304, 305, 306, 307, 308, 
309, 310, 311, 312, 313, 314, 315, 316, 317, 318, 319, 320, 321, 322, 
323, 324, 325, 326, 327, 328, 329, 330, 331, 332, 333, 334, 335, 336, 
337, 338, 339, 340, 341, 342, 




** Constraint: Constraint-1
*Equation
2
y1, 2, 1.
refNode, 2, -1.
** 
** Constraint: Constraint-2
*Equation
2
z1, 3, 1.
refNode, 3, -1.
** 
*End Assembly 
** 
** MATERIALS
** \n""")
    # Loop throgh materials and texture
    for j in range(numElsets):
        f.write('*Material, name=Material-1-' + str(j+1) + '\n')
        f.write('*DEPVAR \n')
        f.write(str(DEPVAR)+', \n')
        f.write('*User Material, constants=33, unsymm \n')
        f.write(str(input_props[0])+', ')            #C11A
        f.write(str(input_props[1])+', ')            #C12A
        f.write(str(input_props[2])+', ')            #C44A
        f.write(str(input_props[3])+', ')            #C11M
        f.write(str(input_props[4])+', ')            #C12M
        f.write(str(input_props[5])+', ')            #C44M
        f.write(str(input_props[6])+', ')            #alpha_A
        f.write(str(input_props[7])+'\n')            #alpha_M
        f.write(str(input_props[8])+', ')            #f_c
        f.write(str(input_props[9])+', ')            #gamma_0
        f.write(str(input_props[10])+', ')           #theta_t
        f.write(str(input_props[11])+', ')           #theta_ref_low
        f.write(str(input_props[12])+', ')           #lambda_t
        f.write(str(input_props[13])+', ')           #C11A
        f.write(str(input_props[14])+', ')           #C12A
        f.write(str(input_props[15])+'\n')           #C44A
        f.write(str(input_props[16])+', ')           #gamma_dot_0
        f.write(str(input_props[17])+', ')           #A_m
        f.write(str(input_props[18])+', ')           #S0
        f.write(str(input_props[19])+', ')           #H0
        f.write(str(input_props[20])+', ')           #SS
        f.write(str(input_props[21])+', ')           #A_hard
        f.write(str(input_props[22])+', ')           #QL
        f.write(str(input_props[23])+'\n ')          #theta_ref_high
        # Putting in euler angles (texture info)
        f.write(str(orient[j,0]) + ',' + \
                         str(orient[j,1]) + ',' + \
                         str(orient[j,2]) + ',')     #Euler Angles (1,2,3)
        f.write(str(input_props[24])+', ')           #B_sat
        f.write(str(input_props[25])+', ')           #b1
        f.write(str(input_props[26])+', ')           #N_GEN_F
        f.write(str(input_props[27])+', ')           #d_ir
        f.write(str(input_props[28])+'\n')           #Dir
        f.write(str(input_props[29])+', \n')         #mu
    f.write("""** ----------------------------------------------
** 
** STEP: Step-1
** 
*Step, name=Step-1, nlgeom=YES, inc=500000 
*Static
0.001, 1., 1e-06, 0.01
** 
** 
** 
** BOUNDARY CONDITIONS
** 
*BOUNDARY, type=DISPLACEMENT                          
x0, 1, 1 ,0.0 
*BOUNDARY, type=DISPLACEMENT 
x1, 1, 1 ,0.05
*BOUNDARY, type=DISPLACEMENT                          
y0, 2, 2 ,0.0 
*BOUNDARY, type=DISPLACEMENT                          
z0, 3, 3 ,0.0  
** 
** CONTROLS
** 
*Controls, reset
*Controls, parameters=field, field=displacement
0.9, 0.9, , , 0.9, , ,   
*Controls, parameters=field, field=hydrostatic fluid pressure
0.9, 0.9, , , 0.9, , , 
*Controls, parameters=field, field=rotation
0.9, 0.9, , , 0.9, , , 
*Controls, parameters=field, field=electrical potential
0.9, 0.9, , , 0.9, , , 
** 
** SOLVER CONTROLS
** 
*Solver Controls, reset
*Solver Controls
  0.8,   
*** 
** 
** OUTPUT REQUESTS
** 
*Restart, write, number interval=1, time marks=NO
** 
** FIELD OUTPUT: F-Output-1
** 
*Output, field, variable=PRESELECT, FREQUENCY= 1
*Element Output
EVOL, 
SDV1, SDV2, SDV3, SDV4, SDV5, 
SDV6, SDV7, SDV8, SDV9, SDV10, 
SDV11, SDV12, SDV13, SDV14, SDV15, 
SDV16, SDV17, SDV18, SDV19, SDV20, 
SDV21, SDV22, SDV23, SDV24, SDV25, 
SDV26, SDV27, SDV28, SDV29, SDV30, 
SDV31, SDV32, SDV33, SDV34, SDV35, 
SDV36, SDV37, SDV38, SDV39, SDV40, 
SDV41, SDV42, SDV43, SDV44, SDV45, 
SDV46, SDV47, SDV48, SDV49, SDV50, 
SDV51, SDV52, SDV53, SDV54, SDV55, 
SDV56, SDV57, SDV58, SDV59, SDV60, 
SDV61, SDV62, SDV63, SDV64, SDV65, 
SDV66, SDV67, SDV68, SDV69, SDV70, 
SDV71, SDV72, SDV73, SDV74, SDV75, 
SDV76, SDV77, SDV78, SDV79, SDV80, 
SDV81, 
SDV101, SDV102, SDV103, SDV104, SDV105, 
SDV106, SDV107, SDV108, SDV109, SDV110, 
SDV111, SDV112, SDV113, SDV114, SDV115, 
SDV116, SDV117, SDV118, SDV119, SDV120, 
SDV121, SDV122, SDV123, SDV124, SDV125, 
SDV126, SDV127, SDV128, SDV129, SDV130, 
SDV131, SDV132, SDV133, 
SDV152, SDV153, SDV154, SDV155, 
SDV156, SDV157, SDV158, SDV159, SDV160, 
SDV161, SDV162, SDV163, SDV164, SDV165, 
SDV166, SDV167, SDV168, SDV169, SDV170, 
SDV171, SDV172, SDV173, SDV174, SDV175, 
SDV176, SDV177, SDV178, SDV179, SDV180, 
SDV181, SDV182, SDV183, SDV184, SDV185, 
SDV186, SDV187, SDV188, SDV189, SDV190, 
SDV191, SDV192, SDV193, SDV194, SDV195, 
SDV196, SDV197, SDV198, SDV199, SDV200, 
SDV201, SDV202, SDV203, SDV204, SDV205, 
SDV206, SDV207, SDV208, SDV209, SDV210, 
SDV211, SDV212, SDV213, SDV214, SDV215, 
SDV216, SDV217, SDV218, SDV219, SDV220, 
SDV221, SDV222, SDV223, SDV224, SDV225, 
SDV226, SDV227, SDV228, SDV229, SDV230, 
SDV231, SDV232, SDV233, SDV234
*End Step 
** 
** STEP: Step-2
** 
*Step, name=Step-2, nlgeom=YES, inc=500000 
*Static
0.001, 1., 1e-06, 0.01
** 
** 
** 
** BOUNDARY CONDITIONS
** 
*BOUNDARY, type=DISPLACEMENT                          
x0, 1, 1 ,0.0 
*BOUNDARY, type=DISPLACEMENT 
x1, 1, 1 ,0.0024
*BOUNDARY, type=DISPLACEMENT                          
y0, 2, 2 ,0.0 
*BOUNDARY, type=DISPLACEMENT                          
z0, 3, 3 ,0.0  
** 
** CONTROLS
** 
*Controls, reset
*Controls, parameters=field, field=displacement
0.9, 0.9, , , 0.9, , ,   
*Controls, parameters=field, field=hydrostatic fluid pressure
0.9, 0.9, , , 0.9, , , 
*Controls, parameters=field, field=rotation
0.9, 0.9, , , 0.9, , , 
*Controls, parameters=field, field=electrical potential
0.9, 0.9, , , 0.9, , , 
** 
** SOLVER CONTROLS
** 
*Solver Controls, reset
*Solver Controls
  0.8,   
*** 
** 
** OUTPUT REQUESTS
** 
*Restart, write, number interval=1, time marks=NO
** 
** FIELD OUTPUT: F-Output-1
** 
*Output, field, variable=PRESELECT, FREQUENCY= 1
*Element Output
EVOL, 
SDV1, SDV2, SDV3, SDV4, SDV5, 
SDV6, SDV7, SDV8, SDV9, SDV10, 
SDV11, SDV12, SDV13, SDV14, SDV15, 
SDV16, SDV17, SDV18, SDV19, SDV20, 
SDV21, SDV22, SDV23, SDV24, SDV25, 
SDV26, SDV27, SDV28, SDV29, SDV30, 
SDV31, SDV32, SDV33, SDV34, SDV35, 
SDV36, SDV37, SDV38, SDV39, SDV40, 
SDV41, SDV42, SDV43, SDV44, SDV45, 
SDV46, SDV47, SDV48, SDV49, SDV50, 
SDV51, SDV52, SDV53, SDV54, SDV55, 
SDV56, SDV57, SDV58, SDV59, SDV60, 
SDV61, SDV62, SDV63, SDV64, SDV65, 
SDV66, SDV67, SDV68, SDV69, SDV70, 
SDV71, SDV72, SDV73, SDV74, SDV75, 
SDV76, SDV77, SDV78, SDV79, SDV80, 
SDV81, 
SDV101, SDV102, SDV103, SDV104, SDV105, 
SDV106, SDV107, SDV108, SDV109, SDV110, 
SDV111, SDV112, SDV113, SDV114, SDV115, 
SDV116, SDV117, SDV118, SDV119, SDV120, 
SDV121, SDV122, SDV123, SDV124, SDV125, 
SDV126, SDV127, SDV128, SDV129, SDV130, 
SDV131, SDV132, SDV133, 
SDV152, SDV153, SDV154, SDV155, 
SDV156, SDV157, SDV158, SDV159, SDV160, 
SDV161, SDV162, SDV163, SDV164, SDV165, 
SDV166, SDV167, SDV168, SDV169, SDV170, 
SDV171, SDV172, SDV173, SDV174, SDV175, 
SDV176, SDV177, SDV178, SDV179, SDV180, 
SDV181, SDV182, SDV183, SDV184, SDV185, 
SDV186, SDV187, SDV188, SDV189, SDV190, 
SDV191, SDV192, SDV193, SDV194, SDV195, 
SDV196, SDV197, SDV198, SDV199, SDV200, 
SDV201, SDV202, SDV203, SDV204, SDV205, 
SDV206, SDV207, SDV208, SDV209, SDV210, 
SDV211, SDV212, SDV213, SDV214, SDV215, 
SDV216, SDV217, SDV218, SDV219, SDV220, 
SDV221, SDV222, SDV223, SDV224, SDV225, 
SDV226, SDV227, SDV228, SDV229, SDV230, 
SDV231, SDV232, SDV233, SDV234
*End Step""")
    f.close()


#%% Time when code ended
print('New input file(s) made and ready to run')
print('Time code ended: ',(time.ctime()))
