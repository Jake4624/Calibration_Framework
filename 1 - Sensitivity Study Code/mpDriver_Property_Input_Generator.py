# 4 January 2023
# Jacob Rusch
#
# This script generates a text file which contains the property data for the 
# material point driver simulations.
#

import numpy as np
import random

# Range of values for each input. Some inputs will remain constant such as
# the elastic constants and thermal coefficients. The plasticity, hardening,
# ratcheting parameters, and Euler angles can all change.
#
# NOTE: The parameters listed in below are not in the same order that the 
# user material reads them in. They will be rearranged and put in the proper
# order when the input file is made.

# Inputs are as follows:
# Name          Value       Units
# -----         -----       -----
# Elastic Constants
C11A = np.array([130000])         # MPa
C12A = np.array([98000])          # MPa
C44A = np.array([34000])          # MPa
C11M = np.array([130000])         # MPa
C12M = np.array([98000])          # MPa
C44M = np.array([34000])          # MPa
# Thermal Expansion Coefficients
alpha_A = np.array([1.1E-05])     # /K
alpha_M = np.array([6.6E-06])     # /K
# Transformation strains for each of the 24 transformation systems. Assumed
# to be the same for all 24 systems
gamma_0 = np.array([0.1308])      # -
# Reference rate of shear
gamma_dot_0 = np.array([0.002])   # /s
# Ratio of self to latent hardening
# This value will not change.It is taken as 1.0 for coplanar slip systems 
# alpha and beta and 1.4 otherwise which renders the hardening model 
# anisotropic. Taken from:
# F. Roters, P. Eisenlohr, L. Hantcherli, D. D. Tjahjanto, T. R. Bieler, and 
# D. Raabe, "Overview of constitutive laws, kinematics, homogenization and 
# multiscale methods in crystal plasticity finite-element modeling: Theory, 
# experiments, applications," Acta Mater., vol. 58, no. 4, pp. 1152-1211, 
# 2010, doi: 10.1016/j.actamat.2009.10.058.
QL = np.array([1.4])               # -
##############################################################################
# Values listed below will change
# Critical energy barrier per unit volume for transformation
f_c = np.array([4,10])                # MPa
# Equilibrium transformation temperature
Theta_T = np.array([253,383])         # K
Theta_Ref_Low = np.array([253,383])   # K
Theta_Ref_High = np.array([253,383])  # K
# Latent heat of transformation per unit volume.
# Note that latent heat is usually presented in (J/g) or (kJ/kg) or similar 
# because it is presented as a specific latent heat. To get to a volumetric 
# latent heat, simply multiply by the density of NiTi (~6.45 g/cm^3).
lambda_T = np.array([110,250])        # MJ/m^3
# Strain rate exponent
Am =            np.array([0.01,0.05]) # -
# Initial slip system hardness
S0 = np.array([300,800])              # MPa
# Self-hardening coefficient 
H0 = np.array([300,1500])             # MPa
# Saturation hardness
SS = np.array([500,1500])             # MPa
# Hardening exponent
Ahard = np.array([0.166667,0.1])      # -
# Euler angles for texture are made in a separate file called
# writeTexture.py
#
# Transformation ratcheting parameters
# Saturated value of internal stress
Bsat = np.array([50,500])             # MPa
# Governs the evolution rate of internal stress during the cyclic phase 
# transformation
b1 = np.array([1,5])                  # -
# A non-negative material paramter and can be determined from the 
# corresponding experimental results. Represents the dependence of 
# transformation-induced plasticity and residual martensite on the 
# applied stress.
N_GEN_F = np.array([3,7])             # -
# Material parameter
d_ir = np.array([0.0001,0.01])        # -
# Reference resistance
D_ir = np.array([0.0001,0.01])        # -
# A material conostant describing the different evolution features of 
# interface friction slip occurred in forward martensite transformation and 
# its reverse.
mu = np.array([0.1,1.0])              # -

fileOutput = open('Properties_mpDriver.inc','w')
#
# For loop where i goes from 1 to number of elements or number of simulations
# mpDriver will run
#
for b in range(len(f_c)):
    for c in range(len(Theta_T)):
        for d in range(len(Theta_Ref_Low)):
            for e in range(len(lambda_T)):
                for f in range(len(Am)):
                    for g in range(len(S0)):
                        for h in range(len(H0)):
                            for i in range(len(SS)):
                                for j in range(len(Ahard)):
                                    for k in range(len(Theta_Ref_High)):
                                        for l in range(len(Bsat)):
                                            for m in range(len(b1)):
                                                for n in range(len(N_GEN_F)):
                                                    for o in range(len(d_ir)):
                                                        for p in range(len(D_ir)):
                                                            for q in range(len(mu)):
                                                                # Write property value to file              property number in user material
                                                                # ---------------------------               ---------------------------
                                                                fileOutput.write(str(C11A[0])+',')             # props(1)     Constant
                                                                fileOutput.write(str(C12A[0])+',')             # props(2)     Constant
                                                                fileOutput.write(str(C44A[0])+',')             # props(3)     Constant
                                                                fileOutput.write(str(C11M[0])+',')             # props(4)     Constant
                                                                fileOutput.write(str(C12M[0])+',')             # props(5)     Constant
                                                                fileOutput.write(str(C44M[0])+',')             # props(6)     Constant
                                                                fileOutput.write(str(alpha_A[0])+',')          # props(7)     Constant
                                                                fileOutput.write(str(alpha_M[0])+',')          # props(8)     Constant
                                                                fileOutput.write(str(f_c[b])+',')              # props(9)
                                                                fileOutput.write(str(gamma_0[0])+',')          # props(10)    Constant
                                                                fileOutput.write(str(Theta_T[c])+',')          # props(11)
                                                                fileOutput.write(str(Theta_Ref_Low[d])+',')    # props(12)
                                                                fileOutput.write(str(lambda_T[e])+',')         # props(13)
                                                                fileOutput.write(str(C11A[0])+',')             # props(14)    Constant
                                                                fileOutput.write(str(C12A[0])+',')             # props(15)    Constant
                                                                fileOutput.write(str(C44A[0])+',')             # props(16)    Constant
                                                                fileOutput.write(str(gamma_dot_0[0])+',')      # props(17)    Constant
                                                                fileOutput.write(str(Am[f])+',')               # props(18)
                                                                fileOutput.write(str(S0[g])+',')               # props(19)
                                                                fileOutput.write(str(H0[h])+',')               # props(20)
                                                                fileOutput.write(str(SS[i])+',')               # props(21)
                                                                fileOutput.write(str(Ahard[j])+',')            # props(22)
                                                                fileOutput.write(str(QL[0])+',')               # props(23)    Constant
                                                                fileOutput.write(str(Theta_Ref_High[k])+',')   # props(24)
                                                                # Euler 1                                      # props(25)    From texture file
                                                                # Euler 2                                      # props(26)    From texture file
                                                                # Euler 3                                      # props(27)    From texture file
                                                                fileOutput.write(str(Bsat[l])+',')             # props(28)
                                                                fileOutput.write(str(b1[m])+',')               # props(29)
                                                                fileOutput.write(str(N_GEN_F[n])+',')          # props(30)
                                                                fileOutput.write(str(d_ir[o])+',')             # props(31)
                                                                fileOutput.write(str(D_ir[p])+',')             # props(32)
                                                                fileOutput.write(str(mu[q]))                   # props(33)
                                                                
                                                                fileOutput.write('\n')

fileOutput.close()
