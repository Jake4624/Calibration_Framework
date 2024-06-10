#!/usr/bin/python
#
# Script is from John A. Moore, PhD at Marquette University
# Used by Jacob Rusch for research pertaining to PhD dissertation
# 5 May 2023
#
# This generates texture with quaternions, need to use Matlab 
# firstPoleFigure.m script with the MTEX toolbox to convert 
# them to Euler angles and make pole figures. 
#
# SEE: mtex-toolbox.github.io for info on MTEX and how to
# download and install.
#
# Will also need one of the following other toolboxes installed
# in order to use the commands pertaining to quaternions
#  in the firstPoleFigure.m script
# - Navigation Toolbox
# - Robotics System Toolbox
# - UAV Toolbox
# If you try to run the firstPoleFigure.m without one of the
# three toolboxes listed above it will give an error with
# links to download them.

import numpy as np
import random

#type = 'euler'
type = 'quaternion'


fileName = 'quaternion_texture.txt'
f = open(fileName,'w')

numOrient = 64
pi = np.pi

orient1 = [0.5000,   -0.5000,    0.5000,    0.5000]
orient2 = [0.0000,         0,    0.7071,    0.7071]

# strength (0-1) of texture in orient1 and orient2
# used used 80pct for all the original runs
strength = 0

# spread in orientation
sig = 0.1

count1 = 0.0
count2 = 0.0
for i in range(numOrient):
    if type == 'euler':
        a = 2.0*pi*random.random()
        b = 2.0*pi*random.random()
        c = 2.0*pi*random.random()
        f.write(str(a) + ',' +  str(b) + ',' + str(c) + '\n')
    elif type == 'quaternion':
        # number to determine which orentation to use
        rand1 = random.random()
        if rand1 >= strength:
            q0 = random.random()
            q1 = random.random()
            q2 = random.random()
            q3 = random.random()
        elif rand1 >=  strength/2.0:
            # use first orientation + perterbation
            q0 = orient1[0] +  np.asscalar(np.random.normal(0.0,sig,1))
            q1 = orient1[1] +  np.asscalar(np.random.normal(0.0,sig,1))
            q2 = orient1[2] +  np.asscalar(np.random.normal(0.0,sig,1))
            q3 = orient1[3] +  np.asscalar(np.random.normal(0.0,sig,1))
            count1 = count1 + 1
        else:
            # second oririentation + perterbation
            q0 = orient2[0] +  np.asscalar(np.random.normal(0.0,sig,1))
            q1 = orient2[1] +  np.asscalar(np.random.normal(0.0,sig,1))
            q2 = orient2[2] +  np.asscalar(np.random.normal(0.0,sig,1))
            q3 = orient2[3] +  np.asscalar(np.random.normal(0.0,sig,1))
            count2 = count2 + 1

        qmag = np.sqrt(q0**2. + q1**2. +  q2**2. + q3**2.)
        q0 = q0/qmag
        q1 = q1/qmag
        q2 = q2/qmag
        q3 = q3/qmag       
        f.write(str(q0) + ',' +  str(q1) + ',' + str(q2) +  ',' + str(q3) + '\n')
print('orient1 ' + str(count1/numOrient))
print('orient2 ' + str(count2/numOrient))
print('actual strength ' + str((count1+count2)/numOrient)) 
f.close()
