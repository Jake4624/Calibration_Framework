# THIS CODE NEEDS TO BE UPDATED!
# THE DISPLACEMENT VALUES WERE MANUALLY ADJUSTED IN THE INPUT FILE
# TO MATCH THE EXPERIMENTAL ONES AFTER THIS SCRIPT WAS RUN.
# WILL NEED TO MAKE AN ARRAY CONTAINING THE STARTING/ENDING
# DISPLACEMENT VALUES BASED ON EXPERIMENTAL DATA!
#
# Code provided by John A Moore, PhD
# Modified/commented by Jacob Rusch
# 17 July 2023
#
# For Objective 2: Uses static direct to get DFGRD1 Values for the
# low fidelity model
#
# This script generates an Abaqus input file deck.
#
# The main code input is number of seed elements along the edge of the cube.
# Other inputs such as the step parameters, state variables to record, 
# material parameters, and others are below and can also be adjusted. Most
# values before the "#%% Writing Abaqus Input File" comment are able to be
# adjusted.
#
# To generate a 64 element cube with each element being a different grain and
# has a unique crystallographic orientation one would enter: numSeedElem = 4
# 

#!/usr/bin/python
import numpy as np

# Orientation file. Make sure this is consistent with the number of grains. 
# Need to have the same number of lines in the texture file as there are 
# grains. Euler angles for texture are made in a separate file called
# writeTexture.py
orientFile = 'texture.inc'
orient = np.loadtxt(orientFile,delimiter=',')

# File containing properties for each simulation
#propFile = 'Properties.txt'
#mat_props = np.loadtxt(propFile,delimiter=',')

#number of seed elements along one edge of cube
numSeedElem = 4

# Number of elements and nodes in cube
# DO NOT CHANGE OR ADJUST THESE EQUATIONS: numElsets and numNodes
numElsets = numSeedElem**3
numNodes = (numSeedElem+1)**3

# Name of Abaqus input file
fileName = str(numElsets)+'_Element_Cube'

## Create Output file
fileOutput = open(fileName + '.inp','w') 

# element type (either C3D8R or C3D8)
elementType = 'C3D8R'

# Number of state variables
DEPVAR = 234

# statevaribales start and end values
sdvStartEnd = [[1,81],[101,133],[152,234]]

# number of steps. Each step is 1 second and is either loading or unloading 
# E.g., numStep = 2 --> 1 cycle (step 1 is loading, step 2 is unloading) 
numStep = 10

# This does a ramped displacement having a cyclic function as follows:
# disp = dispMean +/- 0.5*dispAmp
# Can also just comment out some of the lines and manually input the 
# dispMax and dispMin
#
# mean strain/displacemnt

# Uncomment the following lines if it is desired to use equation, otherwise
# manually endter dispMax/Min below.
#dispMean = 0.01
#
#dispAmp = 0.01
#dispMax = dispMean + 0.5*dispAmp
#dispMin = dispMean - 0.5*dispAmp

dispMax = 0.05
dispMin = 0.01

# Parameters for steps. nlgeom can either be YES or NO, but should be set to 
# YES for these simulations. inc is the maximum number of increments per step
nlgeom='YES'
inc=100000

# Automatic time step parameters
Initial_Time_Step = 0.01
Total_Increment_Time = 1
Max_Step_Size = 0.01
Min_Step_Size = 1E-06

#output frequency
freq1 = 1 # first cycle
freq2 = 1 # subsiquent cycles

# Materials Name
MaterialName = 'NITINOL_CP_MODEL'

#%% Material Parameters (inputs to User Material)
# NOTE: The parameters listed in below are not in the same order that the 
# user material reads them in. They will be rearranged and put in the proper
# order when the input file is made.

# Inputs are as follows:
# Name   Value          Units
# -----  -----          -----
# Elastic Constants
C11A = 130000           # MPa
C12A = 98000            # MPa
C44A = 34000            # MPa
C11M = 130000           # MPa
C12M = 98000            # MPa
C44M = 34000            # MPa
# Thermal Expansion Coefficients
alpha_A = 1.1E-05       # /K
alpha_M = 6.6E-06       # /K
# Critical energy barrier per unit volume for transformation
f_c = 8.4               # MPa
# Transformation strains for each of the 24 transformation systems. Assumed
# to be the same for all 24 systems
gamma_0 = 0.1308        # -
# Equilibrium transformation temperature
Theta_T = 257           # K
# Theta_Ref_Low does ot come into play unless there are many many cycles.
# It can be any number and have no effect on most simulations. See UMAT for
# where it is used to understand what it does.
Theta_Ref_Low = 277     # K
# Theta_Ref_High is important and does have an impact on response unlike
# Theta_ref_low
Theta_Ref_High = 298    # K
# Latent heat of transformation per unit volume.
# Note that latent heat is usually presented in (J/g) or (kJ/kg) or similar 
# because it is presented as a specific latent heat. To get to a volumetric 
# latent heat, simply multiply by the density of NiTi (~6.45 g/cm^3).
lambda_T = 130          # MJ/m^3
# Reference rate of shear
gamma_dot_0 = 0.002     # /s
# Strain rate exponent
Am = 0.02               # -
# Initial slip system hardness
S0 = 350                # MPa
# Self-hardening coefficient 
H0 = 500               # MPa
# Saturation hardness
SS = 900                # MPa
# Hardening exponent
Ahard = 0.125           # -
# Ratio of self to latent hardening
# This value will not change. It is taken as 1.0 for coplanar slip systems 
# alpha and beta and 1.4 otherwise which renders the hardening model 
# anisotropic.
QL = 1.4                # -
#
# Euler angles for texture are made in a separate file called
# writeTexture.py
#
# Saturated value of internal stress
Bsat = 150              # MPa
# Governs the evolution rate of internal stress during the cyclic phase 
# transformation
b1 = 2                  # -
# A non-negative material paramter and can be determined from the 
# corresponding experimental results. Represents the dependence of 
# transformation-induced plasticity and residual martensite on the 
# applied stress.
N_GEN_F = 8            # -
# Material parameter
d_ir = 0.001            # -
# Reference resistance
Dir = 0.1             # -
# A material conostant describing the different evolution features of 
# interface friction slip occurred in forward martensite transformation and 
# its reverse.
mu = 0.2                # -
#%% Writing Abaqus Input File
# DO NOT ADJUST THE CODE BELOW THIS LINE
fileOutput.write("""**Input file for a cube of unit dimension in all directions
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
**
*Heading
*Preprint, echo=NO, model=NO, history=NO, contact=NO
**
** PARTS
**
*Part, name=Part-1
*Node
""")
#%% Part
# Assign coordinates to nodes. Simple nested for loop is sufficient for this
x_node_coords = np.round(np.linspace(0,1,(numSeedElem+1)),12)
y_node_coords = np.round(np.linspace(0,1,(numSeedElem+1)),12)
z_node_coords = np.round(np.linspace(0,1,(numSeedElem+1)),12)

nodecounter = 1
for z_coord_count in range(len(z_node_coords)):
    for y_coord_count in range(len(y_node_coords)):
        for x_coord_count in range(len(x_node_coords)):
            fileOutput.write(str(nodecounter)+', '+\
                             str(x_node_coords[x_coord_count])+', '+\
                             str(y_node_coords[y_coord_count])+', '+\
                             str(z_node_coords[z_coord_count])+ '\n')
            nodecounter=nodecounter+1
    
fileOutput.write('*Element, type=' + elementType +' \n')

# Initial pattern for nodes assigned to each element
#
# Node values increase by 1 for each row for the number of seed elements along
# one edge of the cube, then the numbers skip a value. This pattern repeats
# until all elements have been assigned nodes.
#
# Make initial array of NODE VALUES ONLY
node_vals = np.array([1, 2, (numSeedElem+3), (numSeedElem+2), \
                      (((numSeedElem+1)**2)+1), \
                      ((numSeedElem+1)**2)+2, \
                      (((numSeedElem+1)**2)+3)+numSeedElem, \
                      (((numSeedElem+1)**2)+2)+numSeedElem])
elementcounter = 1
new_jump = 1
for i in range(1,numElsets+1):
    if i % numSeedElem == 0 and i != 0 and i%numSeedElem**2 != 0:
        fileOutput.write(str(elementcounter)+', '+\
                         str(node_vals[0])+', '+\
                         str(node_vals[1])+', '+\
                         str(node_vals[2])+', '+\
                         str(node_vals[3])+', '+\
                         str(node_vals[4])+', '+\
                         str(node_vals[5])+', '+\
                         str(node_vals[6])+', '+\
                         str(node_vals[7])+'\n')
        node_vals = node_vals + 2
        elementcounter=elementcounter+1
        new_jump = new_jump + 1
    elif i % numSeedElem**2 == 0:
        fileOutput.write(str(elementcounter)+', '+\
                         str(node_vals[0])+', '+\
                         str(node_vals[1])+', '+\
                         str(node_vals[2])+', '+\
                         str(node_vals[3])+', '+\
                         str(node_vals[4])+', '+\
                         str(node_vals[5])+', '+\
                         str(node_vals[6])+', '+\
                         str(node_vals[7])+'\n')
        node_vals = node_vals + numSeedElem + 3
        elementcounter=elementcounter+1
    else:
        fileOutput.write(str(elementcounter)+', '+\
                         str(node_vals[0])+', '+\
                         str(node_vals[1])+', '+\
                         str(node_vals[2])+', '+\
                         str(node_vals[3])+', '+\
                         str(node_vals[4])+', '+\
                         str(node_vals[5])+', '+\
                         str(node_vals[6])+', '+\
                         str(node_vals[7])+'\n')
        node_vals = node_vals + 1
        elementcounter=elementcounter+1
        
# GENERATE NSET=ALL
fileOutput.write('*Nset, nset=ALL, generate \n')
fileOutput.write(' 1, '+str(numNodes)+', 1 \n')
# DIFFERENT ELSET FOR EACH ELEMENT CORRESPONDING TO UNIQUE GRAIN
fileOutput.write('*Elset, elset=ALL, generate \n')
fileOutput.write(' 1, '+str(numElsets)+', 1 \n')

for elnum in range(numElsets):
    fileOutput.write('*Elset, elset=poly'+ str(elnum+1) + '\n')
    fileOutput.write(str(elnum+1) + ', \n')

# Section (Need to put elemet sets in part, and remove instance for name)

for i in range(numElsets):
    fileOutput.write('** Section: Section-'+str(i+1) + '\n')
    fileOutput.write('*Solid Section, elset=poly' + str(i+1) +\
                     ', controls=EC-1, material=' + MaterialName  + '-' + \
                     str(i+1) + '\n')
    fileOutput.write(',' + '\n')

fileOutput.write('*End Part')
#%% ASSEMBLY
fileOutput.write("""
**  
**
** ASSEMBLY
**
*Assembly, name=Assembly
**  
*Instance, name=Part-1-1, part=Part-1
*End Instance
**""")
fileOutput.write('\n')

#%% Generating nsets of each face of cube

fileOutput.write('*Nset, nset=refNode, internal, instance=Part-1-1 \n')
fileOutput.write(str(numNodes)+'\n')

fileOutput.write('*Nset, nset=x0, internal, instance=Part-1-1 \n')
x0 = np.linspace(1,1+(numSeedElem+1)*(((numSeedElem+1)**2)-1),\
                 ((numSeedElem+1)**2))
for i in range(len(x0)):
    if (i+1) % 16 == 0 and i != len(x0):
        fileOutput.write(str(int(x0[i]))+', \n')
    elif i == len(x0)-1:
        fileOutput.write(str(int(x0[i]))+' \n')
    else:
        fileOutput.write(str(int(x0[i]))+', ')

fileOutput.write('*Nset, nset=x1, internal, instance=Part-1-1 \n')
x1 = np.linspace((numSeedElem+1),(numSeedElem+1)**3,\
                 ((numSeedElem+1)**2))
for i in range(len(x1)):
    if (i+1) % 16 == 0 and i != len(x1):
        fileOutput.write(str(int(x1[i]))+', \n')
    elif i == len(x1)-1:
        fileOutput.write(str(int(x1[i]))+' \n')
    else:
        fileOutput.write(str(int(x1[i]))+', ')

fileOutput.write('*Nset, nset=y0, internal, instance=Part-1-1 \n')
y0 = np.linspace(1,numSeedElem+1,numSeedElem+1)
for i in range(1,numSeedElem+1):
    y0_new = np.linspace(i*((numSeedElem+1)**2)+1,\
                         i*((numSeedElem+1)**2)+numSeedElem+1,\
                         numSeedElem+1)
    y0 = np.concatenate((y0,y0_new))

for i in range(len(y0)):
    if (i+1) % 16 == 0 and i != len(y0):
        fileOutput.write(str(int(y0[i]))+', \n')
    elif i == len(y0)-1:
        fileOutput.write(str(int(y0[i]))+' \n')
    else:
        fileOutput.write(str(int(y0[i]))+', ')

fileOutput.write('*Nset, nset=y1, internal, instance=Part-1-1 \n')
y1 = np.linspace(((numSeedElem+1)**2)-numSeedElem,\
                 ((numSeedElem+1)**2),\
                 numSeedElem+1)
for i in range(2,numSeedElem+2):
    y1_new = np.linspace(i*((numSeedElem+1)**2)-numSeedElem,\
                         i*((numSeedElem+1)**2),\
                         numSeedElem+1)
    y1 = np.concatenate((y1,y1_new))
y1 = np.delete(y1, -1)
for i in range(len(y1)):
    if (i+1) % 16 == 0 and i != len(y1):
        fileOutput.write(str(int(y1[i]))+', \n')
    elif i == len(y1)-1:
        fileOutput.write(str(int(y1[i]))+' \n')
    else:
        fileOutput.write(str(int(y1[i]))+', ')

fileOutput.write('*Nset, nset=z0, internal, instance=Part-1-1 \n')
z0 = np.linspace(1,(numSeedElem+1)**2,\
                 (numSeedElem+1)**2)
for i in range(len(z0)):
    if (i+1) % 16 == 0 and i != len(z0):
        fileOutput.write(str(int(z0[i]))+', \n')
    elif i == len(z0)-1:
        fileOutput.write(str(int(z0[i]))+' \n')
    else:
        fileOutput.write(str(int(z0[i]))+', ')


fileOutput.write('*Nset, nset=z1, internal, instance=Part-1-1 \n')
z1 = np.linspace((numSeedElem+1)**3-(numSeedElem+1)**2+1,\
                 (numSeedElem+1)*((numSeedElem+1)**2)-1,\
                 (numSeedElem+1)**2-1)
for i in range(len(z1)):
    if (i+1) % 16 == 0 and i != len(z1):
        fileOutput.write(str(int(z1[i]))+', \n')
    elif i == len(z1)-1:
        fileOutput.write(str(int(z1[i]))+' \n')
    else:
        fileOutput.write(str(int(z1[i]))+', ')

################ Equations ###################
fileOutput.write('''** Constraint: Constraint-1
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
** ELEMENT CONTROLS
** 
*Section Controls, name=EC-1, hourglass=ENHANCED
10., , 0., 0.
**
** \n''')
#%% Materials
fileOutput.write("""** 
** MATERIALS
** 
""")
for i in range(numElsets):
    fileOutput.write('*Material, name='+ MaterialName + '-' + str(i+1) + '\n')
    fileOutput.write('*DEPVAR \n')
    fileOutput.write(str(DEPVAR)+', \n')
    fileOutput.write('*User Material, constants=33, unsymm \n')
    fileOutput.write(str(C11A)+', ')            #1
    fileOutput.write(str(C12A)+', ')            #2
    fileOutput.write(str(C44A)+', ')            #3
    fileOutput.write(str(C11M)+', ')            #4
    fileOutput.write(str(C12M)+', ')            #5
    fileOutput.write(str(C44M)+', ')            #6
    fileOutput.write(str(alpha_A)+', ')         #7
    fileOutput.write(str(alpha_M)+'\n')         #8
    fileOutput.write(str(f_c)+', ')             #9
    fileOutput.write(str(gamma_0)+', ')         #10
    fileOutput.write(str(Theta_T)+', ')         #11
    fileOutput.write(str(Theta_Ref_Low)+', ')   #12
    fileOutput.write(str(lambda_T)+', ')        #13
    fileOutput.write(str(C11A)+', ')            #14
    fileOutput.write(str(C12A)+', ')            #15
    fileOutput.write(str(C44A)+'\n')            #16
    fileOutput.write(str(gamma_dot_0)+', ')     #17
    fileOutput.write(str(Am)+', ')              #18
    fileOutput.write(str(S0)+', ')              #19
    fileOutput.write(str(H0)+', ')              #20
    fileOutput.write(str(SS)+', ')              #21
    fileOutput.write(str(Ahard)+', ')           #22
    fileOutput.write(str(QL)+', ')              #23
    fileOutput.write(str(Theta_Ref_High)+'\n ') #24
# Putting in euler angles (texture info)
    fileOutput.write(str(orient[i,0]) + ',' + \
                     str(orient[i,1]) + ',' + \
                     str(orient[i,2]) + ',')    #25,26,27
    fileOutput.write(str(Bsat)+', ')            #28
    fileOutput.write(str(b1)+', ')              #29
    fileOutput.write(str(N_GEN_F)+', ')         #30
    fileOutput.write(str(d_ir)+', ')            #31
    fileOutput.write(str(Dir)+'\n')             #32
    fileOutput.write(str(mu)+', \n')            #33
    fileOutput.write('** ----------------------------------------------'+'\n')

#%% STEPS
for s in range(numStep):
    fileOutput.write('** \n')
    fileOutput.write('** STEP: Step-' + str(s+1)+'\n')
    fileOutput.write('** \n')
    fileOutput.write('*Step, name=Step-'+ str(s+1)+\
                     ', nlgeom='+nlgeom+\
                     ', inc='+ str(inc) +' \n')
#    fileOutput.write('*Static \n')
    fileOutput.write('*Static, direct\n')
    fileOutput.write('0.001, 1., \n')
#    fileOutput.write(str(Initial_Time_Step)+', '+\
#                     str(Total_Increment_Time)+', '+\
#                     str(Min_Step_Size)+', '+\
#                     str(Max_Step_Size)+' \n')
    fileOutput.write("""** 
** 
** BOUNDARY CONDITIONS
** 
*BOUNDARY, type=DISPLACEMENT                          
x0, 1, 1 ,0.0 \n""")
    if s%2 == 0:
        disp = dispMax
    else:
        disp = dispMin
    fileOutput.write('*BOUNDARY, type=DISPLACEMENT \n')                        
    fileOutput.write('x1, 1, 1 ,' + str(disp) + '\n')  
    fileOutput.write("""*BOUNDARY, type=DISPLACEMENT                          
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
*** \n""")
#%% Output Requests
    fileOutput.write("""** 
** OUTPUT REQUESTS
** 
*Restart, write, number interval=1, time marks=NO
** 
** FIELD OUTPUT: F-Output-1
** \n""")
    if s == 0:
        fileOutput.write('*Output, field, variable=PRESELECT, FREQUENCY= ' +\
                         str(freq1) + '\n' )
    else:
        fileOutput.write('*Output, field, variable=PRESELECT, FREQUENCY= ' +\
                         str(freq2) + '\n' )
    fileOutput.write("""*Element Output
EVOL, """)
    skipline = False
    for i in range(len(sdvStartEnd)):
        a = sdvStartEnd[i]
        sdvStart = a[0]
        sdvEnd = a[1] + 1
        if not skipline:
            fileOutput.write('\n')
            for j in range(sdvStart,sdvEnd):
                skipline = False
                fileOutput.write('SDV'+ str(int(j)));
        
                if i == len(sdvStartEnd)-1 and j == sdvEnd - 1:
                    pass
                else:
                    fileOutput.write(', ')

                if j % 5 == 0:
                    fileOutput.write('\n')
                    skipline = True
                    
    fileOutput.write('\n')
    fileOutput.write('*End Step \n')
fileOutput.close()
