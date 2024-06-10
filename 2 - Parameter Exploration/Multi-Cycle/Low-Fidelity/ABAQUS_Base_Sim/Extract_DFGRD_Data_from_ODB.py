# Current as of: 13 April 2023
#
# This script extracts the DFGRD1 values at each time step in every integration 
# point for one simulation.
#
# The DFGRD1 values will be used for Objectives 2 and 3 of the dissertation
#
# To run this script, type the following into the command line in the same 
# directory as where the .odb files are saved
#
# abaqus python script.py => Fast (use this one)
# abaqus cae noGUI=script.py => Slow

# imports
import numpy as np
from odbAccess import *
import sys

# State Variables SDV220-228 are DFGRD1
# Run 0 is the baseline sim that used the published values from Manchiraju
# and Anderson (2010) along with the values from Yu (2013) for transformation
# ratcheting.

# job name
odbname = 'Baseline_Sim.odb'

odb = openOdb(odbname, readOnly=True)
print('In ODB: ',odbname)
    
# Number of steps
nbSteps = len(odb.steps)
#print('Number of steps:', nbSteps)

# number of elements is the same throughout each step since
# element deletion is not being used.
nelem = len(odb.steps.values()[1].frames[0].fieldOutputs['EVOL'].values)
#print('Number of elements:', nelem)

for e in range (0,nelem):
    # open txt file to write to
    out1 = open('Element_'+str(e+1)+'.txt','w')
    print('Element number:', e+1)

    for Step_num in range(0,nbSteps):
        # step number
        #print('Step number:', Step_num+1)
        # Number of frames in step
        step = odb.steps.values()[Step_num]
        nframes = len(step.frames)
        #print('Number of frames:', nframes)
       
        # part
        #part = odb.rootAssembly.instances['PART-1-1']    
        part = odb.rootAssembly.instances.keys()[0]
        
        for i in range(nframes):
            # The first frame of the first step in Abaqus are
            # zeros. The values in the first frame of all 
            # subsequent steps are the same as the last entry
            # of the step before it. (e.g., Values for final
            # frame of step 2 are the same as the first
            # values in step 3)
            if i != 0:
            #print('Frame', i + 1, '/', nframes)
                frame = step.frames[i]

                # extract varibles desired
                outvar1=frame.fieldOutputs['SDV220']
                outvar2=frame.fieldOutputs['SDV221']
                outvar3=frame.fieldOutputs['SDV222']
                outvar4=frame.fieldOutputs['SDV223']
                outvar5=frame.fieldOutputs['SDV224']
                outvar6=frame.fieldOutputs['SDV225']
                outvar7=frame.fieldOutputs['SDV226']
                outvar8=frame.fieldOutputs['SDV227']
                outvar9=frame.fieldOutputs['SDV228']

                # Value of component
                # For variables that do not have sub categories
                # only need to put .data at end. For variables
                # with subcategories (e.g., S.S11, S.S13, etc.)
                # Need to specify position in array. For example
                # To get S33 need to have fieldOutputs['S']
                # outvar.values[e].data[3] since Abaqus puts
                # stresses and strains in [11 22 33 12 13 23] order
                # NEED TO CHECK ORDER TO ENSURE IT IS CORRECT 8 April 2023
                var1=outvar1.values[e].data
                var2=outvar2.values[e].data
                var3=outvar3.values[e].data
                var4=outvar4.values[e].data
                var5=outvar5.values[e].data
                var6=outvar6.values[e].data
                var7=outvar7.values[e].data
                var8=outvar8.values[e].data
                var9=outvar9.values[e].data

                # write data
                out1.write(str(var1)+', '+str(var2)+', '+
                           str(var3)+', '+str(var4)+', '+
                           str(var5)+', '+str(var6)+', '+
                           str(var7)+', '+str(var8)+', '+
                           str(var9)+'\n')

    out1.close()
            
print('Done')
# close odb file
odb.close()
