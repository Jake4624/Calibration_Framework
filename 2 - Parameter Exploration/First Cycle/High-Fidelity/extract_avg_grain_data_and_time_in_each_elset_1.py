# Jacob Rusch
# Marquette University
# 5 July 2023
#
# Volume averaged stress of polygranular ABAQUS simulations
# This script goes through every frame in each step and 
# loops over each element set (grain) to find the average value
# of stress at each grain, then takes the average of each 

# imports
import numpy as np
from odbAccess import *
import sys
import time

start_time = time.time()

# replace with the name of the field output you want to retrieve
field_output_1 = 'LE' 
component_1 = 'LE11'

field_output_2 = 'S' 
component_2 = 'S11'

#parent_directory = os.getcwd()

directories = [1,6]     # 1
#directories = [7,12]   # 2
#directories = [13,18]   # 3
#directories = [19,24]   # 4
#directories = [25,30]   # 5
#directories = [31,36]   # 6
#directories = [37,42]   # 7
#directories = [43,48]   # 8
#directories = [49,54]   # 9
#directories = [55,60]   # 10
#directories = [61,66]   # 11
#directories = [67,72]   # 12
#directories = [73,78]   # 13
#directories = [79,84]   # 14
#directories = [85,90]   # 15
#directories = [91,96]   # 16
#directories = [97,102]  # 17
#directories = [103,108] # 18
#directories = [109,114] # 19
#directories = [115,120] # 20
#directories = [121,126] # 21
#directories = [127,132] # 22
#directories = [133,138] # 23
#directories = [139,144] # 24
#directories = [145,150] # 25
#directories = [151,156] # 26
#directories = [157,162] # 27
#directories = [163,168] # 28
#directories = [169,174] # 29
#directories = [175,180] # 30
#directories = [181,186] # 31
#directories = [187,192] # 32


# loop over all simulations
for odb_num in range(directories[0],directories[1]+1):
    #os.chdir(parent_directory+'/'+str(obd_num+1))
    odbname = str(odb_num)+'.odb'
    odb = openOdb(odbname, readOnly=True)
    print('In ODB: ',odbname)
        
    part = odb.rootAssembly.instances.keys()[0]
    #print('Part: ',part)

    # number of element sets
    elsets = odb.rootAssembly.instances[part].elementSets.keys()
    nelsets = len(elsets)
    # Last elset is ALL, can exclude if only interested in grains.
    # Include it if wanted
    #print('Number of element sets:', nelsets-1)

    # Loop over all element sets, steps, and frames
    # Last elset name is ALL, can uncomment if wanted
    # otherwise this will just do the assigned elsets
    # for Neper sims
    
    for elset_num in range(0,nelsets-1):
        elset_name = elsets[elset_num]
        #print('In Elset: ',elset_name)

        # Open file to write to
        out1 = open(elset_name+'_'+field_output_1+'_'+component_1+'_Avg_'+str(odb_num)+'.txt','w')
        out2 = open(elset_name+'_'+field_output_2+'_'+component_2+'_Avg_'+str(odb_num)+'.txt','w')

        # Number of steps
        nbSteps = len(odb.steps)
        #print('Number of steps:', nbSteps)
        # Loop over all steps
        for Step_num in range(0,nbSteps):
            # step number
            #print('Step number:', Step_num+1)
            # Number of frames in step
            step_name = odb.steps.keys()[Step_num]
            #print('Step name: ',step_name)
            step = odb.steps.values()[Step_num]
            
            nframes = len(step.frames)
            #print('Number of frames:', nframes)
            
            # Loop over all frames
            for i in range(nframes):

                # The first frame of the first step in Abaqus are
                # zeros. The values in the first frame of all 
                # subsequent steps are the same as the last entry
                # of the step before it. (e.g., Values for final
                # frame of step 2 are the same as the first
                # values in step 3)
                if i != 0:
                    frame = step.frames[i]
                    #print('Frame number: ',i)
                    strainField = odb.steps[step_name].frames[i].\
                                  fieldOutputs[field_output_1]
                    stressField = odb.steps[step_name].frames[i].\
                                  fieldOutputs[field_output_2]
                    grain_set = odb.rootAssembly.instances[part].\
                                elementSets[elset_name].elements
                    #print('Number of elements in set: ',len(grain_set))
                    # Loop over each element in the frame and take average
                    # stress of all elements in the set to get the stress of
                    # the grains at each step
                    sumvar_1 = 0
                    sumvar_2 = 0
                    for el in range(len(grain_set)):
                        elem = grain_set[el]
                        element_strain = strainField.\
                                         getSubset(position=\
                                                   INTEGRATION_POINT, \
                                                   region=elem).\
                                         getScalarField(componentLabel=\
                                                        component_1).\
                                         values[0].data
                        element_stress = stressField.\
                                         getSubset(position=\
                                                   INTEGRATION_POINT, \
                                                   region=elem).\
                                         getScalarField(componentLabel=\
                                                        component_2).\
                                         values[0].data
                        sumvar_1 += element_strain
                        sumvar_2 += element_stress
                    # Average stress over all elements in grain set
                    avevar_1 = sumvar_1/len(grain_set)
                    avevar_2 = sumvar_2/len(grain_set)

                    out1.write(str(avevar_1)+'\n')
                    out2.write(str(avevar_2)+'\n')
                    
        out1.close()
        out2.close()


    # Make one file that is for time of simulation to make this code run faster
    # Loop over all steps
    #odbname = '64_Grain_Cyclic.odb'
    #odb = openOdb(odbname, readOnly=True)
    #print('In ODB: ',odbname)
    out_time = open('Simulation_Time_'+str(odb_num)+'.txt','w')
    nbSteps = len(odb.steps)
    for Step_num in range(0,nbSteps):
        step = odb.steps.values()[Step_num]
        nframes = len(step.frames)
        total_time = step.totalTime
        # Loop over all frames
        for i in range(nframes):
            if i != 0:
                frame = step.frames[i]
                time_var = total_time + frame.frameValue
                out_time.write(str(time_var)+'\n')
    out_time.close()


    odb.close()

end = time.time()
print('Time to run script in seconds is: ')
print("--- %s seconds ---" % (time.time() - start_time))
print('Done')



