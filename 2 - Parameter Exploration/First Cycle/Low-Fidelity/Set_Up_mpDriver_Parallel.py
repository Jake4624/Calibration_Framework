# Jacob Rusch
# 9 July 2023

import os

# NEED TO RUN OTHER SCRIPTS BEFORE THIS ONE
#     - mpDriver_Property_Input_Generator.py
#     - writeTexture.py


# Need to run mpDriver_Property_Input_Generator.py before running this
# script. mpDriver_Property_Input_Generator.py will generate the 
# Properties_mpDriver.inc file required for this code to run properly.
#
# Need to run writeTexture.py which generates quinternions which need
# to be imported to the Matlab code firstPoleFigure.m which takes in
# the quinternion information from writeTexture.[y, converts it to
# Euler angles, and outputs that information to texture.inc which is
# required for this code to run. firstPoleFigure.m also generates a
# pole figure which can be used to compare to pole figures seen in the
# literature.
#
# Make sure the following files are up-to-date and in the same
# directory as this file before running it!
#     - makefile
#     - mpDriverUmat_DFGRD.f90
#     - Properties_mpDriver.inc
#     - Submit_Files.sub
#     - texture.inc
#
#
# This script sets up different subdirectories to be run simultaneously

pwd = os.getcwd()

# Number of inputs being varied in the analysis
numinp = 8

# Number of simulations for analysis
numsims = 3**numinp
#numsims = 4

# Break up simulations into different subfolders in order to parallelize
# This will also be the number of processors used when the script is run
# since the script will copy the make files, user material, and input deck
# into the subdirectories and run them all simultaneously.
subfolders = 3**4
#subfolders = 2

simnum = 1
for foldernumber in range(0,subfolders):
    for simcounter in range(numsims/subfolders):
        os.makedirs(str(foldernumber+1) + '/' + str(simnum))
        simnum = simnum + 1
        
# Copy make files, user material, texture file, and subdivide the input deck
# into the different subdirectories. mpDriver_Property_Input_Generator.py
# generates all inputs at once, need to break it up into the number of
# subfolders to run all the different input variations.

# read in input values and save to string
with open('Properties_mpDriver.inc') as values:
    lines = values.readlines()
values.close()

subdirectory_sims = int(numsims/subfolders)

# Make text file which has the beginning and ending value for each
# subfolder sim to be read into mpDriver code
for foldernumber in range(0,subfolders):
    with open('Beginning_and_Ending_Simnum_'+str(foldernumber+1)+'.inc','w') as file:
        file.write(str(subdirectory_sims*(foldernumber)+1)+',')
        file.write(str(subdirectory_sims*(foldernumber+1)))
    file.close()

for foldernumber in range(0,subfolders):
    with open('Properties_mpDriver_'+str(foldernumber+1)+'.inc','w') as file:
        file.writelines(lines[subdirectory_sims*foldernumber:subdirectory_sims*(foldernumber+1)])
        file.close()

for copycounter in range(0,subfolders):
    os.system('cp Updated_Umat.f '  + pwd + '/' + str(copycounter+1))
    os.system('cp texture.inc '  + pwd + '/' + str(copycounter+1))
    os.system('cp Submit_Files.sub '  + pwd + '/' + str(copycounter+1))
    os.system('cp makefile '  + pwd + '/' + str(copycounter+1))
    os.system('cp mpDriverUmat_DFGRD.f90 '  + pwd + '/' + str(copycounter+1))
    os.system('mv Properties_mpDriver_'+str(copycounter+1)+'.inc '  + pwd + '/' + str(copycounter+1)+ '/Properties_mpDriver.inc')
    os.system('mv Beginning_and_Ending_Simnum_'+str(copycounter+1)+'.inc '  + pwd + '/' + str(copycounter+1)+ '/Beginning_and_Ending_Simnum.inc')
    os.system('cp -r DFGRD1_Vals '  + pwd + '/' + str(copycounter+1))
      
print('done')
