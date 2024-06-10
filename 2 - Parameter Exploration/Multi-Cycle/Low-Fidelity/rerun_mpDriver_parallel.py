# Jacob Rusch
# 4 May 2023

import os
import time

# Compile Fortran code in all subfolders

# This script goes into the designated number of subdirectories, 
# runs the make command to ensure everything is in order, then
# runs ./mpDriver and records how much time it took to run.

# Number of subfolders
#subfolders = [3,4,6,15,19,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,40,46,49,54,56,57,58,59,60,61,62,63,66,67,68,69,70,71,72]
subfolders = [23,25,29,46,58]

#subfolders = 2



# Get the current directory
parent_directory = os.getcwd()

# Loop over the subfolders and compile the fortran code
for foldernumber in subfolders:
    print('Subfolder: ',foldernumber)
    os.chdir(parent_directory + "/" + str(foldernumber))
    os.system("sbatch Submit_Files.sub")
    os.chdir(parent_directory)


# UNCOMMENT LINES BELOW THIS TO RERUN SELECT SIMULATIONS
#rerun_folders = [11,14,16,17,18,19,27,32,33,34,35,36,37,41,47,48]
## Loop over the subfolders and compile the fortran code
#for foldernumber in rerun_folders:
#    os.chdir(parent_directory + "/" + str(foldernumber))
#    os.system("sbatch Submit_Files.sub")
#    os.chdir(parent_directory)


print("Done")
