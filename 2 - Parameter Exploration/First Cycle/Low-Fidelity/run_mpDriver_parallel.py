# Jacob Rusch
# 4 May 2023

import os
import time

# Compile Fortran code in all subfolders

# This script goes into the designated number of subdirectories, 
# runs the make command to ensure everything is in order, then
# runs ./mpDriver and records how much time it took to run.

# Number of subfolders
subfolders = 3**4

# Get the current directory
parent_directory = os.getcwd()

# Loop over the subfolders and compile the fortran code
for foldernumber in range(0,subfolders):
    print(str(foldernumber+1))
    os.chdir(parent_directory + "/" + str(foldernumber+1))
    os.system("sbatch Submit_Files.sub")
    os.chdir(parent_directory)

print("Done")
