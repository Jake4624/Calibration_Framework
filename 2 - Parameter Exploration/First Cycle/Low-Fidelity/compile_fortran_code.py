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
for foldernumber in range(subfolders):
    os.chdir(parent_directory + "/" + str(foldernumber+1))
    # The make statement compiles the code
    os.system("make")
    # Waiting for the user material .o file to appear to indicate the code 
    # compiling is completed
    while not os.path.exists(parent_directory + "/" + str(foldernumber+1) + "/mpDriverUmat_DFGRD.o"):
        time.sleep(1)
    # Return to parent directory
    os.chdir(parent_directory)

print("Done")
