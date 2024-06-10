# Jacob Rusch
# 4 May 2023

import os
import time

# Get the current directory
parent_directory = os.getcwd()

# UNCOMMENT LINES BELOW THIS TO RERUN SELECT SIMULATIONS
rerun_folders = [11,14,16,17,18,19,27,32,33,34,35,36,37,41,47,48]
for foldernumber in rerun_folders:
    os.chdir(parent_directory + "/" + str(foldernumber))
    os.system("sbatch Submit_Files.sub")
    os.chdir(parent_directory)


print("Done")
