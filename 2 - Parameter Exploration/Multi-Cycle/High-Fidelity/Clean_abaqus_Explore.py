import os

# This script removes all subfolders and their contents
subfolders = 6**3

for copycounter in range(0,subfolders):
    os.system('rm -r '  + str(copycounter+1) + "/")

      
print('done')
