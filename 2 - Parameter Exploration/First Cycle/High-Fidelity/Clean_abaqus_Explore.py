import os

# This script removes all subfolders and their contents
subfolders = 256

for copycounter in range(0,subfolders):
    os.system('rm -r '  + str(copycounter+1) + "/")

      
print('done')
