import os

subfolders = 3**4

for i in range(subfolders):
    os.system('du -sh '+str(i+1)+'/')
