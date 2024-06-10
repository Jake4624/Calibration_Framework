import os

subfolders = 17

for i in range(subfolders):
    os.system('du -sh '+str(i+1)+'/')
