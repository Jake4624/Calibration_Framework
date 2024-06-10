import os
import time

# this removes every abaqus file in directory, uncomment line below to delete certain files 
# os.system('rm *.com *.dat *.log *.mdl *.odb *.prt *.res *.sta *.stt *.abq *.msg *.pac *.sel *.lck *.sim *.odb_f *.023')

numsim = 2**8
#numsim=2


start_time = time.time()
for i in range(0,numsim):
    print('abaqus job='+ str(i+1) +  ' user=Updated_Umat.f inp='+ str(i+1) +  ' cpus=16 interactive')
    os.system('abaqus job='+ str(i+1) +  ' user=Updated_Umat.f inp='+ str(i+1) +  ' cpus=16 interactive')
        
print("--- %s seconds ---" % (time.time() - start_time))
print('done')