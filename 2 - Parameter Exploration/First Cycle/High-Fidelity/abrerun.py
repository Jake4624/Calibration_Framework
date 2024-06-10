import os
import time


for i in range(256):
#    os.system('rm '+str(i)+'.odb')
    os.system('rm '+str(i+1)+'.com')
    os.system('rm '+str(i+1)+'.dat')
    os.system('rm '+str(i+1)+'.mdl')
    os.system('rm '+str(i+1)+'.msg')
    os.system('rm '+str(i+1)+'.prt')
    os.system('rm '+str(i+1)+'.res')
    os.system('rm '+str(i+1)+'.sim')
    os.system('rm '+str(i+1)+'.sta')
    os.system('rm '+str(i+1)+'.stt')

start_time = time.time()
for i in [1,3,5]:
    print('abaqus job='+ str(i) +  ' user=Updated_Umat.f inp='+ str(i) +  ' cpus=16 interactive')
    os.system('abaqus job='+ str(i) +  ' user=Updated_Umat.f inp='+ str(i) +  ' cpus=16 interactive')
        
print("--- %s seconds ---" % (time.time() - start_time))
print('done')
