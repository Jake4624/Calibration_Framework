import os

# Number of simulations
num_sim = 6**3
# num_sim = 1

#for i in range(num_sim):
# #create new single directory
#    os.mkdir(str(i+1))

parent_directory = os.getcwd()

grains_to_exclude = [31,42]

# Uncomment to move files from subdirectories to parent directory
#for i in range(256):
#    os.system('mv '+str(i+1)+'/'+str(i+1)+'.odb '+ parent_directory)

# Uncomment to move files from parent directory to subdirectories
# for i in range(num_sim):
#     os.system('mv '+str(i+1)+'.inp '+ parent_directory+'/'+str(i+1))
#     os.system('mv '+str(i+1)+'.odb '+ parent_directory+'/'+str(i+1))
#     os.system('mv '+str(i+1)+'.com '+ parent_directory+'/'+str(i+1))
#     os.system('mv '+str(i+1)+'.dat '+ parent_directory+'/'+str(i+1))
#     os.system('mv '+str(i+1)+'.mdl '+ parent_directory+'/'+str(i+1))
#     os.system('mv '+str(i+1)+'.msg '+ parent_directory+'/'+str(i+1))
#     os.system('mv '+str(i+1)+'.prt '+ parent_directory+'/'+str(i+1))
#     os.system('mv '+str(i+1)+'.res '+ parent_directory+'/'+str(i+1))
#     os.system('mv '+str(i+1)+'.sim '+ parent_directory+'/'+str(i+1))
#     os.system('mv '+str(i+1)+'.sta '+ parent_directory+'/'+str(i+1))
#     os.system('mv '+str(i+1)+'.stt '+ parent_directory+'/'+str(i+1))
#     for j in range(64):
#         if (j+1 in grains_to_exclude):
#             continue
#         else:
#             os.system('mv POLY'+ str(j+1) +'_S_S11_Avg_'+ str(i+1) +'.txt '+ parent_directory+'/'+str(i+1))
#             os.system('mv POLY'+ str(j+1) +'_LE_LE11_Avg_'+ str(i+1) +'.txt '+ parent_directory+'/'+str(i+1))
#     os.system('mv Simulation_Time_'+ str(i+1) +'.txt '+ parent_directory+'/'+str(i+1))  


# WINDOWS Commands. Use code below if running using Python for Windows
for i in range(num_sim):
    os.system('move '+str(i+1)+'.inp '+str(i+1)+'/')
    os.system('move '+str(i+1)+'.odb '+str(i+1)+'/')
    os.system('move '+str(i+1)+'.com '+str(i+1)+'/')
    os.system('move '+str(i+1)+'.dat '+str(i+1)+'/')
    os.system('move '+str(i+1)+'.mdl '+str(i+1)+'/')
    os.system('move '+str(i+1)+'.msg '+str(i+1)+'/')
    os.system('move '+str(i+1)+'.prt '+str(i+1)+'/')
    os.system('move '+str(i+1)+'.res '+str(i+1)+'/')
    os.system('move '+str(i+1)+'.sim '+str(i+1)+'/')
    os.system('move '+str(i+1)+'.sta '+str(i+1)+'/')
    os.system('move '+str(i+1)+'.stt '+str(i+1)+'/')
    for j in range(64):
        if (j+1 in grains_to_exclude):
            continue
        else:
            os.system('move POLY'+ str(j+1) +'_S_S11_Avg_'+ str(i+1) +'.txt '+str(i+1)+'/')
            os.system('move POLY'+ str(j+1) +'_LE_LE11_Avg_'+ str(i+1) +'.txt '+str(i+1)+'/')
    os.system('move Simulation_Time_'+ str(i+1) +'.txt '+str(i+1)+'/')  
