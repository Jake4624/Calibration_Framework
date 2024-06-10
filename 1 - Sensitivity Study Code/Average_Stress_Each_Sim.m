% Jacob Rusch
% 8 May 2023
%
% Average calculations
% This code takes in the information from the .dat files output by
% mpDriver (low-fidelity model)and calculates the error between a baseline
% simulation and simulations with inputs generated using a full factorial
% analysis technique.
%
% See LOG_mpDriver_Parallel for details on how the simulations were run and
% the what files (code) were used to run them.

clear;
close all;
clc;

%% Inputs Required for RMSE Analysis
% Number of material points (or "elements") to compare this analysis to
num_mp = 27;
% Number of simulations NOT INCLUDING baseline simulation
total_sims = 2^16;
subfolders = 2^7;
% Number of simulations in each folder to loop over
num_sims = total_sims/subfolders;

%% Get baseline sim data
% Baseline sim data is from an abaqus simulation which had inputs taken
% from literature values in Manchiraju and Anderson 2010 and Yu et al 2013
%
% Get path for current folder
%

currentFolder = pwd;

cd(strcat(currentFolder,"\0\1\"));
for mp_count = 1:num_mp
    filename = num2str(mp_count)+".dat";
    file_data = importdata(filename);
    baseline_time(:,mp_count) = round(file_data(:,1),4);
    baseline_stress(:,mp_count) = file_data(:,2);
end
% go back to main folder where this MATLAB script is saved.
cd(currentFolder)

Average_Baseline_Time = mean(baseline_time,2);
Average_Baseline_Stress = mean(baseline_stress,2);

writematrix([Average_Baseline_Time,Average_Baseline_Stress],"Average_Baseline_Stress.txt")

% figure
% subplot(1,2,1)
% plot(baseline_time,baseline_stress)
% title("Baseline Stress v Time for All 27 Material Points")
% xlabel("Time (s)")
% ylabel("Stress in Loading Direction \sigma_1_1 (MPa)")
% subplot(1,2,2)
% plot(Average_Baseline_Time,Average_Baseline_Stress)
% title("Averaged Baseline Stress v Time")
% xlabel("Time (s)")
% ylabel("Stress in Loading Direction \sigma_1_1 (MPa)")

%% Loop Over All Subfolders
for subfolder_num = 1:subfolders

    cd(strcat(currentFolder,"\",num2str(subfolder_num),"\"));
    simcount_data = importdata("Beginning_and_Ending_Simnum.inc");
    simcount_start = simcount_data(1);
    simcount_end = simcount_data(2);

    for sim_count = 1:num_sims
        disp("Sim Count = "+num2str(simcount_start+sim_count-1))
        cd(strcat(currentFolder,"\",num2str(subfolder_num),"\",num2str(simcount_start+sim_count-1),"\"));
        for i = 1:num_mp
            filename = num2str(i)+"_Spline.txt";
            file_data = importdata(filename);
            % mp_time = file_data(:,1);
            mp_stress(:,i) = file_data(:,2);
        end
        Average_Stress(:,simcount_start+sim_count-1) = mean(mp_stress,2);
    end
    % go back to main folder where this MATLAB script is saved.
    cd(currentFolder)
end

writematrix([Average_Baseline_Time,Average_Stress],"Average_Stress_Each_Sim.txt")
