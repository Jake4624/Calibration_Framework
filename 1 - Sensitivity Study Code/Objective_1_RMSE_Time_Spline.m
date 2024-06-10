% Jacob Rusch
% 8 May 2023
%
% RMSE calculations
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

% Flag for test plots: 1 == ON, 0 == OFF
test_flag = 0;

%% Analysis details
% A material point driver code is used to run simulations with varying
% input values and compare it to a baseline simulation.

% The "true" value of stress and strain from the baseline simulation
% that the other simulations are being compared to are the published
% values in Manchiraju and Anderson's 2010 paper and Yu's 2013 paper.

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
    baseline_time{mp_count} = file_data(:,1);
    baseline_stress{mp_count} = file_data(:,2);
end

% go back to main folder where this MATLAB script is saved.
cd(currentFolder)

%% Loop Over All Subfolders
for subfolder_num = 1:subfolders
% Get data from text files
% FILE ORGANIZATION:
% When analyzing the results, put all the simulation output folders into a
% single directory. This means there will be 1+num_sim folders contating
% num_mp files in each folder. There will also be a baseline sim folder
% labeled 0 which all the other simulations will be compared to.
%
% NOTE: The baseline simulation should be in a folder labeled 0
% mpDriver saves the output stresses,strains,and other state variables in a
% .dat file which is organized with the following scheme for columns:
% Title: | Time  |       Strain      |         Stress        | SDV/Other
% Col#:  |  (1)  |(2)(3)(4)(5)(6)(7) |(8)(9)(10)(11)(12)(13) | ...
%        |-------+-------------------+-----------------------+-------
% Var:   |time(2)| 11 22 33 12 13 23 | 11 22 33  12  13  23  | SDV#/Other

    cd(strcat(currentFolder,"\",num2str(subfolder_num),"\"));
    simcount_data = importdata("Beginning_and_Ending_Simnum.inc");
    simcount_start = simcount_data(1);
    simcount_end = simcount_data(2);

    for sim_count = 1:num_sims
        cd(strcat(currentFolder,"\",num2str(subfolder_num),"\",...
            num2str(simcount_start+sim_count-1),"\"));
        for i = 1:num_mp
            filename = num2str(i)+".dat";
            file_data = importdata(filename);
            mp_driver_time{sim_count,i} = file_data(:,1);
            stress{sim_count,i} = file_data(:,2);
        end
    end

    % go back to main folder where this MATLAB script is saved.
    cd(currentFolder)


    %% Seperate the data into the cycles
    % Find extremes of strain values in order to determine cycles. idx is 
    % the index number where the sign changes (i.e. goes from increasing to
    % decreasing strain value or decreasing to increasing)
    % idx_T finds end of tensile loads
    % idx_C finds end of compression loads
    % idx sorts and combines idx_C and idx_T

    % Loading is from 1:{sim_count,i}(j), unloading is from
    % {sim_count,i}(j+1):end for a simulation that only has one cycle, need
    % to adjust for simulations with multiple cycles
    %
    % To generate spline curves for stress strain data, put inputs in this
    % order:
    % spline(strain_sim,stress_sim,actual_strain_being_compared_to)
    %
    % Note: actual_strain_being_compared_to could be experimental data OR
    % it could be a different simulation in order to do a sensitivity study
    % NOTE: In this set of simulations, run 0 is the "true" value of stress
    % and strain that the other simulations are being compared to. If 
    % comparing to experimental data, need to adjust this code.
    %

    for sim_count = 1:num_sims
        for i=1:num_mp
            % Baseline Sim and mpDriver Sim should have the same number of
            % cycles.
            stress_spline{sim_count,i} = ...
                spline(mp_driver_time{sim_count,i},...
                stress{sim_count,i},...
                baseline_time{i});
        end
    end

    %% RMSE Calculations
    %
    % Taking the first simulation as the "true" values, the RMSE is found for
    % the first and last cycles to demonstrate that changing the ratchetting
    % parameters while holding the plasticity parameters constant only has a
    % large effect on the subsequent cycles and not the first cycle.
    %
    % RMSE = sqrt((1/N)*sum((Predicted(i)-Actual(i))^2))
    %
    for sim_count = 1:num_sims
        for i=1:num_mp
            RMSE(sim_count,i) = ...
                sqrt((1/length(stress_spline{sim_count,i}))*...
                sum((stress_spline{sim_count,i} - ...
                baseline_stress{i}).^2));
        end
    end

    %% Gather Statistics on RMSE Values
    % Taking the average across all elements in each simulation and finding
    % the standard deviation will help quantify how each material point
    % behaved with respect to the others in the same simulation with the 
    % same input parameters.
    RMSE_Avg = mean(RMSE,2);
    RMSE_STD = std(RMSE,0,2);

    %% Write to Output File
    writematrix(RMSE_Avg,"RMSE_Values_"+num2str(subfolder_num)+".txt")
end
%% Plots for Test Data
% Do not have this be active when doing the full analysis. Only have this
% section active if there are a few simulations and a few material points
% to see if the code is working properly

mean_baseline_time = mean(cell2mat(baseline_time),2);
mean_baseline_stress = mean(cell2mat(baseline_stress),2);
%%
if test_flag == 1

    figure
    plot(mean_baseline_time,mean_baseline_stress,'--','LineWidth',1.5)
    hold on
    plot(mean_baseline_time,mean(cell2mat(stress_spline(500,:)),2),'LineWidth',1.5)
    legend('Baseline Data','mpDriver Data','Location','northoutside')
    xlabel('Step Time')
    ylabel('Stress (MPa)')
end