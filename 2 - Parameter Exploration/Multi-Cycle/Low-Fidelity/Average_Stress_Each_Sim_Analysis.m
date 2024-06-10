% Jacob Rusch
% 12 July 2023
%
% Average calculations
% This code takes in the information from the .dat files output by
% mpDriver (low-fidelity model) and calculates the volume averaged stress
% for each simulation and saves it in a separate file for later analysis
%
% See LOG_mpDriver_Parallel for details on how the simulations were run and
% the what files (code) were used to run them.

clear;
close all;
clc;

%% Inputs Required for RMSE Analysis
% Number of material points (or "elements") to compare this analysis to
num_mp = 4^3;
% Number of simulations NOT INCLUDING baseline simulation
% Note that some simulations needed to be excluded for this Objective since
% the boundary conditions were such that when Theta_T > Theta_ref_high the
% HF sims would not run due to convergence issues. Therefore, it was
% exclded. These were included in Objective 1 for reasons outlined in the
% dissertation.
total_sims = 17^3;
subfolders = 17;
% Number of simulations in each folder to loop over
num_sims = total_sims/subfolders;

%% Skip sim
% This simulation was not able to be run for some reason. Due to time
% constraints it is left out. Will come back to it later if time permits.

skip_sim = 0;

%% Number of points in output file
% Simulations run using mpDriver often do not have the same number of steps
% for each material point in the simulation. Therefore, in order to be able
% to compare and average all points together, a standard number of points
% will be chosen to interpolate over using the spline function.
%
% This is most easily done by using step time to spline things together.
% Each step is 1 second, the simulations for the First cycle are 2 steps
% which can be divided into many steps. For this analysis, time steps of
% 0.01 seconds were chosen meaning there will be 201 points between 0 and 2
% seconds to spline the information over.
time_spline_base = linspace(0,10,1001);

%% Loop Over All Subfolders

% Unlike HF sims, LF sims were rerun to exclude the simulations where
% Theta_T > Theta_ref_high. No need to exclude any sims here.

currentFolder = pwd;

for subfolder_num = 1:subfolders

    cd(strcat(currentFolder,"\",num2str(subfolder_num),"\"));
    simcount_data = importdata("Beginning_and_Ending_Simnum.inc");
    simcount_start = simcount_data(1);
    simcount_end = simcount_data(2);

    % Note that many of the variables below are being overwritten each
    % loop, this is to save memory as there are several thousand
    % simulations to loop over. Instead the values are output to a text
    % file which can be imported later for analysis.
    for sim_count = 1:num_sims
        if simcount_start+sim_count-1 == skip_sim
            continue
        else
            % disp("Sim Count = "+num2str(simcount_start+sim_count-1))
            cd(strcat(currentFolder,"\",num2str(subfolder_num),"\",...
                num2str(simcount_start+sim_count-1),"\"));
            for i = 1:num_mp
                % Get data from text files
                %
% FILE ORGANIZATION:
% When analyzing the results, put all the simulation output folders into a
% single directory. This means there will be 1+num_sim folders contating
% num_mp files in each folder.
%
% mpDriver saves the output stresses,strains,and other state variables in a
% .dat file which is organized with the following scheme for columns:
% Title: | Time  |       Strain      |         Stress        | SDV/Other
% Col#:  |  (1)  |(2)(3)(4)(5)(6)(7) |(8)(9)(10)(11)(12)(13) | ...
%        |-------+-------------------+-----------------------+-------
% Var:   |time(2)| 11 22 33 12 13 23 | 11 22 33  12  13  23  | SDV#/Other
% Generating splines to be able to average stress and other
% variables in simulations
                filename = num2str(i)+".dat";
                file_data = importdata(filename);
                mp_time = file_data(:,1);
                mp_strain = file_data(:,[2:7]);
                mp_stress = file_data(:,[8:13]);

                % Need to make the first entry 0 since everything should
                % start out at 0 time, 0 stress, 0 strain.
                mp_time = [0;mp_time];
                mp_strain = [0,0,0,0,0,0;mp_strain];
                mp_stress = [0,0,0,0,0,0;mp_stress];

                stress_spline_11(:,i) = spline(mp_time,mp_stress(:,1),...
                    time_spline_base)';
                stress_spline_22(:,i) = spline(mp_time,mp_stress(:,2),...
                    time_spline_base)';
                stress_spline_33(:,i) = spline(mp_time,mp_stress(:,3),...
                    time_spline_base)';
                stress_spline_12(:,i) = spline(mp_time,mp_stress(:,4),...
                    time_spline_base)';
                stress_spline_13(:,i) = spline(mp_time,mp_stress(:,5),...
                    time_spline_base)';
                stress_spline_23(:,i) = spline(mp_time,mp_stress(:,6),...
                    time_spline_base);

                strain_spline_11(:,i) = spline(mp_time,mp_strain(:,1),...
                    time_spline_base)';
                strain_spline_22(:,i) = spline(mp_time,mp_strain(:,2),...
                    time_spline_base)';
                strain_spline_33(:,i) = spline(mp_time,mp_strain(:,3),...
                    time_spline_base)';
                strain_spline_12(:,i) = spline(mp_time,mp_strain(:,4),...
                    time_spline_base)';
                strain_spline_13(:,i) = spline(mp_time,mp_strain(:,5),...
                    time_spline_base)';
                strain_spline_23(:,i) = spline(mp_time,mp_strain(:,6),...
                    time_spline_base)';

            end

            Average_Strain_11 = ...
                mean(strain_spline_11,2);
            Average_Strain_22 = ...
                mean(strain_spline_22,2);
            Average_Strain_33 = ...
                mean(strain_spline_33,2);
            Average_Strain_12 = ...
                mean(strain_spline_12,2);
            Average_Strain_13 = ...
                mean(strain_spline_13,2);
            Average_Strain_23 = ...
                mean(strain_spline_23,2);

            Average_Stress_11 = ...
                mean(stress_spline_11,2);
            Average_Stress_22 = ...
                mean(stress_spline_22,2);
            Average_Stress_33 = ...
                mean(stress_spline_33,2);
            Average_Stress_12 = ...
                mean(stress_spline_12,2);
            Average_Stress_13 = ...
                mean(stress_spline_13,2);
            Average_Stress_23 = ...
                mean(stress_spline_23,2);

            writematrix([Average_Strain_11],"Average_Strain_11_"+...
                num2str(simcount_start+sim_count-1)+".txt")
            writematrix([Average_Strain_22],"Average_Strain_22_"+...
                num2str(simcount_start+sim_count-1)+".txt")
            writematrix([Average_Strain_33],"Average_Strain_33_"+...
                num2str(simcount_start+sim_count-1)+".txt")
            writematrix([Average_Strain_12],"Average_Strain_12_"+...
                num2str(simcount_start+sim_count-1)+".txt")
            writematrix([Average_Strain_13],"Average_Strain_13_"+...
                num2str(simcount_start+sim_count-1)+".txt")
            writematrix([Average_Strain_23],"Average_Strain_23_"+...
                num2str(simcount_start+sim_count-1)+".txt")

            writematrix([Average_Stress_11],"Average_Stress_11_"+...
                num2str(simcount_start+sim_count-1)+".txt")
            writematrix([Average_Stress_22],"Average_Stress_22_"+...
                num2str(simcount_start+sim_count-1)+".txt")
            writematrix([Average_Stress_33],"Average_Stress_33_"+...
                num2str(simcount_start+sim_count-1)+".txt")
            writematrix([Average_Stress_12],"Average_Stress_12_"+...
                num2str(simcount_start+sim_count-1)+".txt")
            writematrix([Average_Stress_13],"Average_Stress_13_"+...
                num2str(simcount_start+sim_count-1)+".txt")
            writematrix([Average_Stress_23],"Average_Stress_23_"+...
                num2str(simcount_start+sim_count-1)+".txt")
        end
        % go back to main folder where this MATLAB script is saved.
        cd(currentFolder)
    end
end


%% Collect all S11 average values into one document
% 
cd(currentFolder)

for subfolder_num = 1:subfolders
    cd(strcat(currentFolder,"\",num2str(subfolder_num),"\"));
    simcount_data = importdata("Beginning_and_Ending_Simnum.inc");
    simcount_start = simcount_data(1);
    simcount_end = simcount_data(2);
    for sim_count = 1:num_sims
        if simcount_start+sim_count-1 == skip_sim
            continue
        else
            cd(strcat(currentFolder,"\",num2str(subfolder_num),"\",...
                num2str(simcount_start+sim_count-1),"\"));
            filename_stress = "Average_Stress_11_"+...
                num2str(simcount_start+sim_count-1)+".txt";
            Avg_Sim_Stress(:,simcount_start+sim_count-1) =...
                importdata(filename_stress);
            filename_strain = "Average_Strain_11_"+...
                num2str(simcount_start+sim_count-1)+".txt";
            Avg_Sim_Strain(:,simcount_start+sim_count-1) =...
                importdata(filename_strain);
        end
    end
end
cd(currentFolder)

% for i = length(skip_sim)
%     Avg_Sim_Stress(:,skip_sim) = [];
%     Avg_Sim_Strain(:,skip_sim) = [];
% end

writematrix([Avg_Sim_Stress],"Average_Stress_11_All_mpDriver_Sims.txt")
writematrix([Avg_Sim_Strain],"Average_Strain_11_All_mpDriver_Sims.txt")

