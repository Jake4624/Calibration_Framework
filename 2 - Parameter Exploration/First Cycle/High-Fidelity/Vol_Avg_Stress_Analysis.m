% Jacob Rusch
% 13 July 2023
%
% Takes overall volume average for each sim and analyzes them against
% experimental data to get error values to put into ML code.

clear;
close all;
clc;

subfolders = 2^8;
numgrains = 64;

%%
% Element sets 31 and 42 do not exist in this Neper simulation. Therefore
% the Neper simulation only has 62 grains instead of 64. The mesh is too
% course to have the 31 and 42 element sets. It still provides a good
% stress-strain curve, though. See 64_Elem_v_Neper folder and dissertation
% for comparison.
skip_grains = [31,42];

currentFolder = pwd;
for subfolder_num = 1:subfolders

    cd(strcat(currentFolder,"\",num2str(subfolder_num),"\"));
    for graincount = 1:numgrains
        if ismember(graincount,skip_grains) == 1
            continue
        else
            stress_filename = "POLY"+num2str(graincount)+...
                "_S_S11_Avg_"+num2str(subfolder_num)+".txt";
            grain_stress(:,graincount) = importdata(stress_filename);
            strain_filename = "POLY"+num2str(graincount)+...
                "_LE_LE11_Avg_"+num2str(subfolder_num)+".txt";
            grain_strain(:,graincount) = importdata(strain_filename);
        end
    end
    Avg_Stress_Sim = mean(grain_stress,2);
    Avg_Stress_Sim = [0;Avg_Stress_Sim];
    Avg_Strain_Sim = mean(grain_strain,2);
    Avg_Strain_Sim = [0;Avg_Strain_Sim];
    writematrix([Avg_Stress_Sim],"Average_Stress_11_"+...
        num2str(subfolder_num)+".txt")
    writematrix([Avg_Strain_Sim],"Average_Strain_11_"+...
        num2str(subfolder_num)+".txt")

    grain_stress = [];
    grain_strain = [];

    % neper_sim_time{subfolder_num} = importdata("Simulation_Time_"+...
    %     num2str(subfolder_num)+".txt");
end

% go back to main folder where this MATLAB script is saved.
cd(currentFolder)

%% Number of points in output file
% Simulations run using Abaqus will have the same number of time steps for
% each grain, but the number of steps between simulations is likely to be
% different. To have a uniform number of steps across all simulations to
% compare to a standard number of points will be chosen to interpolate
% over using the spline function so the average stress for each simulation
% is able to be neatly put into one file and compared with the mpDriver
% results along with experimental results.
%
% This is most easily done by using step time to spline things together.
% Each step is 1 second, the simulations for the First cycle are 2 steps
% which can be divided into many steps. For this analysis, time steps of
% 0.01 seconds were chosen meaning there will be 201 points between 0 and 2
% seconds to spline the information over.
time_spline_base = linspace(0,2,201);

%% Combine all data into one file with the same number of steps
% Use spline to get data into an appropriate format to compare with
% experimental and mpDriver data to analyze and use for Objective 3.

currentFolder = pwd;

col_counter = 1;
for subfolder_num = 1:subfolders
    cd(strcat(currentFolder,"\",num2str(subfolder_num),"\"));

    filename_stress = "Average_Stress_11_"+...
        num2str(subfolder_num)+".txt";
    Avg_Sim_Stress = importdata(filename_stress);

    filename_strain = "Average_Strain_11_"+...
        num2str(subfolder_num)+".txt";
    Avg_Sim_Strain = importdata(filename_strain);

    neper_sim_time = importdata(...
        "Simulation_Time_"+num2str(subfolder_num)+".txt");
    neper_sim_time = [0;neper_sim_time];

    Avg_Stress_All(:,col_counter) = spline(neper_sim_time,...
        Avg_Sim_Stress,...
        time_spline_base)';
    Avg_Strain_All(:,col_counter) = spline(neper_sim_time,...
        Avg_Sim_Strain,...
        time_spline_base)';

    col_counter = col_counter + 1;
    % Spline
end
Avg_Sim_Stress = [];
Avg_Sim_Strain = [];


cd(currentFolder)

writematrix([Avg_Stress_All],"Average_Stress_11_All_Abaqus_Sims.txt")
writematrix([Avg_Strain_All],"Average_Strain_11_All_Abaqus_Sims.txt")

