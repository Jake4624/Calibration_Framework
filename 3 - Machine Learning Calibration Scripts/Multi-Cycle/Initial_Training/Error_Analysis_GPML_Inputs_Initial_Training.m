% Jacob Rusch
% 16 July 2023
% 
% This script takes in the high- and low- fidelity average stress files
% from Objective 2 and calculates RMSE based on experimental data.
%
% The high- and low- fidelity average stress files should have an equal
% number of points per simulation to easily compare. The experimental data
% still needs to splined to have the same number of points at the same
% strains/times. This allows for the calculation of RMSE at each point if
% needed and makes for uniform data sizes.

clear;
close all;
clc;

% The time for the high- and low- fidelity analyses were adjusted to have
% the time be the following:
time_spline_base = linspace(0,10,1001);

% Step starts at time 0 which is the first indexed time
Step_Indices(1) = 1;
Step_Indices(2) = find(time_spline_base == 1);
Step_Indices(3) = find(time_spline_base == 2);

%% Get experimental data
% The csv file has three columns: "step time", strain, and stress,
% respectively. "step time" was added manually. See file for details.
%
% First cycle ends at row 161, only import the first cycle for this
% analysis
experimental_data = readtable("Wang2008_Fig4a.csv");

experimental_time = table2array(experimental_data(:,1));
experimental_strain = table2array(experimental_data(:,2));
experimental_stress = table2array(experimental_data(:,3));

stress_spline_exp = spline(experimental_time,experimental_stress,...
                    time_spline_base);
strain_spline_exp = spline(experimental_time,experimental_strain,...
                    time_spline_base);

%% Get high- and low- fidelity data
Abaqus_stress = importdata("Average_Stress_11_All_Abaqus_Sims.txt");
Abaqus_strain = importdata("Average_Strain_11_All_Abaqus_Sims.txt");
% Abaqus_time = importdata("Abaqus_Time.txt");

mpDriver_stress = importdata("Average_Stress_11_All_mpDriver_Sims.txt");
mpDriver_strain = importdata("Average_Strain_11_All_mpDriver_Sims.txt");

% Noise_stress = importdata("Average_Stress_11_All_mpDriver_Noise_Sims.txt");
% Noise_strain = importdata("Average_Strain_11_All_mpDriver_Noise_Sims.txt");

%% RMSE
% RMSE (or RMSD) is the root-mean-square error (or deviation)
% 
% Low Fidelity
num_lf_sims = size(mpDriver_stress,2);

for i=1:num_lf_sims
    RMSE_mpDriver(i) = sqrt(1/(length(time_spline_base)))*...
        sqrt(sum((stress_spline_exp' - mpDriver_stress(:,i)).^2));
end

% High Fidelity
num_hf_sims = size(Abaqus_stress,2);

for i=1:num_hf_sims
    RMSE_Abaqus(i) = sqrt(1/(length(time_spline_base)))*...
        sqrt(sum((stress_spline_exp' - Abaqus_stress(:,i)).^2));
end

% % mpDriver noise Comparison Sims
% for i=1:20
%     RMSE_Noise(i) = sqrt(1/(length(time_spline_base)))*...
%         sqrt(sum((stress_spline_exp' - Noise_stress(:,i)).^2));
% end

% Noise_Props = importdata("Noise_Props.txt");
% Abaqus_Sim_Equiv = importdata("Abaqus_Simnum_Equivalents.txt");
% 
% % Difference in RMSE between High Fidelity Sims and Low Fidelity sims run
% % using the same inputs
% RMSE_Noise_Diff = RMSE_Abaqus(Abaqus_Sim_Equiv) - RMSE_Noise;

%% Save to file
Neper_Props = importdata('Properties_Abaqus_Neper.txt');
mpDriver_Props = importdata('Properties_mpDriver.inc');

out_Neper = [Neper_Props,RMSE_Abaqus'];
out_mpDriver = [mpDriver_Props,RMSE_mpDriver'];

writematrix(out_Neper,'GPML_High_Fidelity.txt')
writematrix(out_mpDriver,'GPML_Low_Fidelity.txt')



%% Plots
figure
plot(RMSE_Abaqus)
figure
plot(RMSE_mpDriver)
figure
plot(experimental_strain,experimental_stress)
hold on
plot(mpDriver_strain(:,find(RMSE_mpDriver == min(RMSE_mpDriver)))...
    ,mpDriver_stress(:,find(RMSE_mpDriver == min(RMSE_mpDriver))))
plot(Abaqus_strain(:,find(RMSE_Abaqus == min(RMSE_Abaqus)))...
    ,Abaqus_stress(:,find(RMSE_Abaqus == min(RMSE_Abaqus))))
legend('Experimental Data','Low Fidelity','High Fidelity')

