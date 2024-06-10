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

%% Get new data from iteration
Abaqus_stress = importdata("Average_Stress_11_First_Iter_Abaqus_Sim.txt");
Abaqus_strain = importdata("Average_Strain_11_First_Iter_Abaqus_Sim.txt");
% ABAQUS exports logarithmic strain, need to convert it to engineering
% strain
Abaqus_strain  = exp(Abaqus_strain)-1;

%% RMSE
% RMSE (or RMSD) is the root-mean-square error (or deviation)
% 
num_hf_sims = size(Abaqus_stress,2);
% High Fidelity
for i=1:num_hf_sims
    RMSE_Abaqus_Iter(i) = sqrt(1/(length(time_spline_base)))*...
        sqrt(sum((stress_spline_exp' - Abaqus_stress(:,i)).^2));
end

%% Save to file
Neper_Iter_Props = importdata('Properties_Abaqus_Neper.txt');

out_Neper = [Neper_Iter_Props, RMSE_Abaqus_Iter'];

writematrix(out_Neper,'GPML_Iteration_1.txt')

%% Plots
figure
plot(Abaqus_strain(:,1),Abaqus_stress(:,1))
hold on
plot(experimental_strain,experimental_stress,'--')
% legend('Iteration 4 Mean','Iteration 4 Lower Bound','Wang 4a Experimental Data')
legend('Iteration 1 Mean Func','Wang 4a Experimental Data')
xlabel('Strain')
ylabel('Stress (MPa)')

figure
plot(Abaqus_strain(:,2),Abaqus_stress(:,2))
hold on
plot(experimental_strain,experimental_stress,'--')
% legend('Iteration 4 Mean','Iteration 4 Lower Bound','Wang 4a Experimental Data')
legend('Iteration 1 Lower Bound','Wang 4a Experimental Data')
xlabel('Strain')
ylabel('Stress (MPa)')
