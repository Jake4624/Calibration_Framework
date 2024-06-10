% Jacob Rusch
% 11 May 2023
%
% Analyzes the average time series stress curves produced by
% Average_Stress_Each_Sim.m produces to pull out relevant data about the
% curve and store in a file to use in the statistical analysis.
%
clear;
close all;
clc;
%
%% Relevant Information
% There were 8 steps for the mpDriver simulations run for Objective 1, can
% separate them into 4 cycles and analyze each cycle
num_steps = 8;
% Current working directory
currentFolder = pwd;
%
%% Analysis
% The simulations were run so that each second was a loading or unloading
% step. E.g., from 0-1 seconds was loading, from 1-2 sec was unloading,
% etc. and therefore we know where the cycles start and end. This data has
% 8 steps so the data goes from 1-8 seconds (due to how mpDriver was set up
% the data actually stops at 7.9 seconds. The data were already generated
% and mostly analyzed so redoing it would not have been advantageous and it
% does not affect the overall findings or goal of this objctive anyway.)
%
file_data = importdata("Average_Stress_Each_Sim.txt");
time = file_data(:,1);
stress = file_data(:,2:end);
%
% Need to remove data where Theta_T is greater than Theta_Ref_High as that
% data is not valid. See equation for driving force for transformation.
% remove_sims = ...
%     importdata("Sims_to_Exclude.txt");
% stress(:,[remove_sims']) = [];
%
% Find the indices for each step. Note that every cycle consists of two 
% steps, a loading step and unloading step that are 1 second each.
% Save beginning time so analysis is easier. This is the same for the
% baseline data.
%%
Step_Indices(1) = 1;
for i = 1:num_steps-1
    Step_Indices(i+1) = find(time==i);
end
Step_Indices = [Step_Indices,length(time)];
Cycle_Indices = Step_Indices(1:2:end);
%
% ischange command finds abrupt changes in the data. Can possibly be used
% to detect the transformation stress and also the maximum and minimum
% stresses in the simulations.
% ischange command does not work that well. Use max/min functions to find
% peak stress values for each step.
%
for sim_count = 1:length(stress)
    for i = 2:length(Cycle_Indices)
        max_stress_each_step(i-1,sim_count) = max(...
            stress([Cycle_Indices(i-1):Cycle_Indices(i)],sim_count));
        min_stress_each_step(i-1,sim_count) = min(...
            stress([Cycle_Indices(i-1):Cycle_Indices(i)],sim_count));
    end
    % Difference in max/min stress between first cycle and last cycle
    max_cycle_diff(sim_count)= max_stress_each_step(1,sim_count) - ...
        max_stress_each_step(end,sim_count);
    min_cycle_diff(sim_count)= max_stress_each_step(1,sim_count) - ...
        max_stress_each_step(end,sim_count);
end
% 
% trapz can be used to find the area under the stress curves for each
% loading and unloading step to use in the ANOVA study.
for sim_count = 1:length(stress)
    for i = 2:length(Cycle_Indices)
        step_curve_areas(i-1,sim_count) = trapz(...
            time([Cycle_Indices(i-1):Cycle_Indices(i)]),...
            stress([Cycle_Indices(i-1):Cycle_Indices(i)],sim_count));
    end
    % Difference in areas between first cycle and last cycle
    area_diff(sim_count)= step_curve_areas(1,sim_count) - ...
        step_curve_areas(end,sim_count);
end

%% Baseline data comparison
% 
baseline_mp_data = importdata("Average_Baseline_Stress.txt");
baseline_mp_time = baseline_mp_data(:,1);
baseline_mp_stress = baseline_mp_data(:,2);

% Baseline data analysis
for i = 2:length(Cycle_Indices)
    max_stress_mp_baseline(i-1) = max(...
        baseline_mp_stress([Cycle_Indices(i-1):Cycle_Indices(i)]));
    min_stress_mp_baseline(i-1) = min(...
        baseline_mp_stress([Cycle_Indices(i-1):Cycle_Indices(i)]));
end
% Difference in max/min stress between first cycle and last cycle
max_cycle_mp_baseline_diff = max_stress_mp_baseline(1) - ...
    max_stress_mp_baseline(end);
min_cycle_mp_baseline_diff = min_stress_mp_baseline(1) - ...
    min_stress_mp_baseline(end);

% Difference between baseline sim and mp sims
for sim_count = 1:length(stress)
    max_stress_mp_baseline_diff(i-1,sim_count) = ...
        max_stress_mp_baseline(1) - max_stress_each_step(1,sim_count);
    min_stress_mp_baseline_diff(i-1,sim_count) = ...
        min_stress_mp_baseline(1) - min_stress_each_step(1,sim_count);

    % Difference in max/min stress between first cycle and last cycle
    max_cycle_mp_baseline_v_mp_diff(sim_count)= max_cycle_mp_baseline_diff - ...
        max_cycle_diff(sim_count);
    min_cycle_mp_baseline_v_mp_diff(sim_count)= min_cycle_mp_baseline_diff - ...
        min_cycle_diff(sim_count);
end
% 
% trapz can be used to find the area under the stress curves for each
% loading and unloading step to use in the ANOVA study.
for i = 2:length(Cycle_Indices)
    step_mp_baseline_curve_areas(i-1) = trapz(...
        time([Cycle_Indices(i-1):Cycle_Indices(i)]),...
        baseline_mp_stress([Cycle_Indices(i-1):Cycle_Indices(i)]));
end
% Difference in areas between first cycle and last cycle
area_diff_mp_baseline= step_mp_baseline_curve_areas(1) - ...
    step_mp_baseline_curve_areas(end);

% Difference in areas between baseline sim and mp sims
for sim_count = 1:length(stress)
    for i = 1:length(step_mp_baseline_curve_areas)
        baseline_v_mp_area_diff_each_step(i,sim_count) = ...
            step_mp_baseline_curve_areas(i) - step_curve_areas(i,sim_count);
    end
    area_diff_baseline_v_mp_diff(sim_count) = area_diff_mp_baseline - ...
        area_diff(sim_count);
end

%% Comparing differences in mp sims to Abaqus sims
% Need to get Abaqus data
% Get path for current folder
%
comparing_section_flag = 1;
if comparing_section_flag == 1

cd(strcat(currentFolder,"\Abaqus_Comparison_Sims\"));

% Get average baseline stress and average values for each sim
baseline_abaqus_data = importdata("Average_Baseline_Stress_Abaqus.txt");
baseline_abaqus_time = baseline_abaqus_data(:,1);
baseline_abaqus_stress = baseline_abaqus_data(:,2);
% 
% Abaqus Simulation Data
% Columns 
file_data = importdata("Average_Stress_Each_Sim_Abaqus.txt");
abaqus_time = file_data(:,1);
abaqus_stress = file_data(:,2:end);
% 
file_data = importdata("Abaqus_Sim_Random_Props.txt");
% Sims to remove
% remove_abaqus_sims = [8,11,14,15];
% file_data(remove_abaqus_sims,:) = [];
% First entry is 0 for the baseline sim, need to skip that.
simulation_numbers = file_data([2:end],1);
%
% Find the indices for each step. Note that every cycle consists of two 
% steps, a loading step and unloading step that are 1 second each.
% Save beginning time so analysis is easier. This is the same for the
% baseline data.
Abaqus_Step_Indices = [];
Abaqus_Step_Indices(1) = 1;
for i = 1:num_steps-1
    Abaqus_Step_Indices(i+1) = find(abaqus_time==i);
end
Abaqus_Step_Indices = [Abaqus_Step_Indices,length(abaqus_time)];
Abaqus_Cycle_Indices = [];
Abaqus_Cycle_Indices = Abaqus_Step_Indices(1:2:end);
% 

max_stress_each_step_abaqus = [];
min_stress_each_step_abaqus = [];
max_cycle_diff_abaqus = [];
min_cycle_diff_abaqus = [];
for sim_count = 1:length(simulation_numbers)
    % simulation 0 (baseline abaqus sim) is part of the numbers and needs 
    % to be skipped
    for i = 2:length(Abaqus_Cycle_Indices)
        max_stress_each_step_abaqus(i-1,sim_count) = max(...
            abaqus_stress([Abaqus_Cycle_Indices(i-1):Abaqus_Cycle_Indices(i)],sim_count));
        min_stress_each_step_abaqus(i-1,sim_count) = min(...
            abaqus_stress([Abaqus_Cycle_Indices(i-1):Abaqus_Cycle_Indices(i)],sim_count));
    end
    % Difference in max/min stress between first cycle and last cycle
    max_cycle_diff_abaqus(sim_count)= max_stress_each_step_abaqus(1,sim_count) - ...
        max_stress_each_step_abaqus(end,sim_count);
    min_cycle_diff_abaqus(sim_count)= min_stress_each_step_abaqus(1,sim_count) - ...
        min_stress_each_step_abaqus(end,sim_count);
end


max_stress_mp_v_abaqus_each_cycle = [];
min_stress_mp_v_abaqus_each_cycle = [];
max_stress_mp_v_abaqus_diff = [];
min_stress_mp_v_abaqus_diff = [];
% Compare max stress values between high and low fidelity sims
for sim_count = 1:length(simulation_numbers)
    for i = 2:length(Cycle_Indices)
        max_stress_mp_v_abaqus_each_cycle(sim_count) = ...
            max_stress_each_step(i-1,simulation_numbers(sim_count)) - ...
            max_stress_each_step_abaqus(i-1,sim_count);
        min_stress_mp_v_abaqus_each_cycle(sim_count) = ...
            min_stress_each_step(i-1,simulation_numbers(sim_count)) - ...
            min_stress_each_step_abaqus(i-1,sim_count);
    end
    max_stress_mp_v_abaqus_diff(sim_count) = ...
        max_cycle_diff(simulation_numbers(sim_count)) - ...
        max_cycle_diff_abaqus(sim_count);
    min_stress_mp_v_abaqus_diff(sim_count) = ...
        min_cycle_diff(simulation_numbers(sim_count)) - ...
        min_cycle_diff_abaqus(sim_count);
end

% for sim_count = 1:length(simulation_numbers)
for sim_count = 1:2
    figure
    plot(abaqus_time,abaqus_stress(:,sim_count))
    hold on
    plot(time,stress(:,simulation_numbers(sim_count)))
    hold off
end

% go back to main folder where this MATLAB script is saved.
cd(currentFolder)
end

%% Histograms of Residuals to Check Anova Normality Assumption
% Can use a One-sample Kolmogrov-Smirnov test which returns a test decision
% for the null hypothesis that the data in a vector x comes from a standard
% normal distribution, against the alternative that it does not come from
% such a distribution. The result is 1 if the test rejects the null
% hypothesis at the 5% significance level, or 0 otherwise
% Syntax: h = kstest(x)
% if h=0 the data is considered normally distributed
% if h=1 the data is not normally distributed
% 
% Normal distributions have the following characteristics:
%  - symmetric bell shape 
%  - mean and median are equal; both located at the center of the distribution 
%  - ~68 percent of the data falls within 1 standard deviation of the mean 
%  - ~95 percent of the data falls within 2 standard deviations of the mean
%  - ~99.7 percent of the data falls within 3 standard deviations of the mean
% 
% Count the number of elements which differ from the mean value of each
% residual by 1, 2, and 3 standard deviations to see if the data can ba
% analyzed using an ANOVA or if a different statistical method such as
% Friedman's test needs to be used to analyze the data.
% 
% Max Stress First Cycle and Last Cycle Counts
Max_Stress_Count(1,1) = sum(abs(max_stress_each_step(1,:)-...
    mean(max_stress_each_step(1,:)))<=1*std(max_stress_each_step(1,:)))/...
    length(max_stress_each_step(1,:));
Max_Stress_Count(1,2) = sum(abs(max_stress_each_step(1,:)-...
    mean(max_stress_each_step(1,:)))<=2*std(max_stress_each_step(1,:)))/...
    length(max_stress_each_step(1,:));
Max_Stress_Count(1,3) = sum(abs(max_stress_each_step(1,:)-...
    mean(max_stress_each_step(1,:)))<=3*std(max_stress_each_step(1,:)))/...
    length(max_stress_each_step(1,:));
Max_Stress_Mean_Median_Diff(1) = abs(mean(max_stress_each_step(1,:))- ...
    median(max_stress_each_step(1,:)));

Max_Stress_Count(2,1) = sum(abs(max_stress_each_step(end,:)-...
    mean(max_stress_each_step(end,:)))<=1*std(max_stress_each_step(end,:)))/...
    length(max_stress_each_step(end,:));
Max_Stress_Count(2,2) = sum(abs(max_stress_each_step(end,:)-...
    mean(max_stress_each_step(end,:)))<=2*std(max_stress_each_step(end,:)))/...
    length(max_stress_each_step(end,:));
Max_Stress_Count(2,3) = sum(abs(max_stress_each_step(1,:)-...
    mean(max_stress_each_step(end,:)))<=3*std(max_stress_each_step(end,:)))/...
    length(max_stress_each_step(end,:));
Max_Stress_Mean_Median_Diff(2) = abs(mean(max_stress_each_step(end,:))- ...
    median(max_stress_each_step(end,:)));

% Min Stress First Cycle and Last Cycle Counts
Min_Stress_Count(1,1) = sum(abs(min_stress_each_step(1,:)-...
    mean(min_stress_each_step(1,:)))<=1*std(min_stress_each_step(1,:)))/...
    length(min_stress_each_step(1,:));
Min_Stress_Count(1,2) = sum(abs(min_stress_each_step(1,:)-...
    mean(min_stress_each_step(1,:)))<=2*std(min_stress_each_step(1,:)))/...
    length(min_stress_each_step(1,:));
Min_Stress_Count(1,3) = sum(abs(min_stress_each_step(1,:)-...
    mean(min_stress_each_step(1,:)))<=3*std(min_stress_each_step(1,:)))/...
    length(min_stress_each_step(1,:));
Min_Stress_Mean_Median_Diff(1) = abs(mean(min_stress_each_step(1,:))- ...
    median(min_stress_each_step(1,:)));

Min_Stress_Count(2,1) = sum(abs(min_stress_each_step(end,:)-...
    mean(min_stress_each_step(end,:)))<=1*std(min_stress_each_step(end,:)))/...
    length(min_stress_each_step(end,:));
Min_Stress_Count(2,2) = sum(abs(min_stress_each_step(end,:)-...
    mean(min_stress_each_step(end,:)))<=2*std(min_stress_each_step(end,:)))/...
    length(min_stress_each_step(end,:));
Min_Stress_Count(2,3) = sum(abs(min_stress_each_step(end,:)-...
    mean(min_stress_each_step(end,:)))<=3*std(min_stress_each_step(end,:)))/...
    length(min_stress_each_step(end,:));
Min_Stress_Mean_Median_Diff(2) = abs(mean(min_stress_each_step(end,:))- ...
    median(min_stress_each_step(end,:)));

% Area Counts
Area_Count(1,1) = sum(abs(step_curve_areas(1,:)-...
    mean(step_curve_areas(1,:)))<=1*std(step_curve_areas(1,:)))/...
    length(step_curve_areas(1,:));
Area_Count(1,2) = sum(abs(step_curve_areas(1,:)-...
    mean(step_curve_areas(1,:)))<=2*std(step_curve_areas(1,:)))/...
    length(step_curve_areas(1,:));
Area_Count(1,3) = sum(abs(step_curve_areas(1,:)-...
    mean(step_curve_areas(1,:)))<=3*std(step_curve_areas(1,:)))/...
    length(step_curve_areas(1,:));
Area_Mean_Median_Diff(1) = abs(mean(step_curve_areas(1,:))- ...
    median(step_curve_areas(1,:)));

Area_Count(2,1) = sum(abs(step_curve_areas(end,:)-...
    mean(step_curve_areas(end,:)))<=1*std(step_curve_areas(end,:)))/...
    length(step_curve_areas(end,:));
Area_Count(2,2) = sum(abs(step_curve_areas(end,:)-...
    mean(step_curve_areas(end,:)))<=2*std(step_curve_areas(end,:)))/...
    length(step_curve_areas(end,:));
Area_Count(2,3) = sum(abs(step_curve_areas(end,:)-...
    mean(step_curve_areas(end,:)))<=3*std(step_curve_areas(end,:)))/...
    length(step_curve_areas(end,:));
Area_Mean_Median_Diff(2) = abs(mean(step_curve_areas(end,:))- ...
    median(step_curve_areas(end,:)));

% Differences in Max/Min Stress and Area Counts
Max_Diff_Count(1) = sum(abs(max_cycle_diff-...
    mean(max_cycle_diff))<=1*std(max_cycle_diff))/...
    length(max_cycle_diff);
Max_Diff_Count(2) = sum(abs(max_cycle_diff-...
    mean(max_cycle_diff))<=2*std(max_cycle_diff))/...
    length(max_cycle_diff);
Max_Diff_Count(3) = sum(abs(max_cycle_diff-...
    mean(max_cycle_diff))<=3*std(max_cycle_diff))/...
    length(max_cycle_diff);
Max_Stress_Diff_Mean_Median_Diff = abs(mean(max_cycle_diff)- ...
    median(max_cycle_diff));

Min_Diff_Count(1) = sum(abs(min_cycle_diff-...
    mean(min_cycle_diff))<=1*std(min_cycle_diff))/...
    length(min_cycle_diff);
Min_Diff_Count(2) = sum(abs(min_cycle_diff-...
    mean(min_cycle_diff))<=2*std(min_cycle_diff))/...
    length(min_cycle_diff);
Min_Diff_Count(3) = sum(abs(min_cycle_diff-...
    mean(min_cycle_diff))<=3*std(min_cycle_diff))/...
    length(min_cycle_diff);
Min_Stress_Diff_Mean_Median_Diff = abs(mean(min_cycle_diff)- ...
    median(min_cycle_diff));

Area_Diff_Count(1) = sum(abs(area_diff-...
    mean(area_diff))<=1*std(area_diff))/...
    length(area_diff);
Area_Diff_Count(2) = sum(abs(area_diff-...
    mean(area_diff))<=2*std(area_diff))/...
    length(area_diff);
Area_Diff_Count(3) = sum(abs(area_diff-...
    mean(area_diff))<=3*std(area_diff))/...
    length(area_diff);
Area_Diff_Mean_Median_Diff = abs(mean(area_diff)-median(area_diff));

%% Display Values
% Display the values for each assumption to check them
disp('Max_Stress_Count=')
disp(Max_Stress_Count)
disp('Max_Stress_Mean_Median_Diff=')
disp(Max_Stress_Mean_Median_Diff)

disp('Min_Stress_Count=')
disp(Min_Stress_Count)
disp('Min_Stress_Mean_Median_Diff=')
disp(Min_Stress_Mean_Median_Diff)

disp('Area_Count=')
disp(Area_Count)
disp('Area_Mean_Median_Diff=')
disp(Area_Mean_Median_Diff)

disp('Max_Diff_Count=')
disp(Max_Diff_Count)
disp('Max_Stress_Diff_Mean_Median_Diff=')
disp(Max_Stress_Diff_Mean_Median_Diff)

disp('Min_Diff_Count=')
disp(Min_Diff_Count)
disp('Min_Stress_Diff_Mean_Median_Diff=')
disp(Min_Stress_Diff_Mean_Median_Diff)

disp('Area_Diff_Count=')
disp(Area_Diff_Count)
disp('Area_Diff_Mean_Median_Diff=')
disp(Area_Diff_Mean_Median_Diff)

%% Historgram Plots
figure
subplot(1,3,1)
histogram(max_cycle_diff)
title('Difference Between Max Stress in First and Last Cycle')
ylabel('Difference in Max Stress (MPa)')

subplot(1,3,2)
histogram(min_cycle_diff)
title('Difference Between Min Stress in First and Last Cycle')
ylabel('Difference in Min Stress (MPa)')

subplot(1,3,3)
histogram(area_diff)
title('Difference Between Area Under Stress-Time Curve for First and Last Cycle')
ylabel('Difference in Area')
%% Histfit Figures
figure
subplot(1,3,1)
histfit(max_cycle_diff)
title('Difference Between Max Stress in First and Last Cycle')
ylabel('Difference in Max Stress (MPa)')

subplot(1,3,2)
histfit(min_cycle_diff)
title('Difference Between Min Stress in First and Last Cycle')
ylabel('Difference in Min Stress (MPa)')

subplot(1,3,3)
histfit(area_diff)
title('Difference Between Area Under Stress-Time Curve for First and Last Cycle')
ylabel('Difference in Area')

figure
subplot(2,3,1)
histfit(max_stress_each_step(1,:))
title('Max Stress in First Cycle')
ylabel('Max Stress (MPa)')

subplot(2,3,2)
histfit(max_stress_each_step(end,:))
title('Max Stress in Last Cycle')
ylabel('Max Stress (MPa)')

subplot(2,3,3)
histfit(min_stress_each_step(1,:))
title('Min Stress in First Cycle')
ylabel('Min Stress (MPa)')

subplot(2,3,4)
histfit(min_stress_each_step(end,:))
title('Min Stress in Last Cycle')
ylabel('Min Stress (MPa)')

subplot(2,3,5)
histfit(step_curve_areas(1,:))
title('Area of First Cycle')
ylabel('Area')

subplot(2,3,6)
histfit(step_curve_areas(end,:))
title('Area of Last Cycle')
ylabel('Area')


%% RMSE Stuff
file_data = importdata('Props_and_RMSE_Vals.txt');
% Need to remove sims
file_data([remove_sims'],:) = [];
props = file_data(:,[1:end-1]);
RMSE = file_data(:,end);
% 
% [h,p] = adtest(RMSE,'Distribution','weibull');

%% Anova
group = {props(:,9),props(:,11),props(:,12),props(:,13),props(:,18),...
    props(:,19),props(:,20),props(:,21),props(:,22),props(:,24),...
    props(:,25),props(:,26),props(:,27),props(:,28),props(:,29),...
    props(:,30)};
variables = {'f_c','Theta_T','Theta_Ref_Low','lambda_T','Am','S0',...
    'H0','SS','Ahard','Theta_Ref_High','Bsat','b1','N_GEN_F','d_ir',...
    'D_ir','mu'};

Area_Diff_Anova = anovan(area_diff,group,'model',2,'varnames',variables);
Max_Cycle_Diff_Anova = anovan(max_cycle_diff,group,'model',2, ...
    'varnames',variables);
Min_Cycle_Diff_Anova = anovan(min_cycle_diff,group,'model',2, ...
    'varnames',variables);

Curve_Area_First_Cycle_Anova = anovan(step_curve_areas(1,:),group, ...
    'model',2,'varnames',variables);
Curve_Area_Last_Cycle_Anova = anovan(step_curve_areas(end,:),group, ...
    'model',2,'varnames',variables);
Max_Stress_First_Cycle_Anova = anovan(max_stress_each_step(1,:),group, ...
    'model',2,'varnames',variables);
Max_Stress_Last_Cycle_Anova = anovan(max_stress_each_step(end,:),group, ...
    'model',2,'varnames',variables);
Min_Stress_First_Cycle_Anova = anovan(min_stress_each_step(1,:),group, ...
    'model',2,'varnames',variables);
Min_Stress_Last_Cycle_Anova = anovan(min_stress_each_step(end,:),group, ...
    'model',2,'varnames',variables);

RMSE_Anova = anovan(RMSE,group,'model',2,'varnames',variables);

all_anovas = [Area_Diff_Anova, Max_Cycle_Diff_Anova, ...
    Min_Cycle_Diff_Anova,Curve_Area_First_Cycle_Anova, ...
    Curve_Area_Last_Cycle_Anova, Max_Stress_First_Cycle_Anova,...
    Max_Stress_Last_Cycle_Anova,Min_Stress_First_Cycle_Anova,...
    Min_Stress_Last_Cycle_Anova,RMSE_Anova];


