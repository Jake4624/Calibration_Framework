% Jacob Rusch
% 17 May 2023
%
% Analyzes the average RMSE value for each mpDriver simulation
%
clear;
close all;
clc;
%
file_data = importdata('Props_and_RMSE_Vals.txt');
props = file_data(:,[1:end-1]);
RMSE = file_data(:,end);

[h,p] = adtest(RMSE,'Distribution','weibull')