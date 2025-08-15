
% Created by Shutian Xue on 08/09/2025

% This script generates synthetic neural data based on Pitkow et al. (2015)
% Task: heading discrimination

% Inputs (the dataset)
%   - Stimulus: Discrete heading directions for each trial
%   - Neural responses (firing rates) of N neurons
%   - Behavioral choices (left/right) for each trial

% Outputs
%   - measurement error (var of estimator)
%   - estimated choice correlations (CC)
%   - neural thresholds (theta_k)
%   - behavioral thresholds (theta)
%   - tuning curves (f(s))
%   - and their corresponding uncertainties (sigmas)

clear all; clc; close all

% 0. Load sorted spiking data
load('spikeCount_LP_C00_R_N24_T281.mat')

% 0. Load behavioral data
load('behavior_responses.mat')
nTrials_withNOGO = length(behavior_responses);

str_region = 'LP'; 
str_contrast = 'C00'; 
str_priorCond = 'R';

spike_counts = double(spike_counts');
behavior_responses = double(behavior_responses);

% Filter out no-go trials
spike_counts = spike_counts(:, behavior_responses ~= 0);
behavior_responses = behavior_responses(behavior_responses ~= 0);

% 1. Extract parameters
[nNeurons, nTrials] = size(spike_counts);  % Number of neurons

assert(nTrials == length(behavior_responses))

% Print data set info
fprintf('Number of neurons: %d\n', nNeurons);
fprintf('Number of trials: %d/%d (NO-GO included)\n', nTrials, nTrials_withNOGO);

%% Calculate CC and neural threshold
CC_allN = nan(nNeurons, 1);
thresh_allN = nan(nNeurons, 1);
for iN=1:nNeurons
    % Calculate CC
    CC_allN(iN) = corr(spike_counts(iN, :)', behavior_responses');

    % Calculate neural threshold
    % should be divided by the mean firing rate; need the signal
    thresh_allN(iN) = std(spike_counts(iN, :)) / mean(spike_counts(iN, :));
end

% Visualize 
figure;
plot(1./thresh_allN, CC_allN, 'o')
xlabel('1 / Neural threshold');
ylabel('Choice correlation');
title('Neural threshold vs. Choice correlation');

% do robust regression
[r, p] = corr(CC_allN(thresh_allN>0), 1./thresh_allN(thresh_allN>0))

%% Compute covariance
SIGMA = cov(spike_counts');
% visualize
figure;
imagesc(SIGMA);
colorbar;
xlabel('Neuron');
ylabel('Neuron');
title(sprintf('Neural Response Covariance (Region=%s | Contrast=%.2f | %d Trials | Prior cond = %s)', str_region, str_contrast, nTrials, str_priorCond));