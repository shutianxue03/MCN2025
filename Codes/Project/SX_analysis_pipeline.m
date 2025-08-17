
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

clear all; clc; close all; format compact

% 0. Set parameters
str_region = 'CP';
str_priorCond = 'bias0.5';
nNeurons = 819; % Number of selected neurons

% str_contrast_all = {'C0', 'C0625', 'C125', 'C25', 'C50', 'C100'};
contrast_all = [0, .0625, .125, .25, .5, 1];
nContrasts = length(str_contrast_all);

contrasts_allC = [];
spike_counts_allC = [];
behavior_responses_allC = [];

for iContrast = 1:nContrasts
    contrast = contrast_all(iContrast); % Convert string to numerical contrast value

    % 1a. Load sorted spiking data
    load(sprintf('spikeCount_%s_%s_%s_N%d_T%d.mat', str_region, str_contrast, str_priorCond, nNeurons, nTrials))
    load(sprintf('spikeCount_%s_%s_N%d_T%d.mat', str_region, str_priorCond, nNeurons, nTrials))
    spike_counts = double(spike_counts');

    % 1b. Load behavioral data
    load('behavior_responses.mat')
    behavior_responses = double(behavior_responses);

    % Filter out no-go trials
    nTrials_withNOGO = length(behavior_responses); assert(nTrials_withNOGO == nTrials);
    spike_counts = spike_counts(:, behavior_responses ~= 0); assert(size(spike_counts, 2) == nTrials_withNOGO);
    behavior_responses = behavior_responses(behavior_responses ~= 0);

    % Concatenate
    spike_counts_allC = [spike_counts_allC, spike_counts];
    behavior_responses_allC = [behavior_responses_allC, behavior_responses];
    contrasts_allC = [contrasts_allC, ones(1, nTrials) * contrast];
end

% Print data set info
fprintf('Number of neurons: %d\n', nNeurons);
fprintf('Number of trials: %d/%d (NO-GO included)\n', nTrials, nTrials_withNOGO);

%% Derive tuning function


%% Calculate CC and neural threshold
CC_allN = nan(nNeurons, nContrastst);
thresh_allN = nan(nNeurons, 1);
for iN=1:nNeurons
    for iContrast = 1:nContrasts
        % Calculate CC
        CC_allN(iN, iContrast) = corr(spike_counts(iN, :)', behavior_responses');

        % Calculate neural threshold
        % should be divided by the mean firing rate; need the signal
        thresh_allN(iN) = std(spike_counts(iN, :)) / mean(spike_counts(iN, :));
    end
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