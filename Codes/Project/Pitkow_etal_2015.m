% This script simulates neural activity and computes the correlation between neurons
% Created by Shutian Xue on 08/08/2025
% Paper:   Pitkow et al. (2015)

clear all
clc

%% To reconstruct tuning curves, I will simulate and recover it fpr each stim level
nNeurons = 20; % Number of neurons
nStim = 100;
nTrials_perStim = 40;
nTrials = nStim * nTrials_perStim;

stim_all = linspace(0, 2*pi, nStim); % Stimulus values for each trial
pref_all = rand(nNeurons, 1) * 2 * pi;     % Preferred stimulus of each neuron

% Choose tuning diversity
tuningDiversity = 'uniform';
% tuningDiversity = 'lowDiversity';
% tuningDiversity = 'highDiversity';

% Parameters for tuning functions
kappa = 1; % von Mises concentration, +- .5
a = 24;    % Max firing rate, SD=20
b = 13;     % Baseline firing rate, 0 for 35% cells, b=13+-10 Hz for 65% cells

kappa_all = randn(nNeurons,1) * .5 + kappa;
a_all = randn(nNeurons,1) * 20 + a;
b_all = randn(nNeurons,1) * 10 + b;
b_all(1:round(0.35*nNeurons)) = 0; % Ensure baseline firing rate is non-negative

%% Calculate theoretical signal correlation coefficients
%  (just depends on N and params of the tuning fxn)
% Implement equation S13
Corr_signal = nan(nNeurons, nNeurons);

for iN1 = 1:nNeurons
    for iN2 = 1:nNeurons
        k1 = kappa_all(iN1);
        k2 = kappa_all(iN2);

        % Compute signal correlation
        numerator = besseli(0, sqrt(k1.^2 + k2.^2 + 2*k1*k2*cos(pref_all(iN1)-pref_all(iN2)))) - besseli(0, k1) * besseli(0, k2);
        denominator = (besseli(0, 2*k1) - besseli(0, k1)^2) * (besseli(0, 2*k2) - besseli(0, k2)^2);
        assert(denominator >= 0, 'ALERT: Value in sqrt is negative!');
        denominator = sqrt(denominator);

        Corr_signal(iN1, iN2) = numerator / denominator;

    end
end

figure;
imagesc(Corr_signal);
colorbar; axis square
xlabel('Neuron Index');
ylabel('Neuron Index');
title('Theoretical Correlation Coefficients (Eq S13)');

%% Simulate data
% Allocate firing rate matrix
% r_all = zeros(nNeurons, T);
% r_prime_all = r_all;
resp_perStim = cell(nStim, 1); % To store responses for each stimulus
f_theo_allStim = nan(nNeurons, nStim);

% Loop through each stimulus 
for iStim = 1:nStim
% Define the presented stimulus for this trial
    stim = stim_all(iStim);

    % Calculate the theoretical mean firing rate
    switch tuningDiversity
        case 'uniform'
            f_theo = b + a .* exp( kappa      .* (cos(stim - pref_all) - 1) );
        case 'lowDiversity'
            f_theo = b + a .* exp( kappa_all  .* (cos(stim - pref_all) - 1) );
        case 'naturalDiversity'
            f_theo = b_all + a_all .* exp( kappa_all .* (cos(stim - pref_all) - 1) );
    end

    % store
    f_theo_allStim(:, iStim) = f_theo;

    % Impose the correct signal correlation and calculate the theoretical covariance matrix
    Sigma_theo = (sqrt(f_theo) * sqrt(f_theo)').* Corr_signal;
    Sigma_theo = (Sigma_theo + Sigma_theo')/2;
    Sigma_theo = Sigma_theo + 1e-8*eye(nNeurons);

    % Sample data from Wishart distribution
    df = 2*nNeurons;
    A  = chol(Sigma_theo,'lower');
    Z  = randn(df, nNeurons);
    W  = Z * A';
    Sigma_sim = (W'*W)/df;

    % Simulate responses
    resp_perStim{iStim} = mvnrnd(f_theo, Sigma_sim, nTrials_perStim)';  % [N x T_per_s]

end % for iStim

% For the last stim, compare the sim and theo sigma
figure('Position', [100, 100, 1200, 400]);
subplot(1,3,1)
imagesc(Sigma_sim);
colorbar; axis square
title('Covariance Matrix (sampled)');
xlabel('Neuron Index');
ylabel('Neuron Index');

subplot(1,3,2)
imagesc(Sigma_theo);
colorbar; axis square
title('Covariance Matrix (theoretical)');
xlabel('Neuron Index');
ylabel('Neuron Index');

subplot(1,3,3), hold on
plot([min(Sigma_theo(:)), max(Sigma_theo(:))], [min(Sigma_theo(:)), max(Sigma_theo(:))], 'k-', 'LineWidth', 2);
plot(Sigma_theo(:), Sigma_sim(:), 'o');
xlabel('Theoretical');
ylabel('Sampled');
axis square;

xlim([-1 1]*max(abs([Sigma_theo(:); Sigma_sim(:)])));
ylim([-1 1]*max(abs([Sigma_theo(:); Sigma_sim(:)])));
title('Covariance Matrices Comparison');

%% Reconstruct tuning curves
f_rec_allStim  = nan(nNeurons, nStim);
for iStim = 1:nStim
    % Reconstruct tuning curve is the average response across trials
    f_rec_allStim(:, iStim) = mean(resp_perStim{iStim}, 2);
end

% Visualize: overlap the theoretical and reconstructed tuning curves
figure; hold on;
for iNeuron = 1:5%nNeurons
    plot(stim_all, f_theo_allStim(iNeuron, :).', 'k--', 'LineWidth', 1.5); % Theoretical
    plot(stim_all, f_rec_allStim(iNeuron, :).', 'r-', 'LineWidth', 1.5);   % Reconstructed
end
xlabel('Stimulus');
ylabel('Firing Rate');
title('Tuning Curves: Theoretical vs Reconstructed');
legend('Theoretical', 'Reconstructed');

