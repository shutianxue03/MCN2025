% This script simulates neural activity and computes the correlation between neurons
% Created by Shutian Xue on 08/08/2025
% Paper:   Pitkow et al. (2015)

clear all
clc

%% Simulate 
nNeurons = 100; % Number of neurons
pref_all = rand(nNeurons, 1) * 2 * pi;     % Preferred stimulus of each neuron

nStim = 100; % number of stimulus levels
stim_all = linspace(0, 2*pi, nStim); % Stimulus levels

nTrials_perStim = 40; % Number of trials per stimulus level
nTrials = nStim * nTrials_perStim; % Total number of trials

% Choose tuning diversity
tuningDiversity = 'uniform';
tuningDiversity = 'lowDiversity';
% tuningDiversity = 'highDiversity';

% Define true tuning functions
kappa = 1; % von Mises concentration, +- .5
a = 24;    % Max firing rate, SD=20
b = 13;     % Baseline firing rate, 0 for 35% cells, b=13+-10 Hz for 65% cells

kappa_all = randn(nNeurons,1) * .5 + kappa;
a_all = randn(nNeurons,1) * 20 + a;
b_all = randn(nNeurons,1) * 10 + b;
b_all(1:round(0.35*nNeurons)) = 0; % Ensure baseline firing rate is non-negative

%% Calculate theoretical signal correlation coefficients [Eq S13]
Corr_signal = nan(nNeurons, nNeurons);

for iN1 = 1:nNeurons
    for iN2 = 1:nNeurons

        % Define the tuning parameter of each neurons (only kappa matters)
        kappa1 = kappa_all(iN1);
        kappa2 = kappa_all(iN2);

        % Compute signal correlation [Equation S13]
        numerator = besseli(0, sqrt(kappa1.^2 + kappa2.^2 + 2*kappa1*kappa2*cos(pref_all(iN1)-pref_all(iN2)))) - besseli(0, kappa1) * besseli(0, kappa2);
        denominator = (besseli(0, 2*kappa1) - besseli(0, kappa1)^2) * (besseli(0, 2*kappa2) - besseli(0, kappa2)^2);
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
title('Theoretical signal correlation coefficients (Eq S13)');

%% Simulate data
% Allocate firing rate matrix
resp_perStim_sim = cell(nStim, 1); % To store responses for each stimulus
f_theo_allStim = nan(nNeurons, nStim);
f_dev_theo_allStim = nan(nNeurons, nStim);
fisher_allStim = nan(nStim, 1);

% Loop through each stimulus
for iStim = 1:nStim
    % Define the presented stimulus for this trial
    stim = stim_all(iStim);

    % Calculate the theoretical mean firing rate
    switch tuningDiversity
        case 'uniform'
            f_theo = b + a .* exp( kappa      .* (cos(stim - pref_all) - 1) );
            f_dev_theo = -a*exp(kappa .* (cos(stim - pref_all) - 1)) .*kappa .* sin(stim - pref_all);
        case 'lowDiversity'
            f_theo = b + a .* exp( kappa_all  .* (cos(stim - pref_all) - 1) );
            f_dev_theo = -a .* exp(kappa_all .* (cos(stim - pref_all) - 1)) .* kappa_all .* sin(stim - pref_all);
        case 'naturalDiversity'
            f_theo = b_all + a_all .* exp( kappa_all .* (cos(stim - pref_all) - 1) );
            f_dev_theo = -a_all .* exp(kappa_all .* (cos(stim - pref_all) - 1)) .* kappa_all .* sin(stim - pref_all);
    end

    % store tuning curve and its derivative
    f_theo_allStim(:, iStim) = f_theo;
    f_dev_theo_allStim(:, iStim) = f_dev_theo;

    % Generate a mean covariance matrix by imposing the signal correlation
    cov_mean_theo = (sqrt(f_theo) * sqrt(f_theo)').* Corr_signal;
    % cov_mean_theo = (cov_mean_theo + cov_mean_theo')/2;
    % cov_mean_theo = cov_mean_theo + 1e-8*eye(nNeurons);

    % Sample data from Wishart distribution
    df = 2*nNeurons;
    cov_sim = wishrnd(cov_mean_theo,df)/df;

    % Simulate responses
    resp_perStim_sim{iStim} = mvnrnd(f_theo, cov_sim, nTrials_perStim)';  % [N x T_per_s]

    % Calculate response covariance matrix
    cov_resp_sim = cov(resp_perStim_sim{iStim}'); % need to transpose; no need to center before using cov()

    % Calculate Fisher information (Eq S16)
    fisher_allStim(iStim) = 1/(f_dev_theo_allStim(:, iStim).' * inv(cov_resp_sim) * f_dev_theo_allStim(:, iStim));

    % Calcualte the variance of an estimator based on the response of the k-th neuron (Eq S17)
    % sigma_est_k = var(resp_perStim_sim{iStim})* ()

    % Check the alignment between noise and signal
    [eigenvectors, eigenvalues] = eig(cov_resp_sim);
    eigenvalues = diag(eigenvalues);
    
    % Computing the squared cosine of the angle between each PC and the signal vector f'
    % Value > 1: the noise is along the signal direction, which is bed for coding
    %%%%% This analysis makes sense, but where in the paper does it show? %%%%
    squared_cosines = zeros(size(eigenvalues));
    for iPC = 1:length(eigenvalues)
        squared_cosines(iPC) = cos(eigenvectors(:, iPC).' * f_dev_theo_allStim(:, iStim))^2;
    end

    % Select a prefered direction, which maximizes f'(S), to maximize sensitivity
    s_pref = max(f_dev_theo);
        switch tuningDiversity
        case 'uniform'
            f_theo = b + a .* exp( kappa      .* (cos(stim - s_pref) - 1) );
            f_dev_theo = -a*exp(kappa .* (cos(stim - s_pref) - 1)) .*kappa .* sin(stim - s_pref);
        case 'lowDiversity'
            f_theo = b + a .* exp( kappa_all  .* (cos(stim - s_pref) - 1) );
            f_dev_theo = -a .* exp(kappa_all .* (cos(stim - s_pref) - 1)) .* kappa_all .* sin(stim - s_pref);
        case 'naturalDiversity'
            f_theo = b_all + a_all .* exp( kappa_all .* (cos(stim - s_pref) - 1) );
            f_dev_theo = -a_all .* exp(kappa_all .* (cos(stim - s_pref) - 1)) .* kappa_all .* sin(stim - s_pref);
    end

    % Generate a mean covariance matrix by imposing the signal correlation
    cov_mean_theo = (sqrt(f_theo) * sqrt(f_theo)').* Corr_signal;
    % cov_mean_theo = (cov_mean_theo + cov_mean_theo')/2;
    % cov_mean_theo = cov_mean_theo + 1e-8*eye(nNeurons);

    % Sample data from Wishart distribution
    df = 2*nNeurons;
    cov_sim = wishrnd(cov_mean_theo,df)/df;

    % Simulate responses
    % resp_perStim_sim{iStim} = mvnrnd(f_theo, cov_sim, nTrials_perStim)';  % [N x T_per_s]


end % for iStim

% For the last stim, compare the sim and theo sigma
figure('Position', [100, 100, 1200, 400]);
subplot(1,3,1)
imagesc(cov_sim);
colorbar; axis square
title('Covariance Matrix (sampled)');
xlabel('Neuron Index');
ylabel('Neuron Index');

subplot(1,3,2)
imagesc(cov_mean_theo);
colorbar; axis square
title('Covariance Matrix (theoretical)');
xlabel('Neuron Index');
ylabel('Neuron Index');

subplot(1,3,3), hold on
plot([min(cov_mean_theo(:)), max(cov_mean_theo(:))], [min(cov_mean_theo(:)), max(cov_mean_theo(:))], 'k-', 'LineWidth', 2);
plot(cov_mean_theo(:), cov_sim(:), 'o');
xlabel('Theoretical');
ylabel('Sampled');
axis square;

xlim([-1 1]*max(abs([cov_mean_theo(:); cov_sim(:)])));
ylim([-1 1]*max(abs([cov_mean_theo(:); cov_sim(:)])));
title('Covariance Matrices Comparison');

%% Reconstruct tuning curves
f_rec_allStim  = nan(nNeurons, nStim);
for iStim = 1:nStim
    % Reconstruct tuning curve is the average response across trials
    f_rec_allStim(:, iStim) = mean(resp_perStim_sim{iStim}, 2);
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

%% Calculate noise covariance matrix from simulated data
cov_noise_sim = cov(resp_perStim_sim{iStim}');