clear all
clc
%% Params
rng(1);
nNeurons = 100;
pref_all = rand(nNeurons,1)*2*pi;   % each neuron’s preferred angle

nStim = 100;
stim_all = linspace(0, 2*pi, nStim).';  % column

nTrials_perStim = 40;

% Tuning diversity
tuningDiversity = 'lowDiversity';  % 'uniform' | 'lowDiversity' | 'naturalDiversity'

% von Mises params
kappa = 1;
a = 24;
b = 13;

kappa_all = max(0, randn(nNeurons,1)*.5 + kappa);
a_all     = max(0, randn(nNeurons,1)*20 + a);
b_all     = max(0, randn(nNeurons,1)*10 + b);
b_all(1:round(0.35*nNeurons)) = 0;

% Define correlation strength (S11)
c0 = 0.15;  % 0.1–0.5 ( Chen et al., 2013; Liu et al., 2013).

%% Signal correlation (Eq S13)
Corr_signal = nan(nNeurons, nNeurons);
for i=1:nNeurons
    for j=1:nNeurons
        k1 = kappa_all(i); k2 = kappa_all(j);
        dth = pref_all(i)-pref_all(j);
        numerator = besseli(0, sqrt(k1.^2 + k2.^2 + 2*k1*k2*cos(dth))) - besseli(0,k1)*besseli(0,k2);
        denominator = (besseli(0,2*k1) - besseli(0,k1)^2) * (besseli(0,2*k2) - besseli(0,k2)^2);
        denominator = sqrt(max(denominator, 0));  % guard tiny negative due to fp error
        Corr_signal(i,j) = numerator / denominator;
    end
end

% Numerically enforce diag=1 and symmetry
% because correlation matrices must be symmetric and have unit diagonal by definition.
Corr_signal_ = Corr_signal;
Corr_signal_(1:nNeurons+1:end) = 1; % for diagonal = 1
Corr_signal_ = (Corr_signal_ + Corr_signal_')/2; % for symmetry

% ensure Corr_signal_ not too different from Corr_signal
assert(max(abs(Corr_signal_(:) - Corr_signal(:))) < 1e-3, 'ALERT: Corr_signal_ is too different from Corr_signal!');

Corr_signal = Corr_signal_;  % overwrite

% Obtain noise correlation matrix (R) by mixing with identity (Eq S11)
R = (1-c0)*eye(nNeurons) + c0*Corr_signal_;

%% Containers
resp_perStim_sim = cell(nStim,1);   % {stim} [N x T_perStim]
f_theo_allStim   = nan(nNeurons, nStim);
fprime_allStim   = nan(nNeurons, nStim);

%% Helper: build tuning curve at any stimulus level (s)
% see tuning_at.m

%% Simulate
df = 2*nNeurons;  % define Wishart degrees of freedom (S14)

% Loop through stimuli
for iStim = 1:nStim
    s = stim_all(iStim);

    % Derive tuning function and the derivative
    [f_s, fp_s] = tuning_at(tuningDiversity, s, pref_all, a, b, kappa, a_all, b_all, kappa_all);
    
    % Store
    f_theo_allStim(:,iStim) = f_s;
    fprime_allStim(:,iStim) = fp_s;

    % Obtain mean covariance at this s (the in-text eq above Eq S14)
    D  = diag(sqrt(f_s));
    Sig_bar = D * R * D;
    Sig_bar = (Sig_bar + Sig_bar')/2;
    Sig_bar = Sig_bar + 1e-8*eye(nNeurons);

    % sample covariance via Wishart, then re-symmetrize
    Sig = wishrnd(Sig_bar, df) / df;
    Sig = (Sig + Sig')/2;

    % draw Gaussian responses (Gaussian-Poisson approximation)
    X = mvnrnd(f_s, Sig, nTrials_perStim).';  % [N x T]
    resp_perStim_sim{iStim} = X;
end

%% Example: Fisher information at a reference s0 (linear FI J = f' Q^{-1} f')
s0 = 0;  % choose your reference
[f0, fp0] = tuning_at(s0);
Q0 = diag(sqrt(f0)) * R * diag(sqrt(f0));   % covariance at s0
J0 = fp0' * (Q0 \ fp0);   % linear Fisher information at s0
fprintf('Linear Fisher info at s0=%.2f rad: J = %.3f\n', s0, J0);