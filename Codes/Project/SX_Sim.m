
% Created by Shutian Xue on 08/09/2025

% This script generates synthetic neural data based on Pitkow et al. (2015)
% Task: heading discrimination
% The data contains:
%   - Stimulus: Discrete heading directions for each trial
%   - Neural responses (firing rates) of N neurons
%   - Behavioral choices (left/right) for each trial

% Output of simulating virtual neurons: measurement error, estimated choice correlations, neural thresholds, tuning curves, and their corresponding uncertainties

% The thresholds and tuning data were used to predict choice correlations according to Equations 2 and 3 separately,

clear all; clc;

%% Parameters (adjust to match Pitkow et al. 2015)
nNeurons = 100;           % population size
nTrialsPerStim = 100;

% From data
% stimVals = [-6.4, -2.6, -1, 0, 1, 2.6, 6.4]/180*pi;
% nStimVals = length(stimVals);

% Manually set the stimulus values
nStimVals = 20;
stimVals = linspace(-pi/2, pi/2, nStimVals);
nTrials  = nTrialsPerStim * nStimVals;          % total trials

% Tuning function parameters
c0       = 0.2;           % proportionality constant
kappa    = 1;             % tuning width for von Mises
b = 0;
a = 20;

rng(1); % reproducibility

% Define stimulus sequence
stimSeq = repmat(stimVals, 1, nTrialsPerStim); % stimulus sequence
% randomize
stimSeq = stimSeq(randperm(length(stimSeq)));

%% 1. Construct tuning curves
% % define preferred stimuli (Eq. S9–S13 in supplement)
% s_pref_allNeurons = linspace(0, 2*pi, nNeurons)'; % evenly spaced tuning preferences
s_pref_allNeurons = linspace(-pi/2, pi/2, nNeurons)'; % evenly spaced tuning preferences

% Example tuning function: von Mises
fxn_tuning = @(s, s_pref) b + a * exp(kappa * (cos(s - s_pref) - 1)); % mean rates
fxn_tuning_dev = @(s, s_pref)  - a * exp(kappa * (cos(s - s_pref) - 1)) * kappa .* sin(s - s_pref); % derivative of the tuning fxn

% visualize tuning function (top) and derivative (bottom) of the first 10 neurons
figure;
subplot(2, 1, 1); hold on
for i = 1:10:nNeurons
    s_pref = s_pref_allNeurons(i);
    fplot(@(stimVals) fxn_tuning(stimVals, s_pref), [-pi/2, pi/2]);
end
xlim([-pi/2, pi/2]);
ylim([0, 25]);
ylabel('Mean Firing Rate (Hz)');
xlabel('Stimulus (rad)');

subplot(2, 1, 2); hold on
for i = 1:10:nNeurons
    s_pref = s_pref_allNeurons(i);
    fplot(@(stimVals) fxn_tuning_dev(stimVals, s_pref), [-pi/2, pi/2]);
end
xlim([-pi/2, pi/2]);
ylim([0, 25]);
ylabel('Derivative of Firing Rate (Hz/rad)');
xlabel('Stimulus (rad)');


%% 2. Derive theoretical signal correlation matrix (Eq. S13) and averaged noise correlation matrix R (Eq. S13)
Corr_signal = nan(nNeurons);
for i = 1:nNeurons
    for j = 1:nNeurons
        num = besseli(0, sqrt(kappa^2 + kappa^2 + 2*kappa*kappa*cos(s_pref_allNeurons(i)-s_pref_allNeurons(j)))) - besseli(0, kappa)^2;
        den = sqrt((besseli(0, 2*kappa) - besseli(0, kappa)^2)^2);
        Corr_signal(i,j) = num / den;
    end
end

% Derive averaged noise correlation matrix R (Eq. S11)
% Which is also the TRUE noise correlation matrix
R_bar = (1-c0)*eye(nNeurons) + c0*Corr_signal;

figure;
subplot(1,3,1)
imagesc(Corr_signal); colorbar; axis square; clim([-1, 1]);
title('Theoretical Signal Correlation Matrix');

subplot(1,3,2)
imagesc(R_bar); colorbar; axis square; clim([-1, 1]);
title('Averaged Noise Correlation Matrix (the TRUE noise corr matrix)');

%% 3. Simulate neural responses (Eq. 1 in main text)
respNeural   = nan(nNeurons, nTrials);
df = 2*nNeurons;  % define Wishart degrees of freedom (S14)

% evaluate derivative at s=0, which is the reference heading
stim_ref = zeros(size(stimSeq(1))); 

for t = 1:nTrials
    stimVal = stimSeq(t);

    % Exract tuning function
    f_s = fxn_tuning(stimVal, s_pref_allNeurons); % mean firing rates, nNeurons x 1

    % Create SIGMA_bar per trial
    SIGMA_bar_perTrial = nan(nNeurons, nNeurons);
    for i = 1:nNeurons
        for j = 1:nNeurons

            % Equation above Eq S14
            f_s_i = fxn_tuning(stimVal, s_pref_allNeurons(i)); % mean firing rates
            f_s_j = fxn_tuning(stimVal, s_pref_allNeurons(j)); % mean firing rates

            SIGMA_bar_perTrial(i, j) = sqrt(f_s_i*f_s_j)*R_bar(i, j);
        end
    end

    % Sample covariate matrix from the Wishart distribution
    SIGMA_perTrial = wishrnd(SIGMA_bar_perTrial, df) / df;

    % Sample neural responses from a multivariate normal distribution with defined mean and covariance
    respNeural(:,t) = mvnrnd(f_s, SIGMA_perTrial)';    % draw one population vector

    % Ensure non-negative firing rates
    respNeural(:,t) = max(0, respNeural(:,t));
end

% Visualize neural responses
figure;
imagesc(respNeural);
colorbar;
title('Simulated Neural Responses');
xlabel('Trial #');
ylabel('Neuron #');

%% 3b. Create the true covariance matrix (SIGMA_bar)
f_s0 = fxn_tuning(0, s_pref_allNeurons); % Mean firing rates at s=0

SIGMA_theo = nan(nNeurons, nNeurons);
for i = 1:nNeurons
    for j = 1:nNeurons
        SIGMA_theo(i, j) = sqrt(f_s0(i) * f_s0(j)) * R_bar(i, j);
    end
end

%% 4. Compute the optimal decoder weights (Eq. 3)
% Calculate the empirical covariance matrix
SIGMA_emp = cov(respNeural');

% Visualize empirical & theoretical noise covariance
figure('Position', [100, 100, 400, 300]);
subplot(1,2,1);
imagesc(SIGMA_emp);
colorbar; axis square;
xlabel('Neuron #');
title('Empirical Covariance (Sigma\_pooled)');

subplot(1,2,2);
imagesc(SIGMA_theo);
colorbar; axis square;
xlabel('Neuron #');
ylabel('Neuron #');
title('Theoretical Covariance (Sigma\_theo)');

% Derive the optimal decoder (from the equation above Eq 3)
f_s_dev = fxn_tuning_dev(stim_ref, s_pref_allNeurons);  % nNeurons x 1
w_opt_emp_unnorm = SIGMA_emp \ f_s_dev; % more accurate than w_opt = inv(SIGMA_pooled) * f_s_dev;
w_opt_theo_unnorm = SIGMA_theo \ f_s_dev; % more accurate than w_opt = inv(SIGMA_pooled) * f_s_dev;

% Normalize decoder weights to obtain unbiased decoder
w_opt_theo = w_opt_theo_unnorm / (w_opt_theo_unnorm' * f_s_dev);
w_opt_emp = w_opt_emp_unnorm / (w_opt_emp_unnorm' * f_s_dev);

%% 5. Compute the correlation-blind (factorial) suboptimal decoder weights
%A correlation-blind decoder assumes independence between the noise of different neurons, hence it only uses only the diagonal elements (variances) of the noise covariance matrix, ignoring the off-diagonal (covariance) terms.
var_emp = diag(SIGMA_emp);
var_theo = diag(SIGMA_theo);

w_cb_emp_unnorm =  f_s_dev./ var_emp;
w_cb_emp = w_cb_emp_unnorm / (w_cb_emp_unnorm' * f_s_dev);

w_cb_theo_unnorm =  f_s_dev./ var_theo;
w_cb_theo = w_cb_theo_unnorm / (w_cb_theo_unnorm' * f_s_dev);

% Visualize two decoders
figure('Position', [100, 100, 1200, 1200]);
subplot(2,2,1);
bar(w_cb_emp);
xlabel('Neuron #');
title('Correlation-blind Decoder Weights (Empirical)');

subplot(2,2,2);
bar(w_opt_emp);
xlabel('Neuron #');
title('Optimal Decoder Weights (Empirical)');

subplot(2,2,3);
bar(w_cb_theo);
xlabel('Neuron #');
title('Correlation-blind Decoder Weights (Theoretical)');

subplot(2,2,4);
bar(w_opt_theo);
xlabel('Neuron #');
title('Optimal Decoder Weights (Theoretical)');

%% 6. Implement both decoders and generate behavior, and fit PMFs
% Derive estimators (s_hat)
s_hat_opt = w_opt_emp' * respNeural;
s_hat_cb  = w_cb_emp'  * respNeural;

% Derive choices (1 for right, -1 for left)
choice_opt = sign(s_hat_opt);
choice_cb  = sign(s_hat_cb);

% Calculate p(Right|s)
pRight_opt = nan(nStimVals, 1);
pRight_cb = nan(nStimVals, 1);
for iStim = 1:nStimVals
    assert(sum(stimSeq == stimVals(iStim)) == nTrialsPerStim, 'ALERT: trial number is wrong')
    pRight_opt(iStim) = mean(choice_opt(stimSeq == stimVals(iStim))==1);
    pRight_cb(iStim) = mean(choice_cb(stimSeq == stimVals(iStim))==1);
end

% Fit accumulative gaussian functions to PMF
fxn_firCumGauss = @(params, x) 1/2 + 1/2*erf((x-params(1))/(params(2)*sqrt(2)));
params0 = [0, 0.1]; % initial parameters for intercept and threshold
beta_opt = nlinfit(stimVals, pRight_opt', fxn_firCumGauss, params0);
beta_cb = nlinfit(stimVals, pRight_cb', fxn_firCumGauss, params0);

% Visualize data and fitting psychometric function
figure;
hold on;
plot(stimVals, pRight_opt, 'bo-', 'DisplayName', 'Optimal Decoder');
plot(stimVals, fxn_firCumGauss(beta_opt, stimVals), 'b--', 'DisplayName', sprintf('Optimal Fit (threshold=%.2f)', beta_opt(2)));
plot(stimVals, pRight_cb, 'ro-', 'DisplayName', 'Correlation-blind Decoder');
plot(stimVals, fxn_firCumGauss(beta_cb, stimVals), 'r--', 'DisplayName', sprintf('Correlation-blind Fit (threshold=%.2f)', beta_cb(2)));
xlabel('Stimulus Value (s)');
ylabel('P(Right|s)');
title('Psychometric Function');
legend('show', 'location', 'best');

%% 6b. ROC Analysis for Decoder Performance
% This analysis measures how well the decoder can discriminate between
% leftward and rightward stimuli based on the continuous decoder output s_hat.

% Isolate trials for leftward and rightward stimuli
indLEFT = stimSeq < 0;
indRIGHT = stimSeq > 0;

s_hat_opt_left = s_hat_opt(indLEFT);
s_hat_opt_right = s_hat_opt(indRIGHT);

s_hat_cb_left = s_hat_cb(indLEFT);
s_hat_cb_right = s_hat_cb(indRIGHT);

% Define a range of decision thresholds to test
thresholds = linspace(min([s_hat_opt, s_hat_cb]), max([s_hat_opt, s_hat_cb]), 1e3);

% Preallocate arrays for pHit and pFA
pHit_opt = nan(size(thresholds));
pFA_opt = nan(size(thresholds));
pHit_cb = nan(size(thresholds));
pFA_cb = nan(size(thresholds));

% Calculate Hit and False Alarm rates for each threshold
for iThresh = 1:length(thresholds)
    thresh = thresholds(iThresh);
    
    % For the OPTIMAL decoder
    pHit_opt(iThresh) = sum(s_hat_opt_right > thresh) / length(s_hat_opt_right);
    pFA_opt(iThresh) = sum(s_hat_opt_left > thresh) / length(s_hat_opt_left);
    
    % For the CORRELATION-BLIND decoder
    pHit_cb(iThresh) = sum(s_hat_cb_right > thresh) / length(s_hat_cb_right);
    pFA_cb(iThresh) = sum(s_hat_cb_left > thresh) / length(s_hat_cb_left);
end

% Calculate Area Under the Curve (AUC) using the trapezoidal method
auc_opt = abs(trapz(pFA_opt, pHit_opt));
auc_cb = abs(trapz(pFA_cb, pHit_cb));

% Visualize the ROC curves
figure;
hold on;
plot(pFA_opt, pHit_opt, 'b-', 'LineWidth', 2, 'DisplayName', sprintf('Optimal (AUC = %.3f)', auc_opt));
plot(pFA_cb, pHit_cb, 'r-', 'LineWidth', 2, 'DisplayName', sprintf('Correlation-Blind (AUC = %.3f)', auc_cb));
plot([0 1], [0 1], 'k--', 'DisplayName', 'Chance'); % Diagonal chance line
axis square;
xlabel('False Alarm Rate (p(Right|Left))');
ylabel('Hit Rate (p(Right|Right))');
title('ROC Analysis of Decoder Performance');
legend('show', 'location', 'southeast');

%% 7. Calculate empirical choice correlations
% Initialize vectors to store choice correlations
CC_opt_emp = nan(nNeurons, 1);
CC_cb_emp = nan(nNeurons, 1);

% Calculate choice correlation for each neuron
for i = 1:nNeurons
    % Correlation for the optimal decoder
    CC_opt_emp(i) = corr(respNeural(i, :)', s_hat_opt');
    
    % Correlation for the correlation-blind decoder
    CC_cb_emp(i) = corr(respNeural(i, :)', s_hat_cb');
end

% Visualize the choice correlations
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1);
bar(CC_cb_emp);
xlabel('Neuron #');
ylabel('Choice Correlation');
title('Empirical Choice Correlation (Correlation-Blind)');

subplot(1,2,2);
bar(CC_opt_emp);
xlabel('Neuron #');
ylabel('Choice Correlation');
title('Empirical Choice Correlation (Optimal)');

%% 8. Calculate theoretical CC
% Correlation-blind CC (Eq 2)
CC_cb_theo = sqrt(c0)* abs(sin(s_pref_allNeurons));

% Optimal CC (Eq 3)
theta_population = std(s_hat_opt);
theta_perNeuro = std(respNeural, [], 2);
CC_opt_theo = theta_population./theta_perNeuro;

% Compare empirical and theoretical CC
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1); hold on
plot(CC_opt_emp)
plot(CC_opt_theo)
yline(0, 'k--','HandleVisibility', 'off');
xlabel('Neuron #');
ylabel('Choice Correlation (CC)');
legend({'Empirical CC (based on continuous choice)', ' Empirical CC (based on discrete choice)', 'Theoretical CC'}, 'Location', 'best');
title('Choice Correlations (Optimal)');

subplot(1,2,2); hold on
plot(CC_cb_emp)
plot(CC_cb_theo)
yline(0, 'k--', 'HandleVisibility', 'off');
xlabel('Neuron #');
ylabel('Choice Correlation (CC)');
legend({'Empirical CC (based on continuous choice)', ' Empirical CC (based on discrete choice)', 'Theoretical CC'}, 'Location', 'best');
title('Choice Correlations (Correlation-blind)');

% make all linewidt to be 2
set(findall(gcf,'-property','LineWidth'),'LineWidth',2);

