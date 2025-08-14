
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

%% Parameters (adjust to match Pitkow et al. 2015)

flag_uncorrelated = 0;
% If 1, set Corr_signal to identity matrix. 
% This removes all noise correlations, providing a "sanity check" where the correlation-blind decoder's performance should equal the optimal decoder's performance.

flag_measure_threshold = 0;
% If 1, fixes the stimulus to the reference direction (s=0) for all trials. 

nNeurons = 100;  % Number of neurons
nTrialsPerStim = 1; % Number of trials per stimulus

% From Pitkow et al 2015
% stimVals = [-6.4, -2.6, -1, 0, 1, 2.6, 6.4]/180*pi;
% nStimVals = length(stimVals);

% Manually set the stimulus values
nStimVals = 1000; % number of stimulus levels
if flag_measure_threshold
    stimVals = zeros(1, nStimVals); % for testing, all zeros
else
    stimVals = linspace(-.1, .1, nStimVals);
    % stimVals = rand(1, nStimVals)*2-1;
end
stim_ref = 0; % the reference stimulus, which is the front direction

% Set preferred stimuli (Eq. S9–S13 in supplement)
s_pref_allNeurons = linspace(-pi/2, pi/2, nNeurons)'; % evenly spaced tuning preferences
% Xaq: the range should cover the full circle

nTrials  = nTrialsPerStim * nStimVals;  % total trials

% Others
c0 = 0.2; % proportionality constant % usually 0.1-0.5
alpha = .9; % the fraction of uncertainty (Eq 6)
% % Tuning function parameters
kappa = 1.5; % tuning width for von Mises; bigger kappa, narrower curve
b = 0;
a = 20;

% Print flags and parameters in one sentence
fprintf('==================\nCorrelated noise (1=NO, 0=YES): %d\nStimulus are all at ref (1=Yes; 0=No): %d\nParameters\n   %d neurons\n   %d trials per stim levels, %d stim levels\n   c0=%.2f, kappa=%.2f, b=%.2f, a=%.2f\n==================\n', flag_uncorrelated, flag_measure_threshold, nNeurons, nTrialsPerStim, nStimVals, c0, kappa, b, a);
rng(1); % reproducibility

% Set stimulus sequence
stimSeq = repmat(stimVals, 1, nTrialsPerStim); % stimulus sequence
% randomize
stimSeq = stimSeq(randperm(length(stimSeq)));

%% 1. Define true tuning curves
% Example tuning function: von Mises
fxn_tuning = @(s, s_pref) b + a * exp(kappa * (cos(s - s_pref) - 1)); % mean rates
fxn_tuning_dev = @(s, s_pref)  - a * exp(kappa * (cos(s - s_pref) - 1)) * kappa .* sin(s - s_pref); % derivative of the tuning fxn

% Derive the tuning curve and slope at the reference stimulus for all neurons
fprime_ref = fxn_tuning_dev(stim_ref, s_pref_allNeurons); % nNeurons x 1
f_ref = fxn_tuning(stim_ref, s_pref_allNeurons); % nNeurons x 1

% Derive the tuning curve slope at the reference stimulus for all neurons
fprime_ref_allN = nan(nNeurons, 1);
f_ref_allN = fprime_ref_allN;
for iN=1:nNeurons
    fprime_ref_allN(iN) = fxn_tuning_dev(stim_ref, s_pref_allNeurons(iN));
    f_ref_allN(iN) = fxn_tuning(stim_ref, s_pref_allNeurons(iN));
end
% Derive and neural threshold (assuming sigma_k = 1 and sigma_k is supposed to be an empirical value)
threshold = 1./fprime_ref_allN;

% Visualize tuning functions (top) and derivative (bottom) of the first 10 neurons
indNeuron = linspace(1,nNeurons,10);
colors = rand(length(indNeuron), 3);
figure('Position', [100, 100, 800, 800]);
subplot(2, 1, 1); hold on
for i = 1:length(indNeuron)
    s_pref = s_pref_allNeurons(indNeuron(i));
    fplot(@(stimVals) fxn_tuning(stimVals, s_pref), [-pi/2, pi/2], 'Color', colors(i,:), 'LineWidth', 1.5);
    xline(s_pref, 'k--', 'color', colors(i,:), 'LineWidth', 1.5);
    plot(s_pref, fxn_tuning(s_pref, s_pref), 'o', 'markerfacecolor', colors(i,:), 'MarkerEdgeColor', 'k')
end
xlim([-pi/2, pi/2]);
ylim([0, 25]);
ylabel('Mean Firing Rate (Hz)');
xlabel('Stimulus (rad)');

subplot(2, 1, 2); hold on
for i = 1:length(indNeuron)
    s_pref = s_pref_allNeurons(indNeuron(i));
    fplot(@(stimVals) fxn_tuning_dev(stimVals, s_pref), [-pi/2, pi/2], 'Color', colors(i,:), 'LineWidth', 1.5);
    xline(s_pref, 'k--', 'color', colors(i,:), 'LineWidth', 1.5);
    plot(s_pref, 0, 'o', 'markerfacecolor', colors(i,:), 'MarkerEdgeColor', 'k')
end
yline(0, 'k-', 'HandleVisibility', 'off');
xlim([-pi/2, pi/2]);
ylim([-25, 25]);
ylabel('Derivative of Firing Rate (Hz/rad)');
xlabel('Stimulus (rad)');
sgtitle('Tuning Functions and Derivatives for Selected Neurons');

% Visualize: tuning curve properties at the reference stim
figure('Position', [100, 100, 800 1200])
subplot(3, 1, 1)
plot(1:nNeurons, f_ref_allN, 'o-');
xlabel('Neuron Index');
ylabel('Theoretical mean neural response');
title('Tuning Curves for All Neurons at Reference Stimulus');

subplot(3, 1, 2)
plot(1:nNeurons, fprime_ref_allN, 'o-');
yline(0, 'k--', 'LineWidth', 1.5);
xlabel('Neuron Index');
ylabel('Theoretical slopes');
title('Tuning Curve SLOPES for All Neurons at Reference Stimulus');

subplot(3, 1, 3)
plot(1:nNeurons, threshold, 'o-');
yline(0, 'k--', 'LineWidth', 1.5);
xlabel('Neuron Index');
ylabel('Neural Threshold (Hz/rad)');
title('Neural Thresholds (sigma_k/fprime) for All Neurons at Reference Stimulus');

sgtitle('Summary of Tuning Properties for All Neurons at Reference Stimulus');

% Plot the differential correlation (f'f'^T at the reference direction)
Corr_diff = fprime_ref * fprime_ref'; % nNeurons x nNeurons
figure
imagesc(Corr_diff)
axis square; colorbar
xlabel('Neuron #');
ylabel('Neuron #');
title('Differential Correlation (f''f''^T) at reference direction (Fig 1E)');

%% 2a: Derive theoretical signal CORRELATION matrix (Eq. S13)
% CANNOT be derived from the data
% Xaq: this is a very specific example of correlation: noise corr is proportional to signal corr

% Derive the theoretical signal correlation matrix (which is essentially the "Structured Noise")
% The codes below is the same as Corr_signal = corr(tuning_curves_matrix);
% where tuning_curves_matrix(:, iN) = fxn_tuning(stimVals ,s_pref_allNeurons(iN)); looped through iN=1:nNeurons

if flag_uncorrelated
    Corr_signal = eye(nNeurons);
else
    Corr_signal = nan(nNeurons);
    for i = 1:nNeurons
        for j = 1:nNeurons
            num = besseli(0, sqrt(kappa^2 + kappa^2 + 2*kappa*kappa*cos(s_pref_allNeurons(i)-s_pref_allNeurons(j)))) - besseli(0, kappa)^2;
            den = sqrt((besseli(0, 2*kappa) - besseli(0, kappa)^2)^2);
            Corr_signal(i,j) = num / den;
        end
    end
end

%% 2b: Derive averaged noise CORRELATION matrix R_bar (Eq. S11, S13)
noise_idpdt = eye(nNeurons);
R_bar = (1-c0)*noise_idpdt + c0*Corr_signal;

% Visualize signal and noise correlation
figure('Position', [100, 100, 1200, 800]);
subplot(1,2,1)
imagesc(s_pref_allNeurons, s_pref_allNeurons, Corr_signal);
colorbar; axis square; clim([-1, 1]);
xlabel('Neuron''s preferred stimulus (rad)');
ylabel('Neuron''s preferred stimulus (rad)');
title('Theoretical Signal Correlation Matrix (or the structured noise)');

subplot(1,2,2)
imagesc(s_pref_allNeurons, s_pref_allNeurons, R_bar); colorbar;
colorbar; axis square; clim([-1, 1]);
xlabel('Neuron''s preferred stimulus (rad)');
ylabel('Neuron''s preferred stimulus (rad)');
title('Theoretical Noise Correlation Matrix');

fprintf('*** Derived Noise Correlation Matrix (R_bar) ***\n');

close all
%% 3. Simulate neural responses (Eq. 1 in main text)
fprintf('*** Simulating neural responses...     ');

respNeural = nan(nNeurons, nTrials);

if flag_uncorrelated
    for iTrial = 1:nTrials
        % Print a "=" at 10% of trials
        if mod(iTrial, round(nTrials/10)) == 0
            fprintf('='); drawnow;
        end
        stimVal = stimSeq(iTrial);
        % Exract tuning function
        f_s = fxn_tuning(stimVal, s_pref_allNeurons); % mean firing rates, nNeurons x 1
        % Just use the noise corr as the covariance
        SIGMA_perTrial = R_bar;
        % Sample neural responses from a multivariate normal distribution with defined mean and covariance
        respNeural(:,iTrial) = mvnrnd(f_s, SIGMA_perTrial)';    % draw one population vector
    end % end of t

else
    df = 2*nNeurons;  % define Wishart degrees of freedom (S14)

    for iTrial = 1:nTrials
        stimVal = stimSeq(iTrial);

        % Print a "=" at 10% of trials
        if mod(iTrial, round(nTrials/10)) == 0
            fprintf('='); drawnow;
        end

        % Exract tuning function
        f_s = fxn_tuning(stimVal, s_pref_allNeurons); % mean firing rates, nNeurons x 1

        % Create SIGMA_bar per trial
        SIGMA_bar_perTrial = nan(nNeurons, nNeurons);
        for i = 1:nNeurons
            for j = 1:nNeurons

                % Equation above Eq S14
                f_s_i = fxn_tuning(stimVal, s_pref_allNeurons(i)); % mean firing rates
                f_s_j = fxn_tuning(stimVal, s_pref_allNeurons(j)); % mean firing rates

                % based on the poisson-like scaling assumption
                SIGMA_bar_perTrial(i, j) = sqrt(f_s_i*f_s_j)*R_bar(i, j);

            end
        end

        % Sample covariate matrix from the Wishart distribution
        % just for adding randomness
        SIGMA_perTrial = wishrnd(SIGMA_bar_perTrial, df) / df;

        % Sample neural responses from a multivariate normal distribution with defined mean and covariance
        respNeural(:,iTrial) = mvnrnd(f_s, SIGMA_perTrial)';    % draw one population vector

        % Ensure non-negative firing rates
        % respNeural(:,t) = max(0, respNeural(:,t));
    end % end of t

end
fprintf('DONE ***.\n');

% Visualize neural responses
figure('Position', [100, 100, 1200, 400]);
subplot(1,3,1), hold on
imagesc(1:nTrials, s_pref_allNeurons, respNeural);
colorbar; axis square;
title('Simulated Neural Responses');
xlabel('Trial #');
ylabel('Neuron''s preferred direction');

subplot(1,3,2)
histogram(stimSeq, 30)
xlabel('Stimulus Value');
ylabel('Frequency');
title('Distribution of Stimuli of all trials');

subplot(1,3,3)
plot(sort(stimSeq))
xlabel('Sorted Stimulus Value');
ylabel('Frequency');
title('Distribution of Stimuli of all trials');

%% 3b. Derive the theoretical covariance matrix (SIGMA_bar)
if flag_uncorrelated
    SIGMA_theo = R_bar;
else
    SIGMA_theo = sqrt(f_ref * f_ref') .* R_bar;
end

fprintf('*** Created true covariance matrix ***\n');

%% 4a. Calculate the empirical covariance matrix
% Calculate the empirical covariance matrix

% **** (if go back to the complex model, which trial to use matters) ****
% respNeural_selected = respNeural(:, stimVals >= -0.1 & stimVals <= 0.1); % select trials close to the reference
% SIGMA_emp = cov(respNeural_selected');

SIGMA_emp = cov(respNeural');
% The transpose (') is crucial because cov expects variables to be in columns, and in your respNeural matrix, each row represents a variable (a neuron).
% You should NOT normalize SIGMA_emp, because the raw, unscaled SIGMA_emp is required for calculating your decoder weights

% Derive eigenvalues of empirical covariance matrix
[eigenvalues]= eig(SIGMA_emp);
eigenvalues = sort(eigenvalues, 'descend');
figure
plot(eigenvalues)
xlabel('Eigenvalue Index');
ylabel('Eigenvalues');
title(sprintf('Eigenvalues of Empirical Covariance Matrix\nShould resemble the Marchenko-Pastur distribution'));
% see fxn_fitMarchenkoPastur.m to check if eigenvalues of covariance matrix follow the Marchenko-Pastur distribution

% Visualize empirical & theoretical covariance matrix
figure('Position', [100, 100, 1200, 400]);

subplot(1,2,1);
imagesc(s_pref_allNeurons, s_pref_allNeurons, SIGMA_theo);
colorbar; axis square;
xlabel('Neuron''s preferred direction');
ylabel('Neuron''s preferred direction');
title('Theoretical Covariance Matrix');

subplot(1,2,2);
imagesc(s_pref_allNeurons, s_pref_allNeurons, SIGMA_emp);
colorbar; axis square;
xlabel('Neuron''s preferred direction');
title('Empirical Covariance Matrix');

sgtitle('Covariance Matrices');

%% 4b: Derive the theoretical & compute the empirical decoder [optimal] (The equation above Eq 3)
w_opt_emp_unnorm = SIGMA_emp \ fprime_ref; % nNeurons x 1; more accurate than w_opt = inv(SIGMA_emp) * f_s_dev;
w_opt_theo_unnorm = SIGMA_theo \ fprime_ref; % nNeurons x 1; more accurate than w_opt = inv(SIGMA_theo) * f_s_dev;

% Normalize decoder weights to obtain unbiased decoder
w_opt_theo = w_opt_theo_unnorm / (w_opt_theo_unnorm' * fprime_ref);
w_opt_emp = w_opt_emp_unnorm / (w_opt_emp_unnorm' * fprime_ref);
% The goal of this normalization is to create an unbiased estimator. An estimator is unbiased if, on average, its estimate is equal to the true value. In this model, that means when you apply the decoder to the pure, noise-free signal (f'), the output must be exactly 1. The mathematical condition is: w' * f' = 1. Your code, w_opt_emp = w_opt_emp_unnorm / (w_opt_emp_unnorm' * fprime_ref);, mathematically enforces this condition.

% do NOT normalize
% w_opt_theo = w_opt_theo_unnorm;
% w_opt_emp = w_opt_emp_unnorm;

%% 4c. Derive the theoretical & compute the empirical decoder [correlation-blind]
% A correlation-blind (cb) decoder assumes noise independence, using only the
% diagonal elements (variances) of the noise covariance matrix.

% --- Empirical CB Decoder (learns from data) ---

var_emp = diag(SIGMA_emp);
w_cb_emp_unnorm = fprime_ref ./ var_emp;
% Normalization is crucial for an unbiased estimator
w_cb_emp = w_cb_emp_unnorm / (w_cb_emp_unnorm' * fprime_ref);
% w_cb_emp = w_cb_emp_unnorm;

% --- Theoretical CB Decoder (knows true variances) ---
var_theo = diag(SIGMA_theo);
w_cb_theo_unnorm = fprime_ref ./ var_theo;
% Normalize this decoder as well for a fair comparison
w_cb_theo = w_cb_theo_unnorm / (w_cb_theo_unnorm' * fprime_ref);
% w_cb_theo = w_cb_theo_unnorm;

% Visualize two decoders empirical & theoretical
figure('Position', [100, 100, 1200, 400]);

subplot(1,2,1); hold on
plot(s_pref_allNeurons, w_opt_emp, 'DisplayName', 'Empirical decoder');
plot(s_pref_allNeurons, w_opt_theo, 'DisplayName', 'Theoretical decoder');
yline(0, 'k-', 'HandleVisibility','off');
xlabel('Neuron''s preferred direction');
ylabel('Decoder weight');
legend('show', 'location', 'best');
title('Optimal Decoder Weights');

subplot(1,2,2); hold on
plot(s_pref_allNeurons, w_cb_emp, 'DisplayName', 'Empirical decoder');
plot(s_pref_allNeurons, w_cb_theo, 'DisplayName', 'Theoretical decoder');
yline(0, 'k-', 'HandleVisibility','off');
xlabel('Neuron''s preferred direction');
ylabel('Decoder weight');
legend('show', 'location', 'best');
title('Correlation-blind Decoder Weights');

set(findall(gcf,'-property','LineWidth'),'LineWidth',2);
set(findall(gcf,'-property','fontsize'),'fontsize', 15);

saveas(gcf, sprintf('Figures/DecoderWeights.jpg'));

%% 5. Calculate the estimator (s_hat) by applying decoders to the stimulus
% I need to isolate the noise from the signal on each trial: Noise = Actual Response - Expected Mean Response
f_s_ref_mtx = repmat(f_ref, 1, nTrials);
respNeural_deviations = respNeural - f_s_ref_mtx; % <--- This is the correct subtraction

% 1. Select  trials with stim close to the reference
indTrial_selected = stimSeq >= -0.1 & stimSeq <= 0.1; % close to the reference
nTrials_selected = sum(indTrial_selected);

s_hat_opt = w_opt_emp' * respNeural_deviations(:, indTrial_selected);
s_hat_cb  = w_cb_emp'  * respNeural_deviations(:, indTrial_selected);

% % 2. Use all trials
s_hat_opt_full = w_opt_emp' * respNeural_deviations;
s_hat_cb_full  = w_cb_emp'  * respNeural_deviations;

% Plot input vs output, which should fall on the unity line
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1), hold on;
plot(stimSeq, s_hat_opt_full, 'o', 'DisplayName', 'Optimal decoder');
plot(stimSeq, s_hat_cb_full, 'o', 'DisplayName', 'CB decoder');
% plot([-.1, .1], [-.1, .1], 'k--', 'HandleVisibility','off');
xlabel('Input (stimulus)');
ylabel('Output (estimator)');
% xlim([-.1, .1]);
% ylim([-.1, .1]);
legend('show', 'Location', 'best');
axis square 
title('All trials')

subplot(1,2,2), hold on;
plot(stimSeq(indTrial_selected), s_hat_opt, 'o', 'DisplayName', 'Optimal decoder');
plot(stimSeq(indTrial_selected), s_hat_cb, 'o', 'DisplayName', 'Optimal decoder');
% plot([-.1, .1], [-.1, .1], 'k--', 'HandleVisibility','off');
xlabel('Input (stimulus)');
ylabel('Output (estimator)');
% xlim([-.1, .1]);
% ylim([-.1, .1]);
legend('show', 'Location', 'best');
axis square
title('Selected trials')
sgtitle('Input vs Output');
set(findall(gcf,'-property','LineWidth'),'LineWidth',2);
set(findall(gcf,'-property','fontsize'),'fontsize', 15);

%% 7. Derive choices (1 for right, -1 for left)
choice_opt = sign(s_hat_opt);
choice_cb  = sign(s_hat_cb);

%% 6. Calculate empirical choice correlations (just Pearson's correlation)
% Initialize vectors to store choice correlations
% The decoder used to generate estimator matches the decoder to calculate CC
% CC_opt_emp_opt = nan(nNeurons, nStimVals);
CC_opt_emp_opt = nan(nNeurons, 1);
CC_cb_emp_cb = CC_opt_emp_opt;
% The decoder used to generate estimator does NOT match the decoder to calculate CC
CC_opt_emp_cb = CC_opt_emp_opt;
CC_cb_emp_opt = CC_opt_emp_opt;

% Calculate choice correlation for each neuron
iStim_selected = nStimVals/2;
for iN = 1:nNeurons
    for iStim = iStim_selected
        indTrial = stimSeq == stimVals(iStim);
        % indTrial_selected = indTrial & indTrial_selected;

        % Correlation for the optimal decoder
        CC_opt_emp_opt(iN) = corr(respNeural(iN, indTrial)', s_hat_opt_full(indTrial)');
        CC_opt_emp_cb(iN) = corr(respNeural(iN, indTrial)', s_hat_cb_full(indTrial)');

        % Correlation for the correlation-blind decoder
        CC_cb_emp_cb(iN) = corr(respNeural(iN, indTrial)', s_hat_cb_full(indTrial)');
        CC_cb_emp_opt(iN) = corr(respNeural(iN, indTrial)', s_hat_opt_full(indTrial)');
    end
end

% Visualize the choice correlations (see the next section, plotted together with the theoretical CC)

%% 7. Calculate theoretical CC (Eq 2 and 3)
% Correlation-blind CC (Eq 2)
CC_cb_theo = sqrt(c0)* abs(sin(s_pref_allNeurons));

% Optimal CC (Eq 3)
theta_population = std(s_hat_opt); % a scalar; s_hat_opt is 1xnTrials
% theta_perNeuron = nan(nNeurons, 1);
iStim = nStimVals/2;
theta_perNeuron = std(respNeural(:, stimSeq == stimVals(iStim)), [], 2)./fprime_ref; % expression is in p3; nNeurons x 1

CC_opt_theo = theta_population./theta_perNeuron;

%% [Fig 2A] Plot CC (with both matched and mismatched decoder) as a fxn of theta_k and s_pref
figure('Position', [100, 100, 1200, 1200]);
subplot(2, 2, 1), hold on
plot(theta_perNeuron, CC_opt_emp_opt, 'o', 'DisplayName', 'Optimal decoder (empirical and matched)');
plot(theta_perNeuron, CC_opt_theo, 'k-', 'DisplayName', 'Optimal decoder (theoretical)');
xlim([-3, 3])
ylim([-.3, .3])
xlabel('Threshold (Theta_k)');
ylabel('Choice corr. C_k');
xline(0, 'k-','HandleVisibility', 'off'); yline(0, 'k-','HandleVisibility', 'off');
legend('show', 'Location', 'best');

subplot(2, 2, 2), hold on
plot(theta_perNeuron, mean(CC_cb_emp_opt, 2), 'o', 'DisplayName', 'CB decoder (empirical and mismatched)');
plot(theta_perNeuron, CC_opt_theo, 'k-', 'DisplayName', 'Optimal decoder (theoretical)');
xlabel('Threshold (Theta_k)');
ylabel('Choice corr. C_k');
xlim([-3, 3])
ylim([-.3, .3])
xline(0, 'k-','HandleVisibility', 'off'); yline(0, 'k-','HandleVisibility', 'off');
legend('show', 'Location', 'best');

subplot(2, 2, 3), hold on
plot(s_pref_allNeurons, mean(CC_opt_emp_cb, 2), 'o', 'DisplayName',  'Optimal decoder (empirical and mismatched)');
plot(s_pref_allNeurons, CC_cb_theo, 'k-', 'DisplayName', 'CB decoder (theoretical)');
xlabel('Preferred stimulus s_k');
ylabel('Choice corr. C_k');
ylim([-1, 1])
xline(0, 'k-','HandleVisibility', 'off'); yline(0, 'k-','HandleVisibility', 'off');
legend('show', 'Location', 'best');

subplot(2, 2, 4), hold on
plot(s_pref_allNeurons, mean(CC_cb_emp_cb, 2), 'o', 'DisplayName',  'CB decoder (empirical and matched)');
plot(s_pref_allNeurons, CC_cb_theo, 'k-', 'DisplayName', 'CB decoder (theoretical)');
xlabel('Preferred stimulus s_k');
ylabel('Choice corr. C_k');
ylim([-1, 1])
xline(0, 'k-','HandleVisibility', 'off'); yline(0, 'k-','HandleVisibility', 'off');
legend('show', 'Location', 'best');

sgtitle('Replication of Figure 2A')

saveas(gcf, sprintf('Figures/Fig2A.jpg'));

%% [Fig 2B] Plot emp CC as a fxn of pred CC
figure('Position', [100, 100, 1200, 1200]);
subplot(2, 2, 1), hold on
plot(CC_opt_theo, mean(CC_opt_emp_opt, 2), 'o');
plot(CC_opt_theo, CC_opt_theo, 'k-');
xlabel('Predicted CC for optimal decoder');
ylabel('Empirical CC');
xline(0, 'k-','HandleVisibility', 'off'); yline(0, 'k-','HandleVisibility', 'off');
axis square
legend('show', 'Location', 'best');

subplot(2, 2, 2), hold on
plot(CC_cb_theo, mean(CC_cb_emp_opt, 2), 'o');
plot(CC_cb_theo, CC_cb_theo, 'k-');
xlabel('Predicted CC for optimal decoder');
ylabel('Empirical CC');
xline(0, 'k-','HandleVisibility', 'off'); yline(0, 'k-','HandleVisibility', 'off');
axis square
legend('show', 'Location', 'best');

subplot(2, 2, 3), hold on
plot(CC_opt_theo, mean(CC_opt_emp_cb, 2), 'o', 'DisplayName', 'Optimal decoder (empirical)');
plot(CC_opt_theo, CC_opt_theo, 'k-');
xlabel('Predicted CC for CB decoder');
ylabel('Empirical CC');
xline(0, 'k-','HandleVisibility', 'off'); yline(0, 'k-','HandleVisibility', 'off');
axis square
legend('show', 'Location', 'best');

subplot(2, 2, 4), hold on
plot(CC_cb_theo, mean(CC_cb_emp_cb, 2), 'o', 'DisplayName', 'CB decoder (empirical)');
plot(CC_cb_theo, CC_cb_theo, 'k-');
xlabel('Predicted CC for CB decoder');
ylabel('Empirical CC');
xline(0, 'k-','HandleVisibility', 'off'); yline(0, 'k-','HandleVisibility', 'off');
axis square
legend('show', 'Location', 'best');

sgtitle('Replication of Figure 2B')

saveas(gcf, sprintf('Figures/Fig2B.jpg'));

%% [Fig 6C] Compare empirical and theoretical CC
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1); hold on
plot(s_pref_allNeurons, mean(CC_opt_emp_opt, 2))
errorbar(s_pref_allNeurons, mean(CC_opt_emp_opt, 2), std(CC_opt_emp_opt, [], 2), 'k', 'LineStyle', 'none', 'CapSize',0, 'HandleVisibility','off')
plot(s_pref_allNeurons, CC_opt_theo)
yline(0, 'k--','HandleVisibility', 'off');
xlabel('Neuron''s preferred direction');
ylabel('Choice Correlation (CC)');
ylim([-1, 1])
legend({'Empirical CC ', 'Theoretical CC'}, 'Location', 'best');
title('Choice Correlations (Optimal)');

subplot(1,2,2); hold on
plot(s_pref_allNeurons, mean(CC_cb_emp_cb, 2))
errorbar(s_pref_allNeurons, mean(CC_cb_emp_cb, 2), std(CC_cb_emp_cb, [], 2), 'k', 'LineStyle', 'none', 'CapSize',0, 'HandleVisibility','off')
plot(s_pref_allNeurons, CC_cb_theo)
yline(0, 'k--', 'HandleVisibility', 'off');
xlabel('Neuron''s preferred direction');
ylabel('Choice Correlation (CC)');
ylim([-1, 1])
legend({'Empirical CC ', 'Theoretical CC'}, 'Location', 'best');
title('Choice Correlations (Correlation-blind)');

set(findall(gcf,'-property','LineWidth'),'LineWidth',2);
set(findall(gcf,'-property','fontsize'),'fontsize', 15);

saveas(gcf, sprintf('Figures/Fig6C.jpg'));

close all