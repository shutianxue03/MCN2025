
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

clear all; clc;

%% Parameters (adjust to match Pitkow et al. 2015)
nNeurons = 100;           % population size
nTrialsPerStim = 200;

% From Pitkow et al 2015
% stimVals = [-6.4, -2.6, -1, 0, 1, 2.6, 6.4]/180*pi;
% nStimVals = length(stimVals);

% Manually set the stimulus values
nStimVals = 50; % number of stimulus levels
stimVals = linspace(-pi/2, pi/2, nStimVals);

% define preferred stimuli (Eq. S9–S13 in supplement)
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

% Print parameters in one sentence
fprintf('==================\nParameters\n   %d neurons\n   %d trials per stim levels, %d stim levels\n   c0=%.2f, kappa=%.2f, b=%.2f, a=%.2f\n==================\n', nNeurons, nTrialsPerStim, nStimVals, c0, kappa, b, a);
rng(1); % reproducibility

% Define stimulus sequence
stimSeq = repmat(stimVals, 1, nTrialsPerStim); % stimulus sequence
% randomize
stimSeq = stimSeq(randperm(length(stimSeq)));

%% 1. Define true tuning curves
% Example tuning function: von Mises
fxn_tuning = @(s, s_pref) b + a * exp(kappa * (cos(s - s_pref) - 1)); % mean rates
fxn_tuning_dev = @(s, s_pref)  - a * exp(kappa * (cos(s - s_pref) - 1)) * kappa .* sin(s - s_pref); % derivative of the tuning fxn

% [Visualize] tuning function (top) and derivative (bottom) of the first 10 neurons
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

%% 2. Define theoretical signal correlation matrix (Eq. S13) and averaged noise correlation matrix R (Eq. S13)
% Xaq: this is a very specific example of correlation: noise corr is proportional to signal corr

Corr_signal = nan(nNeurons);
for i = 1:nNeurons
    for j = 1:nNeurons
        num = besseli(0, sqrt(kappa^2 + kappa^2 + 2*kappa*kappa*cos(s_pref_allNeurons(i)-s_pref_allNeurons(j)))) - besseli(0, kappa)^2;
        den = sqrt((besseli(0, 2*kappa) - besseli(0, kappa)^2)^2);
        Corr_signal(i,j) = num / den;
    end
end

% Compute the eigenvalues of signal correlation matrix
eigenvalues = eig(Corr_signal);
figure
e = sort(eigenvalues, 'descend');
stem(e(1:10), 'filled');
xlabel('Eigenvalue Index');
ylabel('Eigenvalue');
title('First 10 eigenvalues of Signal Correlation Matrix');

% Derive averaged noise correlation matrix R (Eq. S11)
% Which is also the TRUE noise correlation matrix
R_bar = (1-c0)*eye(nNeurons) + c0*Corr_signal;

figure('Position', [100, 100, 800, 300]);
subplot(1,2,1)
imagesc(s_pref_allNeurons, s_pref_allNeurons, Corr_signal);
colorbar; axis square; clim([-1, 1]);
xlabel('Neuron''s preferred stimulus (rad)');
title('Theoretical Signal Correlation Matrix');

subplot(1,2,2)
imagesc(s_pref_allNeurons, s_pref_allNeurons, R_bar); colorbar;
colorbar; axis square; clim([-1, 1]);
xlabel('Neuron''s preferred stimulus (rad)');
title('Averaged Noise Correlation Matrix (the TRUE noise corr matrix)');

fprintf('*** Derived Noise Correlation Matrix (R_bar) ***\n');

%% 3. Simulate neural responses (Eq. 1 in main text)
fprintf('*** Simulating neural responses...     ');

respNeural   = nan(nNeurons, nTrials);
df = 2*nNeurons;  % define Wishart degrees of freedom (S14)

% evaluate derivative at s=0, which is the reference heading
stim_ref = zeros(size(stimSeq(1)));

for t = 1:nTrials
    stimVal = stimSeq(t);

    % Print a "=" at 10% of trials
    if mod(t, round(nTrials/10)) == 0
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
            SIGMA_bar_perTrial(i, j) = R_bar(i, j);
            % can try scaling down R_bar

        end
    end

    % Sample covariate matrix from the Wishart distribution
    % just for adding randomness
    % SIGMA_perTrial = wishrnd(SIGMA_bar_perTrial, df) / df;
    SIGMA_perTrial = SIGMA_bar_perTrial;

    % Sample neural responses from a multivariate normal distribution with defined mean and covariance
    respNeural(:,t) = mvnrnd(f_s, SIGMA_perTrial)';    % draw one population vector

    % Ensure non-negative firing rates
    % respNeural(:,t) = max(0, respNeural(:,t));
end % end of t

% Visualize neural responses
figure('Position', [100, 100, 1200, 400]);
subplot(1,3,1), hold on
imagesc(1:nTrials, s_pref_allNeurons, respNeural);
colorbar;
title('Simulated Neural Responses');
xlabel('Trial #');
ylabel('Neuron''s preferred direction');

%
subplot(1,3,2)
histogram(stimSeq, 30)

subplot(1,3,3)
plot(sort(stimSeq))

fprintf('DONE ***.\n');

close all


%% 3b. Create the true covariance matrix (SIGMA_bar)
f_s0 = fxn_tuning(stim_ref, s_pref_allNeurons); % Mean firing rates at s=0

% SIGMA_theo = nan(nNeurons, nNeurons);
% for i = 1:nNeurons
%     for j = 1:nNeurons
%         SIGMA_theo(i, j) = sqrt(f_s0(i) * f_s0(j)) * R_bar(i, j);
%     end
% end

SIGMA_theo = R_bar;

fprintf('*** Created true covariance matrix ***\n');

%% 4a. Compute the optimal decoder weights (Eq. 3)
% Calculate the empirical covariance matrix
SIGMA_emp = cov(respNeural');

% Visualize empirical & theoretical noise covariance
figure('Position', [100, 100, 1200, 400]);
subplot(1,2,1);
imagesc(s_pref_allNeurons, s_pref_allNeurons, SIGMA_emp);
colorbar; axis square;
xlabel('Neuron''s preferred direction');
title('Empirical Covariance Matrix');

subplot(1,2,2);
imagesc(s_pref_allNeurons, s_pref_allNeurons, SIGMA_theo);
colorbar; axis square;
xlabel('Neuron''s preferred direction');
ylabel('Neuron''s preferred direction');
title('Theoretical Covariance Matrix');

% Derive the optimal decoder (from the equation above Eq 3)
f_s_dev = fxn_tuning_dev(stim_ref, s_pref_allNeurons);  % nNeurons x 1
w_opt_emp_unnorm = SIGMA_emp \ f_s_dev; % nNeurons x 1; more accurate than w_opt = inv(SIGMA_emp) * f_s_dev;
w_opt_theo_unnorm = SIGMA_theo \ f_s_dev; % nNeurons x 1; more accurate than w_opt = inv(SIGMA_theo) * f_s_dev;

% w_opt_theo_unnorm2 = inv(SIGMA_theo) * f_s_dev;

% figure
% plot(w_opt_theo_unnorm2, w_opt_theo_unnorm, 'o' )

% Normalize decoder weights to obtain unbiased decoder
% w_opt_theo = w_opt_theo_unnorm / (w_opt_theo_unnorm' * f_s_dev);
% w_opt_emp = w_opt_emp_unnorm / (w_opt_emp_unnorm' * f_s_dev);

% do NOT normalize
w_opt_theo = w_opt_theo_unnorm;
w_opt_emp = w_opt_emp_unnorm;

%% 4b. Compute the correlation-blind (factorial) suboptimal decoder weights
% A correlation-blind decoder assumes independence between the noise of different neurons, hence it only uses only the diagonal elements (variances) of the noise covariance matrix, ignoring the off-diagonal (covariance) terms.
var_emp = diag(SIGMA_emp);
var_theo = diag(SIGMA_theo);

w_cb_emp =  f_s_dev./ var_emp;
% w_cb_emp = w_cb_emp / (w_cb_emp' * f_s_dev); % normalize

w_cb_theo =  f_s_dev./ var_theo;
% w_cb_theo = w_cb_theo / (w_cb_theo' * f_s_dev); % normalize

% Visualize two decoders
figure('Position', [100, 100, 1200, 400]);

subplot(1,2,1); hold on
plot(s_pref_allNeurons, w_opt_emp, 'DisplayName', 'Empirical weights');
plot(s_pref_allNeurons, w_opt_theo, 'DisplayName', 'Theoretical weights');
yline(0, 'k-', 'HandleVisibility','off');
xlabel('Neuron''s preferred direction');
ylabel('Decoder weight');
legend('show', 'location', 'best');
title('Optimal Decoder Weights');

subplot(1,2,2); hold on
plot(s_pref_allNeurons, w_cb_emp, 'DisplayName', 'Empirical weights');
plot(s_pref_allNeurons, w_cb_theo, 'DisplayName', 'Theoretical weights');
yline(0, 'k-', 'HandleVisibility','off');
xlabel('Neuron''s preferred direction');
ylabel('Decoder weight');
legend('show', 'location', 'best');
title('Correlation-blind Decoder Weights');

set(findall(gcf,'-property','LineWidth'),'LineWidth',2);
set(findall(gcf,'-property','fontsize'),'fontsize', 15);

saveas(gcf, sprintf('Figures/DecoderWeights.jpg'));

%% 5. Apply decoders and generate behavior
% Derive estimators (s_hat)
% Only choose the trials with stim close to the reference
indTrial_selected = stimSeq >= -0.1 & stimSeq <= 0.1; % close to the reference
nTrials_selected = sum(indTrial_selected);
s_hat_opt_full = w_opt_emp' * respNeural;
s_hat_cb_full  = w_cb_emp'  * respNeural;

s_hat_opt = w_opt_emp' * respNeural(:, indTrial_selected);
s_hat_cb  = w_cb_emp'  * respNeural(:, indTrial_selected);

% Plot input vs output
figure; hold on;
plot(stimSeq(indTrial_selected), s_hat_opt, 'o', 'DisplayName', 'Optimal decoder');
errorbar(unique(stimSeq(indTrial_selected)), mean(s_hat_opt), std(s_hat_opt), 'capsize', 10, 'LineWidth', 2);
% plot(stimSeq(indTrial_selected), s_hat_cb, 'o', 'DisplayName', 'Correlation-blind decoder');
xlabel('Input (stimulus)');
ylabel('Output (estimator)');
legend('show', 'Location', 'best');
title('Input vs Output');
set(findall(gcf,'-property','LineWidth'),'LineWidth',2);
set(findall(gcf,'-property','fontsize'),'fontsize', 15);

% Derive choices (1 for right, -1 for left)
choice_opt = sign(s_hat_opt);
choice_cb  = sign(s_hat_cb);

%% 6. Calculate empirical choice correlations (just Pearson's correlation)
% Initialize vectors to store choice correlations
% The decoder used to generate estimator matches the decoder to calculate CC
CC_opt_emp_opt = nan(nNeurons, nStimVals);
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
theta_perNeuron = std(respNeural(:, stimSeq == stimVals(iStim)), [], 2)./f_s_dev; % expression is in p3; nNeurons x 1

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

%% Some (now-considered-redundant) analysis
%% (a): fitting psychometric functions
% Calculate p(Right|s)
pRight_opt = nan(nStimVals, 1);
pRight_cb = nan(nStimVals, 1);
for iStim = 1:nStimVals
    assert(sum(stimSeq == stimVals(iStim)) == nTrialsPerStim, 'ALERT: trial number is wrong')
    pRight_opt(iStim) = mean(choice_opt(stimSeq == stimVals(iStim))==1);
    pRight_cb(iStim) = mean(choice_cb(stimSeq == stimVals(iStim))==1);
end

% Xaq: PMF is meant to be avoided; but fig looks weird!!

% Fit accumulative gaussian functions to PMF
fxn_firCumGauss = @(params, x) 1/2 + 1/2*erf((x-params(1))/(params(2)*sqrt(2)));
params0 = [0, 0.1]; % initial parameters for intercept and threshold
beta_opt = nlinfit(stimVals, pRight_opt', fxn_firCumGauss, params0);
beta_cb = nlinfit(stimVals, pRight_cb', fxn_firCumGauss, params0);

% Visualize data and fitting psychometric function
figure;
hold on;
plot(stimVals, pRight_opt, 'bo', 'DisplayName', 'Optimal Decoder');
plot(stimVals, fxn_firCumGauss(beta_opt, stimVals), 'b--', 'DisplayName', sprintf('Optimal Fit (threshold=%.2f)', beta_opt(2)));
plot(stimVals, pRight_cb, 'ro', 'DisplayName', 'Correlation-blind Decoder');
plot(stimVals, fxn_firCumGauss(beta_cb, stimVals), 'r--', 'DisplayName', sprintf('Correlation-blind Fit (threshold=%.2f)', beta_cb(2)));
xlabel('Stimulus Value (s)');
ylabel('P(Right|s)');
title('Psychometric Function');
legend('show', 'location', 'best');

%% (b). ROC Analysis for Decoder Performance
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

%% (c). Calculate Trial-to-Trial Variability (Fano Factor & Behavioral)

% --- NEURONAL VARIABILITY ---
fanoFactors = nan(nNeurons, 1);
for iN = 1:nNeurons
    variances_per_stim = nan(nStimVals, 1);
    means_per_stim = nan(nStimVals, 1);

    for iStim = 1:nStimVals
        stim = stimVals(iStim);
        indTrial = stimSeq == stim;

        % Get responses for this neuron for all trials of a specific stimulus
        neuron_responses = respNeural(iN, indTrial);

        variances_per_stim(iStim) = var(neuron_responses);
        means_per_stim(iStim) = mean(neuron_responses);
    end

    % Fano Factor = mean of (variance / mean) across all stimulus conditions
    fanoFactors(iN) = mean(variances_per_stim ./ means_per_stim, 'omitnan');
end

% Visualize the Fano Factors
figure;
histogram(fanoFactors);
xlabel('Fano Factor (var/mean)');
ylabel('Neuron Count');
title('Neuronal Trial-to-Trial Variability');

% --- BEHAVIORAL VARIABILITY ---
var_behav_opt = nan(nStimVals, 1);
var_behav_cb = nan(nStimVals, 1);

for iStim = 1:nStimVals
    stim = stimVals(iStim);
    indTrial = stimSeq == stim;

    % Calculate the variance of the decoder's estimate, which should = threshold (plot as vertical lines below)
    var_behav_opt(iStim) = var(s_hat_opt(indTrial));
    var_behav_cb(iStim) = var(s_hat_cb(indTrial));
end

figure('Position', [100, 100, 1200, 400])
subplot(1,2,1), hold on;
histogram(var_behav_opt, 'FaceColor', 'b', 'FaceAlpha', 0.5);
xline(beta_opt(2)^2, 'k-', 'LineWidth', 2);
xlabel('Behavioral Variance');
ylabel('Count');
title('Behavioral Variance (Optimal)');
legend({'Variance of s_{hat}', 'Threshold theta'});

subplot(1,2,2), hold on;
histogram(var_behav_cb, 'FaceColor', 'r', 'FaceAlpha', 0.5);
xline(beta_cb(2)^2, 'k-', 'LineWidth', 2);
xlabel('Behavioral Variance');
ylabel('Count');
title('Behavioral Variance (Correlation-Blind)');
legend({'Variance of s_{hat}', 'Threshold theta'});


fprintf('\n\n   ALL DONE\n\n')
close all