
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
str_QC = 'QCFalse';
flag_selectPosCC = 0;

%% Load data
filename = sprintf('data_%s_%s_%s.mat', str_region, str_priorCond, str_QC);
indC_CC = 1; % Use CC of the lower contrast
indC_Dis = 5; % Use discriminability of the higher contrast

filename = 'data_sim.mat';
indC_CC = 1; % Use CC of the lower contrast
indC_Dis = 2; % Use discriminability of the higher contrast

load(filename)

%% Extract key parameters
[nNeurons, ~] = size(respNeural{1})

nC = length(contrast_all);

contrasts_allC = [];
spike_counts_allC = [];
behavior_responses_allC = [];

for iC = 1:nC
    contrast = contrast_all(iC); % Convert string to numerical contrast value

    respNeural_perC = double(respNeural{iC});
    respBehav_perC = double(respBehav{iC});
    assert(unique(respBehav_perC(1, :)) == contrast)

    nTrials_perC = size(respNeural_perC, 2);

    figure('Position', [100, 100, 2e3, 2e3])

    % Plot raster
    % figure
    subplot(1,3,1), hold on;
    imagesc(respNeural_perC);
    colorbar;
    xlabel('Trial #');
    ylabel('Neuron #');
    title('Spike count');

    % Covariance
    cov_perC = cov(respNeural_perC');
    subplot(1,3,2), hold on;
    imagesc(cov_perC);
    colorbar; axis square
    xlabel('Neuron #');
    ylabel('Neuron #');
    title('Neural Response Covariance');

    % Conduct PCA and plot eigenvalues
    eigenvalues = eig(cov_perC);
    eigenvalues_s = sort(eigenvalues/sum(eigenvalues), 'descend');
    subplot(1,3,3), hold on;
    stem(eigenvalues_s(1:10));
    xlabel('Principal Component');
    ylabel('Variance Explained');
    title('PCA Variance Explained');

    sgtitle(sprintf('Neural Data Analysis (Region=%s | Contrast=%.2f | %d Trials | Prior cond = %s)', str_region, contrast, nTrials_perC, str_priorCond));
end

%% Derive tuning function
% f = nan(nNeurons, nC);
% for iC=1:nC
%     f(:, iC) = mean(respNeural{iC}, 2);
% end

% figure, hold on
% x_log = [-1.5, log10(contrast_all(2:end))]; % because log10(0) is inf
% for iN = 1:nNeurons
%     plot(x_log, f(iN, :), 'o-')
% end
% % plot ave across neurons and errorbar
% errorbar(x_log, mean(f, 1), std(f, [], 1)/sqrt(nNeurons), 'k', 'LineWidth', 2)
% xticks(x_log)
% xticklabels(string(contrast_all))
% ylim([0, 4])
% xlabel('Contrast')
% ylabel('Spike count')
% % xlim([min(contrast_all), max(contrast_all)])
% title('Tuning curves')

close all
%% Do bootstrapping
nBoot = 1e2;
CC_allBoot = nan(nBoot, nNeurons, nC);
discrim_allBoot = nan(nBoot, nNeurons, nC);
slope_allBoot = nan(nBoot, 1);
rS_allBoot = slope_allBoot;
rK_allBoot = slope_allBoot;
pS_allBoot = slope_allBoot;
pK_allBoot = slope_allBoot;

for iBoot = 1:nBoot
    fprintf('Bootstrap %d / %d\n', iBoot, nBoot);

    %% generate indTrial for each contrast level
    indTrial = cell(nC, 1);
    for iC = 1:nC
        nTrials_perC = size(respNeural{iC}, 2);
        indTrial{iC} = randsample(nTrials_perC, nTrials_perC, true);
        % indTrial{iC} = 1:nTrials_perC;%randsample(nTrials_perC, nTrials_perC, true);
    end
    fprintf('\n >>> IndTrial generated for %d contrasts\n', nC);

    %% Calculate f' (average spike count of stim_L minus stim_R)
    % For each contrast
    fprime = nan(nNeurons, nC);
    for iC = 1:nC
        respNeural_perC = double(respNeural{iC});
        respBehav_perC = double(respBehav{iC});
        choice_perC = respBehav_perC(2, indTrial{iC});
        stim_perC = respBehav_perC(3, indTrial{iC});
        indTrial_L = stim_perC == -1;
        indTrial_R = stim_perC == 1;

        fprime(:, iC) = (mean(respNeural_perC(:, stim_perC == 1), 2) - mean(respNeural_perC(:, stim_perC == -1), 2));
    end

    if iBoot<=5
        figure('Position',[100 100 600 400]);
        hold on; box on;
        x_contrast = [-1.5, log10(contrast_all(2:end))]; % because log10(0) is inf
        plot(x_contrast, fprime, '-');
        errorbar(x_contrast, mean(fprime), std(fprime)/sqrt(nNeurons), 'k-', 'LineWidth', 2);
        xticks(x_contrast)
        xticklabels(string(contrast_all))
        xlabel('Contrast')
        ylabel('$\Delta f$ (Spike count difference)', 'Interpreter', 'latex');
        set(gca, 'FontSize', 20)  % Sets font size for axis tick labels
        xlim([-1.8, .3])
        ylim([-.2, 1])
    end
    
    % collapse across contrast
    choice_allC = [];
    stim_allC = [];
    indTrial_L_allC = [];
    indTrial_R_allC = [];
    for iC = 1:nC
        respNeural_perC = double(respNeural{iC});
        respBehav_perC = double(respBehav{iC});
        choice_perC = respBehav_perC(2, indTrial{iC});
        stim_perC = respBehav_perC(3, indTrial{iC});
        indTrial_L = stim_perC == 1;
        indTrial_R = stim_perC == -1;

        indTrial_L_allC = [indTrial_L_allC, indTrial_L];
        indTrial_R_allC = [indTrial_R_allC, indTrial_R];
    end
    fprime_collapse = abs(mean(respNeural_perC(:, indTrial_R), 2) - mean(respNeural_perC(:, indTrial_L), 2));

    fprintf(' >>> fprime calculated\n   ');

    %% Calculate response variance
    sigma_r = nan(nNeurons, nC);
    for iC = 1:nC
        respNeural_perC = double(respNeural{iC});
        sigma_r(:, iC) = std(respNeural_perC(:, indTrial{iC}), [], 2);
    end

    fprintf(' >>> Response variance calculated\n');

    %% Calculate CC and neural threshold
    CC_allN = nan(nNeurons, nC);
    thresh_allN = nan(nNeurons, nC);
    for iN=1:nNeurons
        for iC = 1:nC
            % Calculate CC
            respNeural_perC = double(respNeural{iC});
            % respNeural_perC_norm = (respNeural_perC-mean(respNeural_perC, 2))/std(respNeural_perC, [], 2);

            respBehav_perC = double(respBehav{iC});
            choice_perC = respBehav_perC(2, indTrial{iC});

            % Pearson's correlation
            CC_allN(iN, iC) = corr(respNeural_perC(iN, indTrial{iC})', choice_perC');

            % GLM
            % tbl = table(respNeural_perC(iN, indTrial{iC})', choice_perC', 'VariableNames', {'x','y'});
            % mdl = fitglm(tbl, 'y ~ x', 'Distribution','binomial','Link','logit');
            % CC_allN(iN, iC) = mdl.Coefficients.Estimate(2);

            % Calculate neural threshold
            % should be divided by the mean firing rate; need the signal
            thresh_allN(iN, iC) = sigma_r(iN, iC) ./ max(abs(fprime(iN, iC)), eps);
        end
    end

    % Ignore the lowest and the highest contrast because they don't have fprime
    discrim_allN = 1./thresh_allN;

    fprintf(' >>> CC and Discriminability calculated\n');

    % Store boot
    CC_allBoot(iBoot, :, :) = CC_allN;
    discrim_allBoot(iBoot, :, :) = discrim_allN;

    %% Conduct and plot robust regression
    if flag_selectPosCC
        indCCpos = (CC_allN(:, indC_CC)>0);
        CC_allN = CC_allN(indCCpos, :);
        discrim_allN = discrim_allN(indCCpos, :);
    end

    x = CC_allN(:, indC_CC); % choice correlation (contrast = 0 )
    y = discrim_allN(:, indC_Dis); % discriminability (contrast = 1)
    indNotNan = (~isnan(x) & ~isnan(y) & ~isinf(x) & ~isinf(y)); % remove NaNs/Infs
    x = x(indNotNan); % remove NaNs
    y = y(indNotNan); % remove NaNs

    [b, stats] = robustfit(x, y);

    % Optional: also show nonparametric associations
    [rS, pS] = corr(x, y, 'Type','Spearman','Rows','complete');
    [rK, pK] = corr(x, y, 'Type','Kendall', 'Rows','complete');

    fprintf('>>> Robust regression conducted\n')

    % Store boot

    slope_allBoot(iBoot) = b(2);
    rS_allBoot(iBoot) = rS;
    rK_allBoot(iBoot) = rK;
    pS_allBoot(iBoot) = pS;
    pK_allBoot(iBoot) = pK;
end % end of iBoot

% Remove nan and Inf together for CC_allBoot and discrim_allBoot
CC_allBoot(isnan(CC_allBoot) | isinf(CC_allBoot)) = 0;
discrim_allBoot(isnan(discrim_allBoot) | isinf(discrim_allBoot)) = 0;

% Get median and CI of CC_allBoot and discrim_allBoot across boots
CC_median = squeeze(median(CC_allBoot, 1));
CC_ci = squeeze(bootci(1000, @median, CC_allBoot));

discrim_median = squeeze(median(discrim_allBoot, 1));
discrim_ci = squeeze(bootci(1000, @median, discrim_allBoot));

% Get median and CI for correlation
slope_med = median(slope_allBoot, 1);
rS_med = median(rS_allBoot, 1);
rK_med = median(rK_allBoot, 1);
slope_ci = squeeze(bootci(1000, @median, slope_allBoot));
rS_ci    = squeeze(bootci(1000, @median, rS_allBoot));
rK_ci    = squeeze(bootci(1000, @median, rK_allBoot));

%% Visualize and color-code three contrasts
% colors = lines(nC);
figure;
hold on;
% Each contrast
% for iC = 2:nC
%     % Plot median and CI
%     plot(discrim_median(:, iC), CC_median(:, iC), 'o', 'Color', colors(iC, :), 'DisplayName', sprintf('Contrast %.4f', contrast_all(iC)));
%     % Plot CI
% end

% Select the most relevant contrast
if flag_selectPosCC
    indCCPos = (CC_median(:, indC_CC)>0);
    CC_median = CC_median(indCCPos, :);
    discrim_median = discrim_median(indCCPos, :);
    discrim_ci = discrim_ci(:, indCCPos, :);
    CC_ci = CC_ci(:, indCCPos, :);
end

plot(discrim_median(:, indC_Dis), CC_median(:, indC_CC), 'ko', 'LineWidth', 2, 'DisplayName', 'd'' of the highest C & CC of the lowest C');

nNeurons_posCC = size(CC_median, 1);

% Draw ellipse for each neuron
edgeCol = [0.1 0.4 0.8];
faceCol = edgeCol;
alphaVal = 0.12;
nTheta   = 200;
for iN = 1:nNeurons_posCC
    x0 = discrim_median(iN,  indC_Dis);
    y0 = CC_median(iN,  indC_CC);

    % Skip if center is invalid
    if ~isfinite(x0) || ~isfinite(y0), continue; end

    % 95% CI half-widths (axis-aligned radii)
    xl = discrim_ci(1, iN,  indC_Dis);  xu = discrim_ci(2, iN,  indC_Dis);
    yl = CC_ci(1, iN,  indC_CC);       yu = CC_ci(2, iN,  indC_CC);
    if ~all(isfinite([xl xu yl yu])), continue; end

    rx = 0.5 * (xu - xl);
    ry = 0.5 * (yu - yl);
    if rx <= 0 || ry <= 0, continue; end

    % Parametric ellipse (axis-aligned)
    t  = linspace(0, 2*pi, nTheta);
    Ex = x0 + rx * cos(t);
    Ey = y0 + ry * sin(t);

    % Filled ellipse + outline
    fill(Ex, Ey, faceCol, 'FaceAlpha', alphaVal, 'EdgeColor', 'none');
    % plot(Ex, Ey, '-', 'Color', edgeCol, 'LineWidth', 1);
end

% Plot the median on top
plot(discrim_median(:, indC_Dis), CC_median(:, indC_CC), 'ko', 'LineWidth', 2, 'DisplayName', 'd'' of the highest C & CC of the lowest C');

xlabel('Discriminability (1/Neural threshold)');
ylabel('Choice correlation');

title(sprintf(['Neural threshold vs. Choice correlation\n' ...
    'Slope = %.3f [%.3f, %.3f] | Spearman ρ = %.3f [%.3f, %.3f] | Kendall τ = %.3f [%.3f, %.3f]'], ...
    slope_med, slope_ci(1 ), slope_ci(2 ), ...
    rS_med,    rS_ci(1 ),    rS_ci(2 ), ...
    rK_med,    rK_ci(1 ),    rK_ci(2 )));
