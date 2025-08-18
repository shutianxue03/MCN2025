%% IBL analysis pipeline: tuning, CC theory vs empirical, CC vs discriminability
% Assumes variables exist in workspace:
%   respNeural: {1 x nC}, each [nNeurons x nTrials_c]
%   respBehav:  {1 x nC}, each [3 x nTrials_c] = [contrast; choice; stim]
%   contrast_all: [1 x nC]
% ----------------------------------------------------------------------------
clear all, clc, rng(0); format compact

sz_font = 18;
%% --- Config ---
load('data_CP_bias0.5_QCFalse.mat'), str_fig = 'Data';
contrast_all  = [0, .0625, .125, .25, 1];      % [1 x nC] list of contrasts

load('data_sim.mat'); str_fig = 'Sim';
% contrast_all  = [0, 1];      % [1 x nC] list of contrasts; automatically loaded

nC            = numel(contrast_all);  % [scalar] number of contrasts
indC_CC  = 1;                     % index of contrast for CC (use a LOW contrast)
indC_Dis = numel(contrast_all);   % index of contrast for discriminability (use a HIGH contrast)

%% --- Infer sizes ---
[nNeurons, ~] = size(respNeural{1});     % neurons (assumed constant across contrasts)

%% --- Preallocate ---
df_est_allC      = nan(nNeurons, nC);    % Δf per neuron per contrast
Sigma_allC       = cell(nC,1);           % noise covariance per contrast (from residuals)
w_opt_allC       = nan(nNeurons, nC);    % optimal unbiased weights per contrast
theta_behav_allC = nan(nC,1);            % behavioral threshold (SD of s_hat) per contrast
CC_emp_allC      = nan(nNeurons, nC);    % empirical CC per contrast
CC_theo_allC     = nan(nNeurons, nC);    % theoretical CC per contrast
dprime_allC      = nan(nNeurons, nC);    % per-neuron discriminability (|Δf|/σ_k)
theta_neur_allC  = nan(nNeurons, nC);    % per-neuron neural threshold (σ_k/|Δf|)

%% --- Loop contrasts: estimate Δf, noise covariance, decoder, CCs ---
for iC = 1:nC
    respNeural_perC = double(respNeural{iC});     % [N x T_c] spike counts / rates
    respBehav_perC = double(respBehav{iC});      % [3 x T_c] [contrast; choice; stim]
    stim_perC = respBehav_perC(3, :)';                  % [T_c x 1] stimulus in {-1,+1}
    nTrials_perC = size(respNeural_perC,2);

    % 1) Estimate tuning slope (binary) as class-conditional mean difference: Δf = E[r|s=+1]-E[r|s=-1]
    indLeft = (stim_perC == +1);
    indRight = (stim_perC == -1);

    muLeft = mean(respNeural_perC(:, indLeft), 2, 'omitnan');   % [N x 1]
    muRight = mean(respNeural_perC(:, indRight), 2, 'omitnan');   % [N x 1]
    df  = muLeft - muRight;                        % [N x 1]
    df_est_allC(:, iC) = df;

    % 2) Residuals (remove signal) and noise covariance (estimated at fixed s)
    respNeural_perC_res = respNeural_perC - df * stim_perC';  % [N x T_c], ε = r - df*s

    % Covariance across trials (rows = trials): cov expects [T x N]
    Sigma = cov(respNeural_perC_res');                    % [N x N]
    
    % Regularize slightly if needed for stability
    Sigma = (Sigma + Sigma')/2;
    tiny = 1e-8 * trace(Sigma)/nNeurons;
    Sigma = Sigma + tiny*eye(nNeurons);
    Sigma_allC{iC} = Sigma;

    % 3) Derive the optimal unbiased linear decoder
    %    w = Σ^{-1} df / (df' Σ^{-1} df), so that E[w^T r | s] = s
    df_iszero = (norm(df)==0);
    if df_iszero
        w = zeros(nNeurons,1);
    else
        Sigma_inv = pinv(Sigma);                         % robust inverse
        denumerator    = (df' * Sigma_inv * df);                % scalar
        w      = (Sigma_inv * df) / max(denumerator, eps);      % [N x 1]
    end
    w_opt_allC(:, iC) = w;

    % 4) Decision variable and residual decision noise
    s_hat     = w' * respNeural_perC;                 % [1 x T_c]
    s_hat_res = s_hat - stim_perC';             % [1 x T_c], = w^T ε
    theta_behav_allC(iC) = std(s_hat, 0, 2);   % behavioral SD (could also use std(s_hat_res))

    % 5) Theoretical CC (Eq. 10, general Σ): C_k = f'_k / ( sqrt(Σ_kk) * sqrt(df' Σ^{-1} df) )
    sig_k = sqrt(max(diag(Sigma), 0));         % [N x 1] per-neuron noise SD
    denumerator   = max(df' * (pinv(Sigma)) * df, eps);
    CC_theo_allC(:, iC) = df ./ ( sig_k * sqrt(denumerator) );   % [N x 1]

    % 6) Empirical CC (noise-only): corr(ε_k, w^T ε) across trials
    for iN = 1:nNeurons
        CC_emp_allC(iN, iC) = corr( respNeural_perC_res(iN,:)', s_hat_res', 'type','Pearson', 'rows','complete');
    end

    % 7) Per-neuron discriminability and neural threshold
    %    d'_k = |Δf_k| / σ_k,    θ_k = σ_k / |Δf_k|
    dprime_allC(:, iC)     = abs(df) ./ max(sig_k, eps);
    theta_neur_allC(:, iC) = sig_k ./ max(abs(df), eps);
end


fprintf('\n ==== DONE ==== \n')

%% (1) Estimated tuning function (Δf vs contrast)
figure('Position',[100 100 750 420]); hold on; box on;
x = [-1.5, log10(contrast_all(2:end))];                          % [1 x nC]
for iN = 1:nNeurons
    plot(x, df_est_allC(iN, :), '-', 'Color', [0.5 0.5 0.5]);
end
plot(x, mean(df_est_allC,1), 'k-', 'LineWidth', 2);
sem = std(df_est_allC,[],1)/sqrt(nNeurons);
errorbar(x, mean(df_est_allC,1), sem, 'k', 'LineWidth', 1.5, 'CapSize', 0);
xticks(x);
xticklabels(contrast_all);
xlabel('Contrast'); ylabel('\Delta f (r_{s=+1} - r_{s=-1})');
title('Estimated tuning (Δf) across contrasts');
set(gca, 'FontSize', sz_font);
saveas(gcf, sprintf('%s_DeltaF.png', str_fig))

%% (2) Empirical vs theoretical CC (per contrast)
figure('Position',[100 100 900 900]); hold on; box on; axis square
for iC = 1:nC
    % subplot(1, nC, c); hold on; box on; axis square
    plot(CC_theo_allC(:,iC), CC_emp_allC(:,iC), 'o', 'displayname', sprintf('Contrast %.3g', contrast_all(iC)));
end
plot([-1 1], [-1 1], 'r--', 'HandleVisibility', 'off');
xlim([-1 1]); ylim([-1 1]);
xlabel('CC theory'); ylabel('CC empirical');
title('Theoretical vs Empirical Choice Correlation');
legend('show', 'Location', 'Best');
set(gca, 'FontSize', sz_font);
set(gca, 'linewidth', 2);
saveas(gcf, sprintf('%s_CC_Emp_Theo.png', str_fig))

%% (3) Empirical CC (low C) vs discriminability (high C)
CC_low   = CC_emp_allC(:, indC_CC);     % [N x 1]
dprime_hi= dprime_allC(:, indC_Dis);    % [N x 1]

CC_low   = CC_lowC_allN_allC(:, indC_CC);       % [N x 1]
dprime_hi = Dis_highC_allN_allC(:, indC_Dis);    % [N x 1]

good     = isfinite(CC_low) & isfinite(dprime_hi);

% Robust correlation (Spearman or robustfit)
[r_pearson, p_pearson] = corr(dprime_hi(good), CC_low(good), ...
                              'type','Pearson');
[r_spearman, p_spearman] = corr(dprime_hi(good), CC_low(good), ...
                                'type','Spearman');

% Optionally: robust linear regression
b = robustfit(dprime_hi(good), CC_low(good));
yfit = b(1) + b(2)*dprime_hi(good);

figure('Position',[100 100 520 520]); hold on; box on; axis square
plot(dprime_hi(good), CC_low(good), 'o');
plot(dprime_hi(good), yfit, 'r-', 'LineWidth',2); % robust fit line

xlabel('Discriminability d'' (high contrast)');
ylabel('Choice correlation (low contrast)');

title(sprintf('Choice correlation vs. Discriminability \nPearson r=%.2f (p=%.3g)\nSpearman ρ=%.2f (p=%.3g)', ...
              r_pearson, p_pearson, r_spearman, p_spearman));

set(gca, 'FontSize', sz_font);
saveas(gcf, sprintf('%s_CC_Dis.png', str_fig));