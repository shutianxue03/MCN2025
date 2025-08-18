%% Synthetic neurons with diagonal noise; unbiased linear decoder
clear all; clc; rng(0); format compact

% --------------------
% DESIGN (scalars)
% --------------------
nNeurons      = 50;          % [scalar] number of neurons (N)
nTrials       = 1000;        % [scalar] trials per contrast (T)
contrast_all  = [0, 1];      % [1 x nC] list of contrasts
nC            = numel(contrast_all);  % [scalar] number of contrasts
indC_CC       = 1;           % [scalar] index of contrast used for CC (low)
indC_Dis      = nC;          % [scalar] index of contrast used for d' (high)

% For creating a diverging Δf pattern at each contrast
df_min = [-.01, -0.3];        % [1 x nC] min Δf for each contrast
df_max = [.01, 2.5];         % [1 x nC] max Δf for each contrast


% --------------------
% STIMULUS (binary)
% --------------------
% stim_all: [T x 1] binary stimulus per trial (s ∈ {-1,+1}; -1=Left, +1=Right)
stim_all = sign(randn(nTrials,1));  % ~balanced

% --------------------
% NOISE SCALE (diagonal Σ)
% --------------------
% sigma0: [N x 1] baseline noise SD per neuron (diagonal of Σ)
% Choose ONE of the following:
% Heterogeneous noise:
% sigma0 = 1.0 + 0.3*randn(nNeurons,1); sigma0 = max(sigma0, 0.3);
% Homogeneous noise (matches your Option B derivation; Σ = σ^2 I):
sigma0 = ones(nNeurons,1);

% --------------------
% THEORETICAL Δf PER CONTRAST
% --------------------
% df_theo_allC: [N x nC] Δf_k(c) for each neuron k and contrast c
df_theo_allC = nan(nNeurons, nC);
for iContrast = 1:nC
    df_theo_allC(:, iContrast) = linspace(df_min(iContrast), df_max(iContrast), nNeurons).';
end

%% --------------------
% PREALLOCATION (shapes)
% --------------------
CC_lowC_allN_allC   = nan(nNeurons, nC);
Dis_highC_allN_allC = nan(nNeurons, nC);
w_allC              = nan(nNeurons,nC);
theta_allC          = nan(nC, 1);
CC_theo_allC        = nan(nNeurons, nC);
theta_allN          = nan(nNeurons, nC);
respNeural          = cell(nC, 1);
respBehav           = cell(nC, 1);

%% --------------------
% LOOP OVER CONTRASTS
% --------------------
for iContrast = 1:nC
    % Contrast
    contrast = contrast_all(iContrast);   % [scalar]

    % Δf at this contrast
    df = df_theo_allC(:, iContrast);      % [N x 1]
    df_norm2 = df' * df;                  % [scalar] ||df||^2
    if df_norm2 == 0
        warning('Contrast idx %d (%.3f): ||df||=0 → no signal.', iContrast, contrast);
    end

    % Diagonal noise SD vector (Σ = diag(sigma0.^2))
    SIGMA = sigma0;                        % [N x 1] SDs

    % --------------------
    % Unbiased decoder: w = df / (df' * df)
    % Ensures w'^T df = 1 ⇒ E[s_hat|s]=s when Σ ∝ I
    % --------------------
    w = df / max(df_norm2, eps);          % [N x 1]
    w_allC(:, iContrast) = w;

    % --------------------
    % Simulate population responses
    % r = df * s' + noise, noise ~ N(0, diag(SIGMA.^2))
    % --------------------
    noise = (SIGMA .* randn(nNeurons, nTrials));  % [N x T]
    r     = df * stim_all' + noise;               % [N x T]

    % Decision variable and choice
    s_hat     = w' * r;                           % [1 x T]
    choice    = sign(s_hat)';                     % [T x 1] in {-1,+1}
    
    % Residuals (remove signal) – critical for Eq.10 consistency
    r_res     = r - df * stim_all';               % [N x T] ε = r - df*s
    s_hat_res = s_hat - stim_all';                % [1 x T] w^Tε

    % Behavioral threshold (empirical SD of estimator)
    theta = std(s_hat, 0, 2);                     % [scalar]
    theta_allC(iContrast) = theta;

    % Sanity readout
    pHit = mean(choice(stim_all== 1) == 1);       % [scalar]
    pFA  = mean(choice(stim_all==-1) == 1);       % [scalar]
    fprintf('Contrast %.1f: pHit = %.2f, pFA = %.2f, w^T df = %.3f\n', ...
            contrast, pHit, pFA, w' * df);

    % Store simulated data (match your Python-like format)
    % respNeural{iC}: [N x T], respBehav{iC}: [3 x T]
    respNeural{iContrast} = r;
    respBehav{iContrast}  = [ones(1, nTrials)*contrast; choice'; stim_all']; 

    % --------------------
    % Empirical CC (noise-only): corr(ε_k, w^Tε)
    % --------------------
    for iN = 1:nNeurons
        CC_lowC_allN_allC(iN, iContrast) = corr( r_res(iN,:)', s_hat_res', ...
            'type','Pearson', 'rows','complete');
    end

    % --------------------
    % Per-neuron discriminability and neural threshold
    % d'_k = |Δf_k| / σ_k ;  θ_k = σ_k / |Δf_k|
    % --------------------
    Dis_highC_allN_allC(:, iContrast) = abs(df) ./ max(SIGMA, eps);  % [N x 1]
    theta_allN(:, iContrast)          = SIGMA ./ max(abs(df), eps);  % [N x 1]

    % --------------------
    % Theoretical CC (Option B, Σ ∝ I): C_k = f'_k / ||f'||
    % --------------------
    CC_theo_allC(:, iContrast) = df / max(norm(df), eps);            % [N x 1]
end

% Convenience aliases for plotting the “expected relationship”
CC_lowC_allN   = CC_lowC_allN_allC(:, indC_CC);       % [N x 1]
Dis_highC_allN = Dis_highC_allN_allC(:, indC_Dis);    % [N x 1]

%% --------------------
% SAVE (cells + meta)
% --------------------
save('data_sim.mat', 'respNeural', 'respBehav', 'contrast_all');

%% --------------------
% THEO vs EMP CC SCATTER (per contrast)
% --------------------
figure; hold on; box on
for iC = 1:nC
    CC_theo = CC_theo_allC(:, iC);               % [N x 1]
    CC_emp  = CC_lowC_allN_allC(:, iC);          % [N x 1]
    plot(CC_theo, CC_emp, 'o', 'DisplayName', sprintf('Contrast %.1f', contrast_all(iC)));
end
plot([-1,1], [-1,1], 'k--', 'HandleVisibility','off');
xline(0,'k-', 'HandleVisibility','off'); 
yline(0,'k-', 'HandleVisibility','off');
xlim([-1,1]); ylim([-1,1]); axis square
xlabel('Theoretical CC'); ylabel('Empirical CC (residualized)');
legend('show', 'Location', 'best')

%% --------------------
% PLOT Δf vs contrast (fan-out)
% --------------------
figure('Position',[100 100 600 400]); hold on; box on;
x_contrast = [-1.5, log10(contrast_all(2:end))]; % log axis anchor for 0
for iN = 1:nNeurons
    plot(x_contrast, df_theo_allC(iN, :), '-o');
end
errorbar(x_contrast, mean(df_theo_allC, 1), std(df_theo_allC, [], 1)/sqrt(nNeurons), ...
         'k-', 'LineWidth', 2);
xticks(x_contrast); xticklabels(string(contrast_all));
xlabel('Contrast'); ylabel('\Delta f (Spike count difference)');
xlim([-1.8, .3]); set(gca, 'FontSize', 14);

%% --------------------
% EXPECTED RELATIONSHIP: d'(high C) vs CC(low C)
% --------------------
figure('Position',[100 100 600 550]); hold on; box on; axis square
plot(Dis_highC_allN, CC_lowC_allN, 'o');
xlabel('Discriminability d'' (high contrast)');
ylabel('Choice correlation (low contrast)');
title('Expected: higher d'' \rightarrow stronger CC');