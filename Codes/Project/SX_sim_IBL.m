%% Synthetic neurons with diagonal noise; unbiased linear decoder
clear all; clc; rng(0); format compact

% --------------------
% DESIGN
% --------------------
nNeurons = 50; % neurons
nTrials = 1000; % trials per contrast
contrast_all = [0, 1]; % contrasts
nC = numel(contrast_all);
indC_CC = 1; indC_Dis = nC;

df_max = [.01, 2.5];
df_min = [-.01, 0.3];

% Sample stimulus (binary around s0): s in {-1, +1}, -1=Left, 1=right
stim_all = sign(randn(nTrials,1)); % balanced on average

% neuron “tuning strengths” (Δf) at unit contrast; create a diverging pattern
% base_df = linspace(.001, 2.5, nNeurons)'; % some neurons near 0, others larger
% Hetergeneous noise
sigma0 = 1.0 + 0.3*randn(nNeurons,1); % baseline noise SD per neuron
% Homogeneous noise
sigma0 = ones(nNeurons,1); % baseline noise SD per neuron

% Optional: some untuned neurons
% untuned_frac = 0.25;
% isUntuned = false(nNeurons,1);
% isUntuned(randperm(nNeurons, round(untuned_frac*nNeurons))) = true;
% base_df(isUntuned) = 0;

% Theoretical df
df_theo_allC = nan(nNeurons, nC);
for iContrast = 1:nC
    df_theo_allC(:, iContrast) = linspace(df_min(iContrast), df_max(iContrast), nNeurons);
end

%% LOOP OVER CONTRASTS: simulate population responses & behavior
% Preallocate
% CP_low = nan(nNeurons,1); % CP at low contrast (we'll use contr(2))
CC_lowC_allN = nan(nNeurons,1); % CC at low contrast (Pearson)
CC_lowC_allN_allC = nan(nNeurons, nC); % store CC at low contrast for all contrasts
Dis_highC_allN = nan(nNeurons,1); % discriminability at high contrast
Dis_highC_allN_allC = nan(nNeurons, nC); % discriminability at high contrast for all contrasts
w_allC = nan(nNeurons,nC); % decoder weights per contrast (just for inspection)
% df_theo_allC = nan(nNeurons, nC);
theta_allC = nan(nC, 1); % behav threshold
CC_theo_allC = nan(nNeurons, nC); % theoretical choice correlation
respNeural = cell(nC, 1);
respBehav = cell(nC, 1);

for iContrast = 1:nC
    % Contrast
    contrast = contrast_all(iContrast);

    % Define the df at this contrast
    df = df_theo_allC(:, iContrast);


    % Define the covariance matrix
    SIGMA = sigma0; % keep noise independent of contrast

    % Construct unbiased optimal decoder for diagonal Σ:
    % w ∝ f', with normalization so that w^T f' = 1 (i.e., <s_hat|s> = s)
    % If Σ = diag(sig.^2), the full MLE would be w ∝ Σ^{-1} f' = f'./sig.^2,
    % but your note assumes Σ diagonal + use w = f'/||f'||^2 to satisfy w^T f' = 1.
    % if norm(df) > 0
    w = df / (df' * df); % unbiased normalization
    % else % if norm(df)==0, which is the case when contrast=0
    %     fprintf('norm(df)==0\n')
    %     w = zeros(nNeurons,1);
    % end
    w_allC(:, iContrast) = w;

    % Generate responses for T trials:
    % r = f' * s + noise, with noise ~ N(0, diag(sig.^2))
    noise = (SIGMA .* randn(nNeurons,nTrials)); % independent across neurons
    r = df * stim_all' + noise; % [N x T]

    % Behavioral estimate and choice:
    % s_hat = w' * r; % [1 x T]
    % choice = sign(s_hat)'; % {-1,+1}; ties rare
    if norm(df) > 0
        s_hat = w' * r;
        r_res      = r - df * stim_all';         % subtract class means: ε = r - f's
        s_hat_res  = s_hat - stim_all';          % noise in decision variable: w^T ε
        choice = sign(s_hat)';
    else
        s_hat = zeros(1,nTrials);           % no signal
        choice = randi([0 1], nTrials, 1)*2 - 1;  % random {-1,+1}
    end

    % Obtain behavioral threshold
    theta = std(s_hat);
    theta_allC(iContrast) = theta;

    % Calculate and print pHit and pFA
    pHit = mean(choice(stim_all==1) == 1);
    pFA = mean(choice(stim_all==-1) == 1);
    fprintf('Contrast %.1f: pHit = %.2f, pFA = %.2f\n', contrast, pHit, pFA);

    % Store neural * behav response
    respNeural{iContrast} = r;
    respBehav{iContrast} = [ones(1, nTrials)*contrast; choice'; stim_all']; % row1: contrast per trial, row2: choice per trial; row3: stim per trial

    %  Compute CC and discriminability
    for iN = 1:nNeurons

        CC_lowC_allN_allC(iN, iContrast) = corr(r_res(iN,:)', s_hat_res', 'type','Pearson','rows','complete');
    end

    % d'_k = |Δf_k| / σ_k since s ∈ {-1,+1} ⇒ Δf = 2 f'
    % For binary s with equal prob, a common single-neuron d' is |f'|/σ (up to a 2 factor).
    % Dis_highC_allN_allC(:, iContrast) = abs(df) ./ SIGMA;
    Dis_highC_allN_allC(:, iContrast) = abs(df) ./ max(SIGMA, eps);  % avoid div-by-zero

    % Behavioral threshold
    theta_allN(:, iContrast) = SIGMA ./ max(abs(df), eps);

    CC_theo_allC(:, iContrast) = df / norm(df);

end % end of iContrast

%% Save the synthetic data
save('data_sim.mat', 'respNeural', 'respBehav', 'contrast_all');

%% Compare theo CC and emp CC
figure, hold on
for iC=1:nC
    CC_theo = CC_theo_allC(:, iC);
    CC_emp = CC_lowC_allN_allC(:, iC);
    plot(CC_theo, CC_emp, 'o', 'DisplayName', sprintf('Contrast %.1f', contrast_all(iC)));
end
xlim([-1,1])
ylim([-1,1])
plot([-1,1], [-1,1], 'k--')
xline(0, 'k-')
yline(0, 'k-')
legend('show', 'Location', 'best')
xlabel('Theoretical CC')
ylabel('Empirical CC')

%% Plot fprime
figure('Position',[100 100 600 400]);
hold on; box on;
x_contrast = [-1.5, log10(contrast_all(2:end))]; % because log10(0) is inf
for iN = 1:nNeurons
    plot(x_contrast, df_theo_allC(iN, :), '-o');
end
errorbar(x_contrast, mean(df_theo_allC, 1), std(df_theo_allC, [], 1)/sqrt(nNeurons), 'k-', 'LineWidth', 2);
xticks(x_contrast)
xticklabels(string(contrast_all))
xlabel('Contrast')
ylabel('$\Delta f$ (Spike count difference)', 'Interpreter', 'latex');
xlim([-1.8, .3])
% ylim([-1, 1])
set(gca, 'FontSize', 20)  % Sets font size for axis tick labels

%% PLOTS / EXPECTED RELATIONSHIP
figure('Position',[100 100 1100 420]);
hold on
plot(Dis_highC_allN_allC(:, indC_Dis), CC_lowC_allN_allC(:, indC_CC), 'o');
axis square
legend('show', 'Location', 'best')
xlabel('Discriminability d'' (high contrast)'); ylabel('CC (low contrast)');
title('Expected: higher d'' → stronger CC');
