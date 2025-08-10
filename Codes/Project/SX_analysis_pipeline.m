%% Mice L/R grating dataset: metrics + hypothesis tests
% Requires: data_mice.mat with variables:
%   data_mice : [Nneurons x Ntrials] firing rates (or event rates) per trial
%   stim_side : [1 x Ntrials]   -1 = LEFT, +1 = RIGHT (ground truth stimulus location)
%   contrast  : [1 x Ntrials]   scalar >= 0
%   choice    : [1 x Ntrials]   -1 = chose LEFT, +1 = chose RIGHT (mouse report)
%
% MATLAB toolboxes: Statistics and Machine Learning

clear; clc; close all;

%% Load (synthetic) data
S = load('data_mice.mat');
dataNeural = S.data_mice; % N x T 
stim_side = S.stim_side(:)';  % 1 x T
stimLevel  = S.contrast(:)';   % 1 x T
dataBehav    = S.choice(:)';     % 1 x T

[N,T] = size(dataNeural);

% Signed stimulus: s = contrast * side (right positive, left negative)
s_signed = stimLevel .* stim_side;  % 1 x T

%% 1. Obtain population/behav threshold by fitting psychometric function
% Fit cumulative Gaussian via probit GLM: P(right) = Phi(beta0 + beta1 * s)
tbl = table(s_signed', dataBehav'==+1, 'VariableNames', {'s','right'});
glm_beh = fitglm(tbl, 'right ~ s','Distribution','binomial','Link','probit');
beta = table2array(glm_beh.Coefficients(:,1));
beta0 = beta(1);  % intercept
beta1 = beta(2);  % slope/sigma

% Behavioral threshold θ = change in s that moves P from 0.5 to 0.84 (one SD)
% Under probit, that SD = 1/|beta1|
theta_behav = 1/abs(beta1);  % psychophysical σ (68% correct in 2AFC symmetric case)
fprintf('Behavioral threshold θ (probit SD): %.4f (signed-contrast units)\n', theta_behav);

%% 2. Obtain neural threshold by fitting neurometric function
% For each neuron, fit P(right | firing) using probit; define θ_k analogously as 1/|w1|
theta_neu = nan(N,1);
pref_sign = nan(N,1);  % +1 means more spikes => predicts RIGHT
y_bin = dataBehav==+1;
for k = 1:N
    r = dataNeural(k,:)';
    try
        glm_k = fitglm(table(r,y_bin'),'y_bin ~ r','Distribution','binomial','Link','probit');
        b = table2array(glm_k.Coefficients(:,1));
        b1 = b(2);
        theta_neu(k) = 1/abs(b1);  % neurometric SD (proxy for slope at boundary)
        pref_sign(k) = sign(b1);   % which side neuron tends to support
    catch
        theta_neu(k) = NaN;
        pref_sign(k) = NaN;
    end
end

valid = isfinite(theta_neu) & isfinite(CP);
fprintf('Neurometric fits: %d/%d valid neurons\n', nnz(valid), N);

%% 3. Calculate choice probabilities (CP) and choice correlation (CC)
% Use "near-threshold" trials: the lowest tertile of |s|
q = quantile(abs(s_signed), 1/3);
idx_th = abs(s_signed) <= q & stimLevel>0;  % exclude zero-contrast edge cases if any
if nnz(idx_th) < 50, warning('Few near-threshold trials (%d). CPs may be noisy.', nnz(idx_th)); end

CP = nan(N,1);
for k = 1:N
    r = dataNeural(k,idx_th)';
    y = dataBehav(idx_th)';              % -1/ +1
    % perfcurve expects binary labels; convert to 0/1 with "right" as positive class
    labels = y==+1;
    scores = r;                       % higher firing predicts RIGHT if neuron is "right-preferring"
    [~,~,~,auc] = perfcurve(labels,scores,true);
    CP(k) = auc;                      % choice probability (ROC area)
end

% Convert CP to CC for all neurons
% Pitkow p15: Haefner et al., 2013
CC = (pi/sqrt(2))./(CP-.5);

%% ---------- D) Test H1 vs H2 using CP vs θ/θ_k ----------
% we used linear regression to fit coefficients for optimal and suboptimal predictors of choice correlation to determine whether choice correlations were better predicted by optimal or suboptimal decoding.

% Optimal-decoder prediction (Pitkow et al.): CP_k ≈ θ/θ_k (scaled)
x_opt = theta_behav ./ theta_neu(valid);
y_cp  = CP(valid);

% Regress CP on optimal predictor and on an intercept to capture overall bias
mdl_opt = fitlm(x_opt, y_cp,'Intercept',false); % through origin: slope should be ~1 if perfectly optimal
slope_opt = mdl_opt.Coefficients.Estimate;
ci_opt = coefCI(mdl_opt);
r2_opt = mdl_opt.Rsquared.Ordinary;

fprintf('CP ~ (θ/θ_k): slope=%.3f (95%% CI [%.3f, %.3f]), R^2=%.3f\n',...
    slope_opt, ci_opt(1,1), ci_opt(1,2), r2_opt);

% (Optional) Compare against a "suboptimal" proxy: correlation-blind predictor = |pref_sign|
% In a strict 2-class setting this carries little structure, but include for completeness.
x_sub = abs(pref_sign(valid));  % mostly 1—will have low explanatory power unless diversity exists
if var(x_sub)>0
    mdl_sub = fitlm(x_sub, y_cp);
    fprintf('CP ~ |pref_sign|: slope=%.3f, R^2=%.3f\n', mdl_sub.Coefficients.Estimate(2), mdl_sub.Rsquared.Ordinary);
else
    fprintf('Suboptimal proxy has no variance (|pref_sign| constant); skipped.\n');
end

%% ---------- E) Population decoding vs neuron count (real vs shuffled) ----------
% Goal: show saturation (real) vs steady improvement (shuffled) if noise correlations limit info.
% Decoder: linear SVM to predict RIGHT vs LEFT from population activity.
% Cross-validate across 10 folds; evaluate accuracy by signed-contrast bins and by neuron count.
rng(1);
Kfold = 10;
folds = cvpartition(T,'KFold',Kfold);

% Build condition strata for shuffling: group by (stim_side, contrast bin)
nC = 5; % contrast bins
edges = quantile(stimLevel, linspace(0,1,nC+1));
cond_id = 1 + discretize(stimLevel, edges); % 1..nC
cond_id(isnan(cond_id)) = 1;
strata = stim_side.*(nC) + cond_id; % combine

% neuron-count sweep
counts = unique(round(logspace(log10(max(10,min(60,N))), log10(min(300,N)), 8)));
acc_real  = zeros(numel(counts), Kfold);
acc_shuff = zeros(numel(counts), Kfold);

for ii = 1:numel(counts)
    p = counts(ii);
    for kfold = 1:Kfold
        tr = training(folds,kfold); te = test(folds,kfold);
        idx_neu = randsample(N, p);

        Xtr = dataNeural(idx_neu, tr)';   Ytr = stim_side(tr)==+1; % predict RIGHT
        Xte = dataNeural(idx_neu, te)';   Yte = stim_side(te)==+1;

        % Standardize
        mu = mean(Xtr,1); sd = std(Xtr,[],1)+1e-9;
        Xtrz = (Xtr-mu)./sd; Xtez = (Xte-mu)./sd;

        % Train linear SVM (hinge)
        mdl = fitcsvm(Xtrz, Ytr,'KernelFunction','linear','Standardize',false,'ClassNames',[false true]);
        acc_real(ii,kfold) = mean(predict(mdl, Xtez) == Yte);

        % --- Shuffle (remove noise correlations): independently permute trials within (side,contrast-bin) for each neuron
        Xtr_sh = Xtr;
        Xte_sh = Xte;
        for n = 1:p
            % shuffle training data within strata
            for ss = [-1 1]
                for cbin = 1:nC
                    mask_tr = tr & (stim_side==ss) & (cond_id==cbin);
                    idx_tr = find(mask_tr);
                    if numel(idx_tr)>1
                        perm = idx_tr(randperm(numel(idx_tr)));
                        Xtr_sh( idx_tr, n) = Xtr(ismember(find(tr), perm)-0, n); %#ok<*ISMAT>
                    end
                    mask_te = te & (stim_side==ss) & (cond_id==cbin);
                    idx_te = find(mask_te);
                    if numel(idx_te)>1
                        perm2 = idx_te(randperm(numel(idx_te)));
                        Xte_sh( idx_te, n) = Xte(ismember(find(te), perm2)-0, n);
                    end
                end
            end
        end

        Xtrz_sh = (Xtr_sh-mu)./sd; Xtez_sh = (Xte_sh-mu)./sd;
        mdl_sh = fitcsvm(Xtrz_sh, Ytr,'KernelFunction','linear','Standardize',false,'ClassNames',[false true]);
        acc_shuff(ii,kfold) = mean(predict(mdl_sh, Xtez_sh) == Yte);
    end
end

mean_real  = mean(acc_real,2);  sem_real  = std(acc_real,[],2)/sqrt(Kfold);
mean_shuff = mean(acc_shuff,2); sem_shuff = std(acc_shuff,[],2)/sqrt(Kfold);

%% ---------- F) Simple saturation test ----------
% Fit a saturating curve: A*n / (1 + n/N0) to accuracy (after logit transform to stabilize)
f_sat = @(p,n) 1./(1+exp(-(p(1)*n./(1+n/p(2)) + p(3)))); % logistic of saturating growth
opts = statset('nlinfit'); opts.RobustWgtFun = 'cauchy';

p0 = [ (mean_real(end)-mean_real(1))*2,  max(counts)/3, logit(mean_real(1)+0.01) ];
p_real = nlinfit(counts, mean_real, f_sat, p0, opts);
p_shuf = nlinfit(counts, mean_shuff, f_sat, p0, opts);

% Where does each curve reach 95% of its asymptote?
asym_real  = max(f_sat(p_real, 1e6), f_sat(p_real, max(counts)));
asym_shuf  = max(f_sat(p_shuf, 1e6), f_sat(p_shuf, max(counts)));
n95_real = fminbnd(@(n) (f_sat(p_real,n) - (0.95*asym_real)).^2, counts(1), 1e6);
n95_shuf = fminbnd(@(n) (f_sat(p_shuf,n) - (0.95*asym_shuf)).^2, counts(1), 1e6);

fprintf('Saturation (95%% of asymptote): real n≈%.0f, shuffled n≈%.0f (larger means less saturation)\n',...
    n95_real, n95_shuf);

%% ---------- G) Plots ----------
figure('Color','w','Position',[50 50 1200 700]);

subplot(2,3,1);
edgesS = linspace(min(s_signed),max(s_signed),9);
[~,~,bin] = histcounts(s_signed, edgesS);
propR = accumarray(bin(:), (dataBehav==+1)', [], @mean);
centS = movmean(edgesS,2,'Endpoints','discard');
plot(centS, propR,'ko-','MarkerFaceColor','k'); hold on;
xx = linspace(min(s_signed),max(s_signed),200);
plot(xx, normcdf(beta0 + beta1*xx), 'r-', 'LineWidth',2);
xlabel('signed contrast (s)'); ylabel('P(right)'); title(sprintf('Psychometric (\\theta=%.3f)', theta_behav)); grid on;

subplot(2,3,2);
scatter(theta_behav./theta_neu(valid), CP(valid), 12, 'filled'); hold on;
xlim([0 prctile(theta_behav./theta_neu(valid), 99)]); ylim([0.4 0.7]);
lsline; % OLS line through origin handled above
text(0.02,0.68, sprintf('slope=%.2f, R^2=%.2f', slope_opt, r2_opt),'Units','normalized');
xlabel('\theta / \theta_k'); ylabel('CP_k'); title('CP vs optimal predictor'); grid on;

subplot(2,3,3);
errorbar(counts, mean_real, sem_real,'-o','LineWidth',1.5); hold on;
errorbar(counts, mean_shuff, sem_shuff,'-s','LineWidth',1.5);
nn = linspace(min(counts), max(counts), 300);
plot(nn, f_sat(p_real,nn),'b--'); plot(nn, f_sat(p_shuf,nn),'r--');
legend({'Real','Shuffled','Real fit','Shuffled fit'},'Location','southeast'); grid on;
xlabel('# neurons'); ylabel('Decoding accuracy (RIGHT vs LEFT)');
title(sprintf('Population decoding (n_{95} real=%.0f, shuf=%.0f)', n95_real, n95_shuf));

subplot(2,3,4);
histogram(CP(valid), 'FaceColor',[.2 .2 .7]); xlabel('CP'); ylabel('# neurons'); title('Choice probabilities');

subplot(2,3,5);
histogram(theta_neu(valid), 30,'FaceColor',[.4 .7 .4]); xlabel('\theta_k'); title('Neurometric thresholds');

subplot(2,3,6);
plot(counts, mean_shuff-mean_real,'k-o','LineWidth',1.5); grid on;
xlabel('# neurons'); ylabel('Shuffled - Real accuracy');
title('Benefit of removing noise correlations');

%% ---------- H) Hypothesis test summaries ----------
% H1 support if: slope_opt ~ 1 (or clearly >0 with decent R^2) AND n95_real << n95_shuf (saturation in real)
% H2 support if: shuffled grows faster and does not saturate in range; CPs poorly explained by θ/θ_k

fprintf('\n--- Summary ---\n');
if slope_opt > 0.5 && r2_opt > 0.1
    fprintf('CPs align with optimal prediction (θ/θ_k). Supports H1.\n');
else
    fprintf('CPs weakly align with θ/θ_k. Weaker support for H1.\n');
end

if n95_real < n95_shuf && mean_shuff(end) > mean_real(end)
    fprintf('Decoding saturates in REAL but keeps improving in SHUFFLED. Supports H1 (info-limiting noise).\n');
else
    fprintf('No clear saturation advantage for REAL over SHUFFLED in this neuron range. Weaker H1 / possible H2.\n');
end

%% ---------- Helper ----------
function y = logit(p), y = log(p./(1-p)); end