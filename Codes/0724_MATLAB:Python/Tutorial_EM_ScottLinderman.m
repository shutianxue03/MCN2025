% EM Algorithm for 1D Gaussian Mixture Model

clear; clc; rng(1);  % For reproducibility

%% Step 1: Simulate data from a 2-component Gaussian mixture
N = 500;                      % Number of data points
true_mu = [-2, 3];            % True means
true_sigma = [1, 0.7];        % True std deviations
true_pi = [0.4, 0.6];         % True mixing weights

% Sample data
z = rand(N,1) < true_pi(1);   % Component assignment
x = z .* (true_sigma(1)*randn(N,1) + true_mu(1)) + ...
    ~z .* (true_sigma(2)*randn(N,1) + true_mu(2));

% Visalize the simulated data
figure;
histogram(x, 30, 'Normalization', 'pdf'); hold on;
x_vals = linspace(min(x)-1, max(x)+1, 1000);
pdf_true = true_pi(1)*normpdf(x_vals, true_mu(1), true_sigma(1)) + ...
           true_pi(2)*normpdf(x_vals, true_mu(2), true_sigma(2));
plot(x_vals, pdf_true, 'r--', 'LineWidth', 2);
title('Simulated Data from 2-Component GMM');
legend('Data Histogram','True GMM');
xlabel('x'); ylabel('Probability Density');

%% Step 2: Initialize EM parameters
K = 2;                        % Number of components
mu = randn(1, K);             % Random initial means
sigma = ones(1, K);           % Initial std dev
pi_k = ones(1, K)/K;          % Equal weights
log_likelihoods = [];

%% Step 3: Run EM algorithm
max_iter = 100; % maximum iterations
tol = 1e-6; % convergence tolerance
N = length(x); % number of data points

for iter = 1:max_iter
    % === E-Step ===
    gamma = zeros(N, K); % responsibility matrix, which holds the posterior probabilities, [N,K]
    % Compute responsibilities
    for k = 1:K
        gamma(:,k) = pi_k(k) * normpdf(x, mu(k), sigma(k));
    end
    gamma = gamma ./ sum(gamma,2);

    % visualize gamma
    figure;  hold on;
    plot(x, gamma(:,1), 'ro', 'DisplayName', 'Gamma 1');
    plot(x, gamma(:,2), 'bo', 'DisplayName', 'Gamma 2');
    title(['E-Step: Responsibilities at Iteration ' num2str(iter)]);
    xlabel('x'); ylabel('Responsibility');
    legend show;

    % === M-Step ===
    Nk = sum(gamma, 1);
    % Update parameters
    for k = 1:K
        mu(k) = sum(gamma(:,k) .* x) / Nk(k);
        sigma(k) = sqrt(sum(gamma(:,k) .* (x - mu(k)).^2) / Nk(k));
        pi_k(k) = Nk(k) / N;
    end

    % === Log-Likelihood ===
    ll = sum(log(sum(gamma,2)));
    log_likelihoods = [log_likelihoods; ll];

    % Check for convergence
    if iter > 1 && abs(log_likelihoods(end) - log_likelihoods(end-1)) < tol
        break
    end
end

%% Step 4: Visualize results
x_vals = linspace(min(x)-1, max(x)+1, 1000);
pdf_est = zeros(1, length(x_vals));
for k = 1:K
    pdf_est = pdf_est + pi_k(k) * normpdf(x_vals, mu(k), sigma(k));
end

figure;
histogram(x, 30, 'Normalization', 'pdf'); hold on;
plot(x_vals, pdf_est, 'k-', 'LineWidth', 2);
title('EM Algorithm - Gaussian Mixture Model');
legend('Data Histogram','Estimated GMM');
xlabel('x'); ylabel('Probability Density');

%% Display estimated parameters
fprintf('Estimated means: %.2f, %.2f\n', mu(1), mu(2));
fprintf('Estimated std devs: %.2f, %.2f\n', sigma(1), sigma(2));
fprintf('Estimated mixing weights: %.2f, %.2f\n', pi_k(1), pi_k(2));