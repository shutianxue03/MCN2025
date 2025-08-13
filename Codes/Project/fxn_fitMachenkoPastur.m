%% 4b. Fit Marchenko-Pastur Distribution to Eigenvalues

% --- Step 1: Normalize the Empirical Covariance Matrix ---
% This is crucial for comparing to the standard Marchenko-Pastur distribution.
% We scale the matrix so the average variance on the diagonal is 1.
avg_variance = mean(diag(SIGMA_emp));
SIGMA_emp_norm = SIGMA_emp / avg_variance;

% Get the eigenvalues of the NORMALIZED matrix
eigenvalues_norm = eig(SIGMA_emp_norm);

% --- Step 2: Define Parameters for the Theoretical Distribution ---
N = nNeurons;
T = nTrials; % Make sure this is the number of trials used for cov()
gamma = N / T;
sigma2 = 1; % The variance of the underlying data, which is 1 after our normalization

% Define the theoretical bounds of the distribution
lambda_plus = sigma2 * (1 + sqrt(gamma))^2;
lambda_minus = sigma2 * (1 - sqrt(gamma))^2;

% --- Step 3: Define the Marchenko-Pastur PDF ---
% This is the function for the theoretical curve.
fxn_mp_pdf = @(x) (1 ./ (2 * pi * sigma2 * gamma * x)) .* sqrt((lambda_plus - x) .* (x - lambda_minus));

% Create a range of x-values for plotting the theoretical curve
x_theory = linspace(lambda_minus, lambda_plus, 1000);
y_theory = fxn_mp_pdf(x_theory);

% --- Step 4: Plot the Empirical Histogram and the Theoretical PDF ---
figure;
hold on;

% Plot the histogram of your normalized eigenvalues.
% Using 'pdf' normalization scales the histogram so its area is 1.
histogram(eigenvalues_norm, 20, 'Normalization', 'pdf', 'FaceColor', [0.5 0.7 1.0], 'EdgeColor', 'none');

% Plot the theoretical Marchenko-Pastur curve on top.
plot(x_theory, y_theory, 'r-', 'LineWidth', 2);

% Formatting
title('Marchenko-Pastur Fit to Empirical Eigenvalues');
xlabel('Eigenvalue');
ylabel('Probability Density');
legend('Empirical Eigenvalues', 'Marchenko-Pastur Fit');
grid on;
axis tight;