% This script simulates neural activity and computes the correlation between neurons
% Created by Shutian Xue on 08/08/2025
% Paper:   Pitkow et al. (2015) 


% Starting from the "SImulation" in EXPERIMENTAL PROCEDURES
% Parameters
N = 500; % Number of neurons
T = 30;  % Number of trials (stimulus presentations)
s_all = linspace(0, 2*pi, T); % Stimulus values for each trial
s_k = rand(N, 1) * 2 * pi;     % Preferred stimulus of each neuron

%% Simulate tuning functions
kappa = 1; % von Mises concentration
a = 24;    % Max firing rate
b = 0;     % Baseline firing rate

% Allocate firing rate matrix
r_all = zeros(N, T);

% Equation 11: Compute tuning curves
for iTrial = 1:T
    s = s_all(iTrial); % stimulus for this trial
    r_all(:, iTrial) = b + a * exp(kappa * (cos(s - s_k) - 1));
end

% Simulate spike counts (Poisson)
spike_counts = poissrnd(r_all);

% Visualize a few tuning curves
figure;
plot(s_all, r_all(1:5, :)'); % each row: firing rates for a neuron across s_vals
xlabel('Stimulus (rad)');
ylabel('Firing rate (Hz)');
title('Example Tuning Curves');
legend(arrayfun(@(i) sprintf('Neuron %d', i), 1:5, 'UniformOutput', false));

%% Compute correlation between neurons
correlationMatrix = zeros(N, N);
for i = 1:N
    for j = i+1:N
        % Compute the correlation across all trials
        correlationMatrix(i, j) = corr(spike_counts(i, :)', spike_counts(j, :)');
        correlationMatrix(j, i) = correlationMatrix(i, j); % Symmetric matrix
    end
end 