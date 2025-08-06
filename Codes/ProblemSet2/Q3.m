% The Bienenstock Cooper Munro (BCM) Rule
clear all
clc
close all

%% 3.1
rule = 'BCM'; % learning rule: 'BCM' or 'Hebbian'
% rule = 'Hebbian'; % Uncomment to use Hebbian learning rule

% Inputs
x1 = [20, 0]'; % firing rate in Hz
x2 = [0, 20]';

eta = 1e-6; % learning rate, in Hz
y0 = 20; % initial threshold in Hz
tau = 50; % time constant in ms

T= 1e4; % duration in ms
dt = 1; % time step in ms
t_all = 0:dt:T; % time vector in ms

% Preallocate weight matrix
W_all = rand(2, length(t_all)); % preallocate input vector
x_all = zeros(2, length(t_all)); % preallocate input vector
y_all = zeros(1, length(t_all)); % preallocate output vector
theta_all = zeros(1, length(t_all)); % preallocate threshold vector

% Initiate a random weight matrix
W0 = rand(2, 1);
% W0 = [.2,1;1,.7];
W_all(:, 1) = W0; % set the first weight matrix

for t = 1:length(t_all)-1

    % Decide the input pattern
    if rand <.5
        x_all(:, t) = x1;
    else
        x_all(:, t) = x2;
    end

    % Calculate the output of the neuron
    y_all(t+1) = W_all(:, t).' * x_all(:, t);

    % Update the weights according to a rule
    switch rule
        case 'BCM'
            % Calculate the threshold theta
            dtheta = (-theta_all(t) + y_all(t).^2 / y0)/tau;
            theta_all(t+1) = theta_all(t) + dtheta * dt; % update the threshold
            % BCM rule: the further output is from threshold, the larger the weight change
            dW = eta * (y_all(t+1) - theta_all(t+1)) * y_all(t+1) * x_all(:, t);  
        case 'Hebbian'
            % Hebbian rule
            dW = eta * y_all(t+1) * x_all(:, t);  % Hebbian rule, 2x1 * 1x2 = 2x2
        otherwise
            error('Unknown rule: %s', rule);
    end

    % Update the weights
    W_all(:, t+1) = W_all(:, t) + dt * dW;

    % Apply a lower bound to W
    W_all(W_all < 0) = 0;
end

%% 3.2  Plot the results
figure('Position', [0 0 1e3 1e3]);
% plot the weight evolution
subplot(3, 1, 1); hold on;grid on;
plot(t_all, squeeze(W_all(1, :)), 'b', 'LineWidth', 1.5);
plot(t_all, squeeze(W_all(2, :)), 'r', 'LineWidth', 1.5);
xlabel('Time (ms)');
ylabel('Weights');
title('Weight Evolution Over Time');
legend({'W_{1}', 'W_{2}'}, 'Location', 'Best');

% plot the threshold evolution
subplot(3, 1, 2); hold on;grid on;
plot(t_all, theta_all, 'k', 'LineWidth', 1.5);
xlabel('Time (ms)');
ylabel('Theta-Threshold (Hz)');
title('Threshold Evolution Over Time');

% plot the output evolution
subplot(3, 1, 3); hold on;grid on;
plot(t_all, y_all, 'k', 'LineWidth', 0.5);
xlabel('Time (ms)');
ylabel('y-Output fitting rate (Hz)');
title('Neuron Outputs Over Time');

% Add a super title
sgtitle(sprintf('Learning Rule: %s', rule), 'FontSize', 16, 'FontWeight', 'bold');

% Observations:
% 1. The weights evolve over time and gradually stabilize.
% 2. The threshold of two neurons diverged, evolved while fluctuating, and seems to enter a stable state.
% 3. The trend of the output follows the shape of the threshold


%% 3.3
% eta: the learning rate, in Hz; lower values (faster learning) lead to faster weight changes

% y0: is an initial guess of average of y output

% tau: bigger the tau, slower the threshold changes