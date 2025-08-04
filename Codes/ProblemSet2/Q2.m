
%  w_self: the connection between either neuron population and itself
%  w_other:  the connection between the different neuron populations
%  tau1: time constant for the first neuron population
%  tau2: time constant for the second neuron population
%  r1: firing rate of the first neuron population
%  r2: firing rate of the second neuron population
%  t: time vector

clear all
clc
close all

%% (e)
I1=63; % input current to the first neuron population. in Hz
I2=57; % input current to the second neuron population. in Hz
I = [I1, I2]';

state = 'stable'; 
% state = 'winner-take-all';
% state = 'winner-take-too-much';

% Add adaptation and noise to the dynamics to allow state switch
sigmaNoise = 10; % add gaussian noise to the dynamics (for answering g(ii))
adapt_strength = 10; % add adaptation to the dynamics (for answering g(ii))
tau_adapt = 500; % adaptation time constant in ms

switch state
    case 'stable'
        fprintf('State: Stable\n');
        % Both neurons reach stable states
        %   when both eigenvalues of W are < 1 (does not have to be positive!)
        w_self = .2; % self-connection weight
        w_other = -.7; % connection weight between the two populations
    case 'winner-take-all'
        fprintf('State: Winner-take-all\n');
        %   when w_other is very negative (hence the mutual inhibition is strong)
        %  and when both eigenvalues of W are still < 1, so that both neurons reach stable states
        w_self = .2; % self-connection weight
        w_other = -10; % for Q2g: if w_other is a large negative value
    case 'winner-take-too-much'
        fprintf('State: Winner-take-too-much\n');
        % unstable, runaway growth
        %   when lambda1 or lambda_common> 1, ie. w_self + w_other > 1, both neurons will grow exponentially
        %   when lambda2 or lambda_diff > 1, ie. w_self - w_other > 1, one neuron will grow exponentially
        w_self = .1; % self-connection weight
        w_other = -1.2; % for Q2g: if w_other is a large negative value
    otherwise
        error('Unknown state: %s', state);
end

W = [w_self, w_other; w_other, w_self]; % connectivity matrix
tau_bio = 18; % time constant for both populations, in ms
dt = 1; % time step in ms
T = 500; % total time in ms
t_all = 0:dt:T; % time vector in ms
r_max = 100; % maximum firing rate in Hz

% Define eigenvectors of connectivity matrix W
eigenvector1 = [1,1];
eigenvector2 = [1,-1];

% Calculate the eigenvalues of connectivity matrix W
lambda1 = w_self + w_other;
lambda2 = w_self -w_other;

% Calculate the amplification factors of two eigenvectors
A1 = 1/(1-lambda1);
A2 = 1/(1-lambda2);

% Calculate the effective time constants
tau_eff1 = tau_bio * A1;
tau_eff2 = tau_bio * A2;

% Print w_other and w_self
fprintf('\n\nw_self = %.2f, w_other = %.2f\n', w_self, w_other);

% print eigenvalues and eigenvectors
fprintf('\n\nEigenvalues of W: lambda1 = %.2f, lambda2 = %.2f\n', lambda1, lambda2);
fprintf('Eigenvectors of W: [%d, %d], [%d, %d]\n', eigenvector1(1), eigenvector1(2), eigenvector2(1), eigenvector2(2));

%% (1)
% Decompose I1 and I2 into the eigenvectors of W
b1 = (I1+I2)/2;
b2 = (I1-I2)/2;
fprintf('\n\n(e-1) Effective time constants: tau_eff1 = %.2f ms, tau_eff2 = %.2f ms\n', tau_eff1, tau_eff2);

%% (2) Calculate the resting firing rates of the two populations
r_inf = b1/(1-lambda1)*eigenvector1' + b2/(1-lambda2)*eigenvector2';
fprintf('\n\n(e-2) Resting firing rates: r1 = %.2f Hz, r2 = %.2f Hz\n', r_inf(1), r_inf(2));

%% f: simulate the network by numerically solving the equations
% Set initial firing rates
nRand = 3; % Number of random initial conditions
r_init_all = [round(rand(nRand, 2)*100); [0,0]; I']; % Initial conditions including zero and random values
r_init_all = [I']; % Initial conditions including zero and random values
% r_init_all = [10, 5; 10,10; 10,15; 10,20; 10,30; 10,50]

nInitial = size(r_init_all, 1);

figure('Position', [100, 100, 1200, 500]);

subplot(1,2,1), hold on;
% plot r_max
yline(r_max, 'k--', 'LineWidth', 1, 'DisplayName', 'r_{max}', 'HandleVisibility', 'off');
xlabel('Time (ms)');
ylabel('Firing Rate (Hz)');
xlim([-5 T]);
ylim([0 100]);
title(sprintf('Network dynamics from different initial conditions\nSolid: r1, Dashed: r2'));

subplot(1,2,2), hold on;
% plot eigenvectors and their directions
plot([0, 100], [0, 100], 'k--', 'LineWidth', 1, 'DisplayName', sprintf('Eigenvector 1 [%d, %d]', eigenvector1));
plot([0, 100], [100, 0], 'k--', 'LineWidth', 1, 'DisplayName', sprintf('Eigenvector 2 [%d, %d]', eigenvector2));

% Plot the steady state
plot(r_inf(1), r_inf(2), 'k*', 'LineWidth', 1, 'DisplayName', 'Steady State');
xline(r_inf(1), '-', "color", ones(1,3)*.5, 'LineWidth', .5, 'HandleVisibility', 'off');
yline(r_inf(2), '-', "color", ones(1,3)*.5, 'LineWidth', .5, 'HandleVisibility', 'off');
xlabel('Population 1 Firing Rate (Hz)');
ylabel('Population 2 Firing Rate (Hz)');
xlim([-5 100]);
ylim([-5 100]);
title('Phase Plane of Firing Rates');

% plot r_max
xline(r_max, 'k--', 'LineWidth', 1, 'DisplayName', 'r_{max}', 'HandleVisibility', 'off');
yline(r_max, 'k--', 'LineWidth', 1, 'DisplayName', 'r_{max}', 'HandleVisibility', 'off');

% Put w_other and w_self in the title
sgtitle(sprintf('Network Dynamics and Phase Plane Analysis\nw_{self} = %.2f, w_{other} = %.2f', w_self, w_other), 'FontSize', 16, 'FontWeight', 'bold');

for iInitial = 1:nInitial

    % preallocate arrays for firing rates and adaptation
    r_all = nan(2, length(t_all));
    A_all = zeros(2, length(t_all)); % adaptation current

    % Set initial firing rate
    r0 = r_init_all(iInitial, :)';
    r_all(:, 1) = r0;

    % Simulate
    for t = 2:length(t_all)
        % Update adaption
        % dAdt = (-A_all(:, t-1) + adapt_strength * r_all(:, t-1)) / tau_adapt;
        dAdt = (adapt_strength * r_all(:, t-1).^2) / tau_adapt;
        A_all(:, t) = A_all(:, t-1) + dt * dAdt;

        % Calculate derivative
        drdt = (-r_all(:, t-1) + W * r_all(:, t-1) + I - A_all(:, t-1)) / tau_bio;
        % drdt = (-r_all(:, t-1) + W * r_all(:, t-1) + I) / tau_bio;
        drdt = drdt + randn(2, 1) * sigmaNoise; % Add some stochastic noise to the dynamics

        % Update firing rates
        r_all(:, t) = r_all(:, t-1) + dt * drdt;
        % Ensure firing rates do not go negative (optional constraint)
        r_all(:, t) = max(r_all(:, t), 0);
        % Ensure firing rates do not exceed r_max (optional constraint)
        r_all(:, t) = min(r_all(:, t), r_max);

    end

    c = rand(1, 3); % Random color for each initial condition

    % Plot population 1 and 2 firing rates
    subplot(1,2,1), hold on
    axis square
    plot(t_all, r_all(1,:), '-', 'Color', c, 'LineWidth', 1.5, 'DisplayName', sprintf('r1 (r_0 = %d)', r0(1)));
    plot(t_all, r_all(2,:), '--', 'Color', c, 'LineWidth', 1.5, 'DisplayName', sprintf('r2 (r_0 = %d)', r0(2)));

    % plot starting point
    plot(0, r0(1), 'o', 'MarkerFaceColor', c, 'MarkerEdgeColor', 'k', 'MarkerSize', 6, 'HandleVisibility', 'off', 'DisplayName', 'Starting Point');
    plot(0, r0(2), 'o', 'MarkerFaceColor', c, 'MarkerEdgeColor', c, 'MarkerSize', 6, 'HandleVisibility', 'off', 'DisplayName', 'Starting Point');

    % plot analytical steady state
    yline(r_inf(1), 'k--', 'LineWidth', 1, 'HandleVisibility', 'off', 'DisplayName', sprintf('r_{inf}1 (r0 = %d)', r0(1)));
    yline(r_inf(2), 'k--', 'LineWidth', 1, 'HandleVisibility', 'off', 'DisplayName', sprintf('r_{inf}2 (r0 = %d)', r0(2)));

    % Plot r1 against r2
    subplot(1,2,2), hold on
    plot(r_all(1, :), r_all(2, :), 'color', c, 'LineWidth', 1.5, 'DisplayName', sprintf('r_0 = [%d,%d])', r0));

    % plot starting point
    plot(r0(1), r0(2), 'o', 'MarkerFaceColor', c, 'MarkerEdgeColor', 'k', 'MarkerSize', 6, 'HandleVisibility', 'off', 'DisplayName', 'Starting Point');

    axis square

    legend('Location', 'best', 'Interpreter', 'none');

    % Set font size for all axes in the figure
    set(findall(gcf, 'Type', 'axes'), 'FontSize', 15);

    % Pause for visualization
    pause(1);
end

%% g: Winner-take-all networks
% (i)
% The neurons with a weaker input will be driven to zero firing rate.
% Runaway growth of the winner's firing rate can occur if then different mode of the eigenvalue of the weight is > 1

% (ii)
% Add stochastic noise and adaptation to the dynamics
dominant = r_all(1,:) > r_all(2,:);
switches = sum(abs(diff(dominant)) > 0);
fprintf("g(ii): Number of switches: %d\n", switches);

