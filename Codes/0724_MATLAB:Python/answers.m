%% Part 1: Matrix operations for a feedforward network
N=50; % number of nodes in the input layer (x)
M=10; % number of nodes in the output layer (y)

% Create a weight matrix W with random values
W = rand(M, N); % testing
%
% Define the input vector x
x = randn(N, 1); % input vector with N random values

% Calculate the output vector y
y = W * x; % matrix multiplication to get the output vector

disp(y)

%% Part 2: Logical operations, for-loops, and plotting (random walk

% Define a function
function [V, nSpikes] = GenerateVoltage(p, T, Vreset, Vthresh, V0)
% Inputs
% P: the probability of going up, p
% T: the number of time steps to simulate
% Vreset: the reset voltage, Vreset
% Vthres: the spike threshold
% V0: the initial voltage

% Output: a vector V, the voltage at all time steps from 1 to T.

V = zeros(T, 1); % Initialize the voltage vector
V(1) = V0; % Set the initial voltage
nSpikes = 0;
for t = 1:T-1
    % Check if voltage exceeds the threshold
    if V(t) >= Vthresh
        V(t+1) = Vreset; % Reset voltage if threshold is exceeded
        nSpikes = nSpikes + 1; % Count the spike
    else
        % Compute the voltage at the next time step
        if rand < p % Randomly decide to go up or down
            V(t+1) = V(t) + 1; % Increase voltage
        else
            V(t+1) = V(t) - 1; % Decrease voltage
        end
    end
end
end

% Simulate a random walk
V0 = -65; % in unit of mV
Vthres = -45;
Vreset = -70;

p_all = [.6, .62, .64, .68, .7]; % Different probabilities
T = 1000; % Number of time steps, 1ms per step

% try different probabilities
nSpikes_all = []; % Initialize nSpikesuency array to store nSpikesuencies for legend
figure('Position', [0,0,2e3 2e3]); hold on, grid on
iplot = 1;
for p = p_all
    [V, nSpikes] = GenerateVoltage(p, T, Vreset, Vthres, V0);
    subplot(length(p_all), 1, iplot)
    
    % Plot the voltage over time
    plot(1:T, V, '-');
    xlabel('Time (ms)');
    ylabel('Voltage (mV)');
    
    yline(Vthres, 'r--', 'Threshold'); % threshold line
    yline(Vreset, 'b--', 'Reset'); % reset line

    ylim([-80, -30]);
    iplot = iplot+1;
    title(sprintf('Random Walk with p=%.2f, nSpikes=%d', p, nSpikes));
end

%% Part 3: Convolution to estimate voltage response to a spike train
T = 3; % in seconds
dt = 0.001; % time step in seconds (1 ms)
N = T/dt; % number of time steps
firing_rate = 20; % frequency of the spike train in Hz
p = firing_rate * dt; % probability of a spike in each time step

% Generate a spike train
spiketrain = rand(1,N)>(1-p);

fprintf('When p=%.2f, the number of spikes is %d\n, freq = %.1f Hz', p, sum(spiketrain), sum(spiketrain)/T);

% Construct a kernel for the voltage response
mu = 5;     % mean of the exponential kernel, in ms
k = exppdf(-50:50, mu); % Exponential kernel for the voltage response

figure
plot(-50:50, k, 'k', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Kernel Value');
title('Exponential Kernel for Voltage Response');

% convert spiketrain to a double from a logical
spiketrain = double(spiketrain);
% Convolve the spike train with the kernel
voltage_response = conv(spiketrain, k, 'same');

% Plot the voltage response
figure("Position", [0, 0, 1000, 500]);
plot1=subplot(2,1,1); hold on, grid on
plot(1:N, voltage_response, 'b', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Voltage Response');
title('Voltage Response to Spike Train');

plot2=subplot(2,1,2); hold on, grid on
plot(1:N, spiketrain, 'r', 'LineWidth', 2);
xlabel('Time (ms)');
ylabel('Spike Train');
title('Spike Train');   

% Link x-axes for zoom/pan alignment
linkaxes([plot1, plot2], 'x');

% Note: The really high spike is due to multiple closely spikes firing together, which are overlapping in the bottom figure 

%% Part 4: Convolution to detect edges in images
octopus = imread('octopus_1.png'); % Load the image
octopus = rgb2gray(octopus); % Convert to grayscale if it's a color image

% Create the filter kernel for edge detection
k = [0 0 0; 0 1.125 0; 0 0 0] - .125*ones(3,3);

imagesc(octopus), colormap gray

% convolve the image with the kernel
octopus_filtered = conv2(double(octopus), k, 'same');

% Display the filtered image
figure("Position", [0, 0, 1000, 500]);
subplot(1,2,1), imagesc(octopus), colormap gray, axis image, title('Original Image');
subplot(1,2,2), imagesc(abs(octopus_filtered)), colormap gray, axis image, title('Filtered Image');
% Note: The filtered image highlights the edges in the original image