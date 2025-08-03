% Created by Shutian Xue on 07/28/2025 
% This script is a pilot model for the project of MCN summer course 2025 at MBL
% Collaborator: xx

% First, let's simuate the average activity the i-th neuron across trials
% A: the neural activity
%   i: the i-th neuron
%   j: the j-th stimulus (determine mu and sigma)
%  k: the k-th trial
nNeurons = 10; % Number of neurons
nStimuli = 5; % Number of stimuli
nTrials = 100; % Number of trials

A = randn(nNeurons, nStimuli, nTrials); % Simulated neural activity

% Compute the correlation for each pair of neurons
correlationMatrix = zeros(nNeurons, nNeurons);
for i = 1:nNeurons
    for j = i+1:nNeurons
        % Compute the correlation across all stimuli and trials
        correlationMatrix(i, j) = corr(reshape(A(i, :, :), [], 1), reshape(A(j, :, :), [], 1));
        correlationMatrix(j, i) = correlationMatrix(i, j); % Symmetric matrix
    end
end 

% plot histogram of the correlation values
figure; hold on;
histogram(correlationMatrix(:), 'Normalization', 'pdf');
title('Histogram of Neuron Correlation Values');
xlabel('Correlation Value');
ylabel('Probability Density');  
% Add a vertical line at zero
xline(0, 'r--', 'Zero Correlation', 'LabelVerticalAlignment', 'middle');