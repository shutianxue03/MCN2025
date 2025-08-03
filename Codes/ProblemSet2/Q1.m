%% 1.3 (ii)
% Plot s head as a function of x
x_all = linspace(-10, 10, 100);
sigma = 1;
s_head_pos = x_all - sqrt(2)*sigma^2; % when s>0
s_head_neg = x_all + sqrt(2)*sigma^2; % when s<0

% plot
figure;
hold on;
% s_head when s>0, blue
plot(x_all, s_head_pos, '--','color' , ones(1,3)*.5, 'LineWidth', 2,'HandleVisibility', 'off');
plot(x_all(s_head_pos>0), s_head_pos(s_head_pos>0), '-','color' , [0,0,1], 'LineWidth', 2);
% s_head when s<0, red
plot(x_all, s_head_neg, '--','color' , ones(1,3)*.5, 'LineWidth', 2, 'handleVisibility', 'off');
plot(x_all(s_head_neg<0), s_head_neg(s_head_neg<0), '-','color' , [1,0,0], 'LineWidth', 2);
% s_head when s=0, black
ind0 = (x_all<=sqrt(2)*sigma^2) & (x_all>=-sqrt(2)*sigma^2);
plot(x_all(ind0), zeros(size(x_all(ind0))), 'k', 'LineWidth', 2);
xlabel('x');
ylabel('s head');
title('s head as a function of x');
legend('s head (s > 0)', 's head (s < 0)', 's head (s = 0)');
% plot x=0 and y=0 as grey solid lines
yline(0, '-', 'color', ones(1,3)*.3, 'LineWidth', 1, 'HandleVisibility', 'off');
xline(0, '-', 'color', ones(1,3)*.3, 'LineWidth', 1, 'HandleVisibility', 'off');
grid on;
hold off;
