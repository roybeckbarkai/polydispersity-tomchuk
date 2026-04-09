function plot_tenor_violin_closest(Results_Table,instrument)
% PLOT_TENOR_VIOLIN_CLOSEST - Violin plot for the heuristic's "Best" solution
% with validity annotations.
%
% Input: Results_Table (struct or table with Noise, True_V, Primary_V)

if isstruct(Results_Table)
    data_tab = struct2table(Results_Table);
else
    data_tab = Results_Table;
end
%     dq=4*pi/lambda*det_side/SD_dist/(2*round(DETpix/2)+1)
if isstruct('instrument')
    dq=4*pi/instrument.lambda*instrument.det_side/instrument.SD_dist/(2*round(instrument.DETpix/2)+1);
else
    dq=0.0024;
end

    % Helper: LaTeX Noise Formatter
    latex_noise = @(x) regexprep(sprintf('$%0.1e$', x), 'e[+]{0,1}(-?)0*(\d+)', '\\times 10^{$1$2}');

% Filter out noiseless reference for the statistical plot
data_tab = data_tab(data_tab.Noise ~= 0, :);
noislist = unique(data_tab.Noise, 'stable');
num_n = numel(noislist);

% Setup Figure
clf %figure('Color', 'w', 'Name', 'TENOR Heuristic Reliability');
hold on; grid on;
colors = lines(num_n);
jitter_width = 0.2;
robust_pct = 95;

all_means = nan(1, num_n);
all_upper = nan(1, num_n);
all_lower = nan(1, num_n);
tick_labels = cell(1, num_n);

for i = 1:num_n
    curr_noise = noislist(i);
    sub = data_tab(data_tab.Noise == curr_noise, :);

    % --- 1. Calculate Validity Percentage ---
    % Valid = the algorithm reached a numerical solution (Primary_V is not NaN)
    total_attempts = height(sub);
    valid_idx = ~isnan(sub.Primary_V);
    num_valid = sum(valid_idx);
    p_valid = (num_valid / total_attempts) * 100;

    % Calculate discrepancy: Delta(sqrt(V))
    deltas = sqrt(max(0, sub.Primary_V(valid_idx))) - sqrt(sub.True_V(valid_idx));

    if isempty(deltas)
        % If no valid solutions, annotate 0% and skip plotting
        text(i, 0, '0%', 'HorizontalAlignment', 'center', 'Color', 'r', 'FontWeight', 'bold');
        continue;
    end

    % --- 2. Robust Statistics (95th Percentile) ---
    s_data = sort(deltas);
    N = numel(s_data);
    p_off = (100 - robust_pct) / 2 / 100;
    idx_low = max(1, round(p_off * N));
    idx_high = min(N, round((1 - p_off) * N));
    subset = s_data(idx_low:idx_high);

    avg = mean(subset);
    r_std = std(subset);
    all_means(i) = avg;
    all_upper(i) = avg + r_std;
    all_lower(i) = avg - r_std;

    % --- 3. Draw Violin Patch ---
    [counts, centers] = histcounts(deltas, 30);
    centers = centers(1:end-1) + diff(centers)/2;
    counts = (counts / max(counts)) * 0.4; % Scale width

    patch([i - counts, fliplr(i + counts)], [centers, fliplr(centers)], colors(i,:), ...
        'FaceAlpha', 0.25, 'EdgeColor', colors(i,:), 'HandleVisibility', 'off');

    % --- 4. Jittered Raw Points ---
    x_jitter = i + (rand(size(deltas)) - 0.5) * jitter_width;
    plot(x_jitter, deltas, '.', 'Color', [0.5 0.5 0.5], 'MarkerSize', 6, 'HandleVisibility', 'off');

    % --- 5. Mean and Error Bars ---
    line([i-0.2, i+0.2], [avg, avg], 'Color', 'r', 'LineWidth', 2, 'HandleVisibility', 'off');
    line([i, i], [avg-r_std, avg+r_std], 'Color', [0.2 0.2 0.2], 'LineWidth', 1.5, 'HandleVisibility', 'off');

    % --- 6. VALIDITY ANNOTATION ---
    % Plotted at the top of the violin/axis
    y_pos = 0.18; % Adjust based on your Y-limits
    text(i, y_pos, sprintf('%d%%\n valid', round(p_valid)), ...
        'HorizontalAlignment', 'center', ...
        'VerticalAlignment', 'bottom', ...
        'FontSize', 9, ...
        'FontWeight', 'bold' ...
        );
    %'BackgroundColor', 'w', ...
    %'EdgeColor', colors(i,:)
    if curr_noise>=0,
        tick_labels{i} = sprintf('%0.1e', curr_noise);
    else
        tick_labels{i} = latex_noise(-curr_noise/dq.^2);

    end
end
% Global Trend Lines
if num_n > 1
    plot(1:num_n, all_means, 'r-', 'LineWidth', 1.2, 'DisplayName', 'Mean discrepancy');
    plot(1:num_n, all_upper, 'k--', 'LineWidth', 1, 'DisplayName', sprintf('Robust (%d%%) \\pm 1 STD',robust_pct));
    plot(1:num_n, all_lower, 'k--', 'LineWidth', 1, 'HandleVisibility', 'off');
end

yline(0, 'k-', 'Alpha', 0.4, 'HandleVisibility', 'off');

% Formatting
set(gca, 'TickLabelInterpreter', 'latex', 'XTick', 1:num_n, 'XTickLabel', tick_labels);
xlabel('Photon density (photon/$\mathrm{nm}^{-2}$)', 'Interpreter', 'latex');
ylabel('$\Delta(V^{1/2})$ discrepancy', 'Interpreter', 'latex');
title('Convergence and Validity of Closest Solutions', 'Interpreter', 'latex');

%     ylim([-0.2, 0.22]); % Added space at the top for annotations
set(gca, 'Layer', 'top'); % Keep grid behind patches
legend('Location', 'northeast');
end