function plot_GT_landscape_3d(GT)
% PLOT_GT_LANDSCAPE_3D Visualizes the look-up surface for V extraction.
%
% Input: GT struct containing V_list, RgMeas_list, Y100_list

    % 1. Extract scattered data from the library
    % X: Relative Variance (V)
    x_raw = GT.V_list;
    % Y: Measured Radius of Gyration (Rg_meas - biased by V)
    y_raw = GT.RgMeas_list;
    % Z: Observable moment-ratio (Yg100)
    z_raw = GT.Yg100_list;

    if numel(x_raw) < 10
        warning('Not enough points in GT library to create a meaningful surface plot.');
        return;
    end

    %% 2. Create a regular grid for visualization
    % Define the resolution of the plot grid
    plot_res = 60; 
    
    % Generate vectors for the grid boundaries based on data limits
    gv = linspace(min(x_raw), max(x_raw), plot_res);
    gr = linspace(min(y_raw), max(y_raw), plot_res);
    
    % Create the 2D mesh matrices
%     [V_grid, R_grid] = meshgrid(gv, gr);
    [V_grid, R_grid] = meshgrid(x_raw, y_raw);
    [V_grid, R_grid] = meshgrid(sort(union(x_raw,gv)), sort(union(y_raw,gr)));

    %% 3. Grid the scattered data onto the regular mesh
    % Uses core Matlab linear interpolation to find Z values for the grid.
    % points outside the convex hull of simulated data will be NaN.
    Y100_surf = scat_interp_unscaled(x_raw, y_raw, z_raw, V_grid, R_grid, 'linear');

% F = scatteredInterpolant(x_raw', y_raw', z_raw', 'linear', 'linear');
% Y100_surf = F(V_grid, R_grid);
    %% 4. Plotting
%     figure('Color', 'w', 'Name', 'TENOR GT Landscape Visualization');
    clf
    % --- Main Surface Plot ---
    % 'FaceAlpha' makes it slightly transparent to see the grid lines
    % 'EdgeColor','none' cleans up the look.
    surf(V_grid, R_grid, Y100_surf, 'FaceAlpha', 0.8, 'EdgeColor', 'none');
    hold on;
    
    % --- Add Raw Data Points ---
    % Superimpose the actual simulated points as black dots to show coverage.
    plot3(x_raw, y_raw, z_raw, 'k.', 'MarkerSize', 8, 'HandleVisibility', 'off');
    
    %% 5. Formatting and Annotation
    colormap parula; % Standard color map (blue to yellow)
    h_cb = colorbar;
    ylabel(h_cb, '$Y_{g100}$ Intensity Ratio', 'Interpreter', 'latex');
    
    grid on;
    view(3); % Set to standard 3D perspective
    
    % LaTeX formatting for axes
    xlabel('Relative Variance ($V$)', 'Interpreter', 'latex');
    ylabel('Measured $R_g$ (nm)', 'Interpreter', 'latex');
    zlabel('$Y_{g100}$ moment-ratio', 'Interpreter', 'latex');
    
    title('TENOR Extraction Landscape: $\mathcal{Y}_{G100}(V, R_{g,meas})$', ...
        'Interpreter', 'latex', 'FontSize', 12);
        
    % Optional: Add lighting for better depth perception
    camlight headlight;
    lighting gouraud;
end

