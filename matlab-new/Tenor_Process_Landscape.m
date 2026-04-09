function [V_est, Sols, Winner, GT, rg_in, Yg100, Yg210, Ym210] = Tenor_Process_Landscape(I_mat0, qx, qy, GT, inst, sim, ens)
% 1. Measure Input Parameters
qvr = hypot(qx, qy);
rg_in = sqrt(-3 * best_origin_quad_b_faster_bins(qvr.^2, log(I_mat0)));
[~, ~, ~, res_in] = MG_extract(sim.Pxn, qx, qy, I_mat0, sim.signum, [], sim.use_r3, sim.use_g3);

Yg100 = res_in.p(2) / (res_in.p(1)^2);
Yg210 = res_in.p(3) / (res_in.p(2)*res_in.p(1));
Ym210 = res_in.p(sim.use_g3+5) / (res_in.p(sim.use_g3+4) * res_in.p(1));

% 2. Dynamic Bracketing Check
% Griddata requires at least two Rg slices to create a 2D convex hull.
needs_update = false;

if isempty(GT.RgTrue_covered)
    % INITIALIZATION: Library is empty. Simulate a bracket around rg_in.
    GT.instrument_par = inst;
    GT.simulation_par = sim;
    GT.ensemble_par = ens;

    fprintf('--> Initializing GT Library with bracket [%.2f, %.2f]\n', rg_in*0.9, rg_in*1.1);
    GT = update_GT_landscape(GT, rg_in * 0.9);
    GT = update_GT_landscape(GT, rg_in * 1.1);
    needs_update = true; % Now we have a span, but let's also add the exact point
else
    rg_min_lib = min(GT.RgMeas_list);
    rg_max_lib = max(GT.RgMeas_list);

    % Check if we need to expand the hull to "enclose" rg_in
    if rg_in <= rg_min_lib
        fprintf('--> Extending Hull BELOW: Adding slice at %.2f\n', rg_in * 0.9);
        GT = update_GT_landscape(GT, rg_in * 0.9);
        needs_update = true;
    elseif rg_in >= rg_max_lib
        fprintf('--> Extending Hull ABOVE: Adding slice at %.2f\n', rg_in * 1.1);
        GT = update_GT_landscape(GT, rg_in * 1.1);
        needs_update = true;
    end

    % Check for internal gaps (>5% relative distance)
    is_in_gap = min(abs(GT.RgTrue_covered - rg_in)/rg_in) > 0.05;
    if is_in_gap
        needs_update = true;
    end
end

% 3. Trigger precise slice simulation if needed
if needs_update
    GT = update_GT_landscape(GT, rg_in);
end

% 4. Multi-Solution Grid Search (1D Slice Method)
% We create a high-resolution 1D curve at the exact measured rg_in
V_fine = linspace(min(GT.V_list), max(GT.V_list), 300);

% Perform 2D interpolation to get the moments for THIS specific Rg
%     Yg100_at_rg = griddata(GT.V_list, GT.RgMeas_list, GT.Yg100_list, V_fine, repmat(rg_in, size(V_fine)), 'linear');
    Yg100_at_rg = scat_interp_unscaled(GT.V_list, GT.RgMeas_list, GT.Yg100_list, V_fine, repmat(rg_in, size(V_fine)), 'linear');
% F = scatteredInterpolant(GT.V_list', GT.RgMeas_list', GT.Yg100_list', 'natural', 'linear');
% Yg100_at_rg = F(V_fine, repmat(rg_in, size(V_fine)));
% Yg210_at_rg = griddata(GT.V_list, GT.RgMeas_list, GT.Yg210_list, V_fine, repmat(rg_in, size(V_fine)), 'linear');
Yg210_at_rg = scat_interp_unscaled(GT.V_list, GT.RgMeas_list, GT.Yg210_list, V_fine, repmat(rg_in, size(V_fine)), 'linear');
% F = scatteredInterpolant(GT.V_list', GT.RgMeas_list', GT.Yg210_list', 'natural', 'linear');
% Yg210_at_rg = F(V_fine, repmat(rg_in, size(V_fine)));
% Find intersections (where Y_sim crosses Y_measured)
idx = find(diff(sign(Yg100_at_rg - Yg100)) ~= 0);
v_sols = [];
for k = 1:numel(idx)
    ii = idx(k);
    % Precise root finding via linear interpolation between the two indices
    v_sols(k) = V_fine(ii) + (Yg100 - Yg100_at_rg(ii)) * (V_fine(ii+1)-V_fine(ii)) / (Yg100_at_rg(ii+1)-Yg100_at_rg(ii));
end

% 5. Consensus Heuristic
if isempty(v_sols)
    % Robust fallback: find the nearest point in the landscape
    %     V_est = griddata(GT.V_list, GT.RgMeas_list, GT.V_list, rg_in, Yg100, 'nearest');
    F = scatteredInterpolant(GT.Yg100_list', GT.RgMeas_list', GT.V_list', 'natural', 'linear');
    V_est=F(Yg100,rg_in);
    Winner = 'Nearest_Fallback';
elseif numel(v_sols) == 1
    V_est = v_sols(1);
    Winner = 'Unique_Solution';
else
    % Use Yg210 to solve the non-monotonic ambiguity
    y210_at_sols = interp1(V_fine, Yg210_at_rg, v_sols, 'linear');
    [~, best_idx] = min(abs(y210_at_sols - Yg210));
    V_est = v_sols(best_idx);
    Winner = 'Consensus_Selection';
end
if V_est < -0.05
    V_est = nan;
end
Sols.Primary_V = V_est;
Sols.Alternatives = setdiff(v_sols, V_est);
Sols.Alternatives = Sols.Alternatives(Sols.Alternatives > -0.05);
Sols.Alternatives = Sols.Alternatives(isfinite(Sols.Alternatives));
end
