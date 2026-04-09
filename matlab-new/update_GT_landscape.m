function GT = update_GT_landscape(GT, rg_target)
% Extends the GT library if the target Rg is not sufficiently covered.
inst=GT.instrument_par;
sim=GT.simulation_par;
ens=GT.ensemble_par;

% Define the V-span for the landscape
V_power = 0.5;
V_vec = (linspace(0.005^V_power, 0.35^V_power, 15)).^(1/V_power);

% Apply a small, unique jitter for this specific Rg slice
% This ensures no two slices have perfectly aligned V-coordinates
jitter_amplitude = mean(diff(V_vec)) * 0.2; 
V_vec = V_vec + (rand(size(V_vec)) - 0.5) * jitter_amplitude;

% Ensure we stay within physical bounds (V > 0)
V_vec = max(1e-5, V_vec);

fprintf('--> Extending GT Landscape for Rg_true approx %.2f...\n', rg_target);

new_slice = struct('V', [], 'RgMeas', [], 'Y100', [], 'Y210', [], 'Ym210', []);

for i = 1:numel(V_vec)
    % Simulate at this specific Rg and V
    [qx, qy, I_sim, distrp] = Scatter2D(rg_target, 0, V_vec(i), ens.nu, ...
        inst.DETpix, inst.SD_dist, inst.lambda, inst.det_side, 1, ...
        ens.d_nam, ens.dist_param, ens.Scatter_R_g_weight);

    % Measure Rg and moments
    qvr = hypot(qx, qy);
    rg_m = sqrt(-3 * best_origin_quad_b_faster_bins(qvr.^2, log(I_sim)));
    c_rg = (rg_target^2) / rg_m;

    % Second pass: Get moments from bias-corrected simulation
    [qx, qy, I_sim, distrp] = Scatter2D(c_rg, 0, V_vec(i), ens.nu, ...
        inst.DETpix, inst.SD_dist, inst.lambda, inst.det_side, 1, ...
        ens.d_nam, ens.dist_param, ens.Scatter_R_g_weight);

    rg_m = sqrt(-3 * best_origin_quad_b_faster_bins(qvr.^2, log(I_sim)));

    [~, ~, ~, res] = MG_extract(sim.Pxn, qx, qy, I_sim, sim.signum, [], sim.use_r3, sim.use_g3);

    % Store
    new_slice.V(i)      = V_vec(i);
    new_slice.RgMeas(i) = rg_m;
    new_slice.Y100(i)   = res.p(2) / (res.p(1)^2);
    new_slice.Y210(i)   = res.p(3) / (res.p(2)*res.p(1));
    new_slice.Ym210(i)  = res.p(sim.use_g3+5) / (res.p(sim.use_g3+4)*res.p(1));
end

% Append to the master landscape table
GT.V_list      = [GT.V_list, new_slice.V];
GT.RgMeas_list = [GT.RgMeas_list, new_slice.RgMeas];
GT.Yg100_list   = [GT.Yg100_list, new_slice.Y100];
GT.Yg210_list   = [GT.Yg210_list, new_slice.Y210];
GT.Ym210_list  = [GT.Ym210_list, new_slice.Ym210];
GT.RgTrue_covered = [GT.RgTrue_covered, rg_target];
end