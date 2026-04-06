function [x, p] = size_distribution_discrete_no_max(N, Vrel, dist_type, threshold, xmin, weight_power)
% Revised for exact moment matching using Center-of-Mass binning
if nargin < 3 || isempty(dist_type), dist_type = 'normal'; end
if nargin < 4 || isempty(threshold), threshold = 0.999; end
if nargin < 5 || isempty(xmin), xmin = 0; end
if nargin < 6 || isempty(weight_power), weight_power = 0; end

Mean = 1; 
sigma = sqrt(Vrel) * Mean; 

% 1. Adaptive xmax to capture the 'mass' (Tomchuk Eq 13 context)
temp_xmax = Mean + 20 * sigma; 
u_fine = linspace(xmin, temp_xmax, 10000).'; % High-res grid for integration

% 2. PDF Calculation (Tomchuk et al. 2023 definitions)
switch lower(dist_type)
    case 'normal'
        w_fine = exp(-(u_fine - Mean).^2 / (2 * sigma^2)); % 
    case 'lognormal'
        s = sqrt(log(1 + Vrel)); % 
        m = log(Mean) - 0.5 * s^2;
        w_fine = (1 ./ u_fine) .* exp(-(log(u_fine) - m).^2 / (2 * s^2)); % 
    case 'schulz'
        z = 1/Vrel - 1; % 
        w_fine = u_fine.^z .* exp(-(z + 1) * u_fine / Mean); % 
    case 'boltzmann'
        w_fine = exp(-sqrt(2) * abs(u_fine - Mean) / sigma); % 
    case 'triangular'
        L = sigma * sqrt(6); % 
        w_fine = max(0, 1 - abs(u_fine - Mean) / L); % 
    case 'uniform'
        L = sigma * sqrt(3); % 
        w_fine = (u_fine >= Mean - L & u_fine <= Mean + L); % 
end

w_fine(isnan(w_fine) | isinf(w_fine)) = 0;
w_fine = w_fine / sum(w_fine);

% 3. Calculate Cumulative Distributions
% Use 'weighted' CDF if weight_power > 0 (Importance Sampling)
w_weighted = w_fine .* (u_fine .^ weight_power);
C_weight = cumsum(w_weighted) / sum(w_weighted);
C_number = cumsum(w_fine) / sum(w_fine);

% Ensure strict monotonicity for interp1
C_weight = C_weight + linspace(0, 10e-12, length(C_weight)).';

% 4. Exact Partitioning
edges_prob = linspace(0, threshold, N + 1);
r_edges = interp1(C_weight, u_fine, edges_prob, 'linear', 'extrap');

x = zeros(N, 1);
p = zeros(N, 1);

% Cumulative product for center-of-mass calculation: Integral of (r * PDF)
C_moment1 = cumsum(w_fine .* u_fine);

for i = 1:N
    % Find indices in fine grid falling within this bin
    idx = u_fine >= r_edges(i) & u_fine < r_edges(i+1);
    
    % Probability of this bin (Exact Number Fraction)
    p_high = interp1(u_fine, C_number, r_edges(i+1), 'linear', 'extrap');
    p_low  = interp1(u_fine, C_number, r_edges(i),   'linear', 'extrap');
    p(i) = p_high - p_low;
    
    % Center of Mass x(i): Integral(r*P(r) dr) / Integral(P(r) dr)
    m1_high = interp1(u_fine, C_moment1, r_edges(i+1), 'linear', 'extrap');
    m1_low  = interp1(u_fine, C_moment1, r_edges(i),   'linear', 'extrap');
    x(i) = (m1_high - m1_low) / p(i);
end

% 5. THE AFFINE FIX: Force exactly Mean=1 and Var=Vrel
curr_mu  = sum(p .* x);
curr_var = sum(p .* (x - curr_mu).^2);

% Scale and shift radii
scale_factor = sqrt(Vrel / curr_var);
x = scale_factor * (x - curr_mu) + Mean;

% Ensure physical validity (no negative radii)
x = max(x, xmin);
% Final cleanup: ensure sum(p) = 1 
p = p / sum(p);
% fprintf('%s mean=%f, dV=%f, max(r)=%f\n',dist_type, sum(p.*x),sum(p.*x.^2)./sum(p.*x).^2-1-Vrel,max(x))
% fprintf('')
% Optional: Plotting for verification
% figure (432); 
% subplot(211); plot(u_fine, w_fine/sum(w_fine),DisplayName=dist_type); hold on; plot(x, p, '+-','DisplayName',dist_type);
% ylabel('Probability'); legend (Location="bestoutside")
% subplot (212); plot(u_fine, C_weight, '-','DisplayName',dist_type,'LineWidth',2);legend (Location="bestoutside")
% ylabel('Weighted Cumulative'); hold on
% plot(x, interp1(u_fine, C_weight,x), '+','HandleVisibility','off')

end