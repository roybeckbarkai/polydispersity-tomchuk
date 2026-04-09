function [best_b, best_b_CI, best_xmax, best_y_at_xmax_CI, best_coef, best_coef_CI, info] = ...
         best_origin_quad_b_faster_bins(x, y, varargin)

%% 1. Parse Inputs
opts.nBins = 200;
opts.B = 100; 
opts.alpha = 0.05;
opts.min_pts = 10;
opts.tol_pct = 5;
opts.abs_tol = 1e-3;
opts.stop_on_tol = true;
opts.threshold = -1/6;

if ~isempty(varargin)
    for ii = 1:2:numel(varargin)
        opts.(varargin{ii}) = varargin{ii+1};
    end
end

% Standardize and Sort
x = double(x(:)); y = double(y(:));
mask = x >= 0 & ~isnan(x) & ~isnan(y) & ~isinf(x) & ~isinf(y);
x = x(mask); y = y(mask);
[x, ord] = sort(x); y = y(ord);

%% 2. Pre-calculate Cumulative Sums (The Speed Secret)
% We need components for X'X and X'y where X = [1, x, x^2]
s1  = cumsum(ones(size(x)));
sx  = cumsum(x);
sx2 = cumsum(x.^2);
sx3 = cumsum(x.^3);
sx4 = cumsum(x.^4);
sy  = cumsum(y);
sxy = cumsum(x.*y);
sx2y= cumsum(x.^2 .* y);
sy2 = cumsum(y.^2); % For fast residual calculation

% Determine bin indices
nBins = opts.nBins;
p_edges = linspace(0, 1, nBins + 1);
edges = quantile_fast(x, p_edges);
[~, bin_idx] = histc(edges(2:end), x); 
bin_idx(bin_idx == 0) = numel(x); % Handle edge case

%% 3. Analytical Search Loop (Ultra Fast)
nCuts = numel(bin_idx);
z_val = 1.96; % Approx for 95% CI without stats toolbox
best_k = 1;
stopped_early = false;

% Pre-allocate info storage
all_b = NaN(nCuts, 1);
all_y_low = NaN(nCuts, 1);

for k = 1:nCuts
    idx = bin_idx(k);
    if idx < opts.min_pts, continue; end
    
    % Construct X'X and X'y from cumsums in O(1)
    XtX = [s1(idx),  sx(idx),  sx2(idx);
           sx(idx),  sx2(idx), sx3(idx);
           sx2(idx), sx3(idx), sx4(idx)];
    
    XtY = [sy(idx); sxy(idx); sx2y(idx)];
    
    % Solve OLS
    beta = XtX \ XtY;
    
    % Analytical Error Estimation
    % MSE = (SumY^2 - beta'*X'Y) / (n - p)
    mse = (sy2(idx) - beta' * XtY) / (idx - 3);
    if mse < 0, mse = 0; end % Numerical safety
    
    % Covariance matrix
    invXtX = XtX \ eye(3);
    se = sqrt(diag(invXtX) * mse);
    
    % Predicted Y at x_max and its CI
    xk = x(idx);
    xvec = [1; xk; xk^2];
    y_val = xvec' * beta;
    y_se = sqrt((xvec' * invXtX * xvec) * mse);
    y_low = y_val - z_val * y_se;
    
    % Reliability check for b (slope)
    b_est = beta(2);
    b_se = se(2);
    rel_err = (z_val * b_se / max(abs(b_est), opts.abs_tol)) * 100;
    
    all_b(k) = b_est;
    all_y_low(k) = y_low;
    
    % Logic: Stop if we hit the curvature threshold or stability
    if opts.stop_on_tol && (y_low < opts.threshold) && (rel_err <= opts.tol_pct)
        best_k = k;
        stopped_early = true;
        break;
    end
    best_k = k;
end

%% 4. Final High-Fidelity Bootstrap (Run only ONCE)
% Now we only do the expensive resampling on the selected "best" window
final_idx = bin_idx(best_k);
xn = x(1:final_idx); yn = y(1:final_idx);
X = [ones(final_idx,1), xn, xn.^2];
XtX = X'*X;
beta = XtX \ (X'*yn);
M = XtX \ X';
resid = yn - X*beta;

% Vectorized Bootstrap
B = opts.B;
R = randi(final_idx, final_idx, B);
beta_b = beta + M * (resid(R) - mean(resid));

% Final CIs
lower_p = 100 * (opts.alpha/2);
upper_p = 100 * (1 - opts.alpha/2);

best_xmax = x(final_idx);
best_coef = beta;
best_coef_CI = [percentile_fast(beta_b(1,:), lower_p, upper_p);
                percentile_fast(beta_b(2,:), lower_p, upper_p);
                percentile_fast(beta_b(3,:), lower_p, upper_p)];
            
y_at_b = [1, best_xmax, best_xmax^2] * beta_b;
best_y_at_xmax_CI = percentile_fast(y_at_b, lower_p, upper_p);
best_b = beta(2);
best_b_CI = best_coef_CI(2,:);

% Info struct
info.stopped_early = stopped_early;
info.best_idx = final_idx;
end

%% Fast Helper Functions
function q = percentile_fast(v, p_low, p_high)
    v = sort(v); n = numel(v);
    idx = [p_low, p_high] / 100 * (n-1) + 1;
    i1 = floor(idx); i2 = min(i1+1, n); f = idx - i1;
    q = v(i1).*(1-f) + v(i2).*f;
end

function q = quantile_fast(v, p)
    v = sort(v); n = numel(v);
    r = 1 + (n-1).*p;
    i1 = floor(r); i2 = min(i1+1, n); f = r - i1;
    q = v(i1).*(1-f) + v(i2).*f;
end