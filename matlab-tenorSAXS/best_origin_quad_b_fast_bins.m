function [best_b, best_b_CI, best_xmax, best_y_at_xmax_CI, best_coef, best_coef_CI, info] = ...
         best_origin_quad_b_fast_bins(x, y, varargin)
% BEST_ORIGIN_QUAD_B_FAST_BINS
% Fast toolbox-free quadratic cumulative fit using binned x.
% Works on large datasets by testing cumulative windows at bin edges.
%
% USAGE:
% [best_b, best_b_CI, best_xmax, best_yCI, best_coef, best_coef_CI, info] = ...
%    best_origin_quad_b_fast_bins(x, y, 'nBins',200, 'B',600, 'tol_pct',5);
%
% Key options (name/value):
%   'nBins'     - number of equal-width bins across x range (default 200)
%   'B'         - bootstrap samples (default 600)
%   'alpha'     - significance (default 0.05 -> 95% CI)
%   'min_pts'   - minimum points in a bin window (default 6)
%   'tol_pct'   - relative half-CI tolerance in percent for b (default 5)
%   'abs_tol'   - fallback denom for small b (default 1e-3)
%   'stop_on_tol' - true/false to enable early stop (default true)
%   'threshold' - y threshold (default -1/6)
%
% Outputs:
%   best_b, best_b_CI, best_xmax, best_y_at_xmax_CI, best_coef, best_coef_CI, info
%
% No toolboxes required.

%% parse inputs & checks
if nargin < 2, error('Usage: best_origin_quad_b_fast_bins(x,y)'); end

% defaults
opts.nBins = 200;
opts.B = 20; %15
opts.alpha = 0.05;
opts.min_pts = 6;
opts.tol_pct = .05;
opts.abs_tol = 1e-3;
opts.stop_on_tol = true;
opts.threshold = -1/6;

% parse varargin
if ~isempty(varargin)
    for ii = 1:2:numel(varargin)
        name = varargin{ii}; val = varargin{ii+1};
        if isfield(opts, name)
            opts.(name) = val;
        else
            error('Unknown option %s', name);
        end
    end
end

% columnize
x = double(x(:)); y = double(y(:));
if numel(x) ~= numel(y), error('x and y must be same length'); end

% keep x >= 0 by default (modify if needed)
mask_nonneg = x >= 0;
x = x(mask_nonneg); y = y(mask_nonneg);
if isempty(x), error('No x >= 0 points found.'); end

% sort by x
[x,ord] = sort(x); y = y(ord);

% make bins across x range
xmin = min(x); xmax_all = max(x);
nBins = max(1, round(opts.nBins));
% make quantile-based bins (equal number of samples per bin)
nBins = max(1, round(opts.nBins));
p_edges = linspace(0,1,nBins+1);            % cumulative percentiles
edges = quantile(x, p_edges);               % toolbox-free helper
xcuts = edges(2:end);                        % right edges for cumulative windows
nCuts = numel(xcuts);

% prealloc diagnostics
coef = NaN(3, nCuts);
coef_CI = NaN(3,2,nCuts);
y_at_xcut_CI = NaN(nCuts,2);
b_at_xcut_CI = NaN(nCuts,2);
npts = zeros(nCuts,1);
rel_err_pct = NaN(nCuts,1);

% parameters
B = opts.B;
alpha = opts.alpha;
min_pts = opts.min_pts;
tol_pct = opts.tol_pct;
abs_tol = opts.abs_tol;
stop_on_tol = opts.stop_on_tol;
threshold = opts.threshold;

lower_pct = 100*(alpha/2);
upper_pct = 100*(1-alpha/2);

stopped_early = false;
pick_k = NaN;

%% loop bins (cumulative windows)
for k = 1:nCuts
    xmaxk = xcuts(k);
    sel = x <= xmaxk;
    xn = x(sel); yn = y(sel);
    n = numel(xn);
    npts(k) = n;
    if n < min_pts
        continue;
    end

    % design matrix
    X = [ones(n,1), xn, xn.^2];

    % OLS solve
    XtX = X' * X;
    XtY = X' * yn;
    try
        beta = XtX \ XtY;   % 3x1 [a;b;c]
    catch
        continue
    end
    coef(:,k) = beta;

    % residuals centered
    yhat = X * beta;
    resid = yn - yhat;
    resid = resid - mean(resid);

    % M = inv(X'*X)*X'  (3 x n)
    M = (XtX \ X');

    % vectorized residual bootstrap
    R = randi(n, n, B);
    r_s = reshape(resid(R), n, B);
    beta_b = beta + M * r_s;   % 3 x B

    % predicted y at xmaxk
    xvec = [1; xmaxk; xmaxk^2];
    y_at_b = xvec' * beta_b;   % 1 x B

    % b samples
    b_b = beta_b(2,:);         % 1 x B

    % percentile CIs (toolbox-free)
    y_ci = percentile(y_at_b, [lower_pct, upper_pct]);  % 1x2
    b_ci = percentile(b_b, [lower_pct, upper_pct]);    % 1x2

    y_at_xcut_CI(k,:) = y_ci;
    b_at_xcut_CI(k,:) = b_ci;

    % coef CIs
    for j = 1:3
        coef_CI(j,:,k) = percentile(beta_b(j,:), [lower_pct, upper_pct]);
    end

    % relative error for b (half-width / |b| *100)
    b_est = beta(2);
    half_w = (b_ci(2) - b_ci(1)) / 2;
    denom = max(abs(b_est), abs_tol);
    rel_err = (half_w / denom) * 100;
    rel_err_pct(k) = rel_err;

    % stopping: require predicted-y lower CI > threshold AND rel err <= tol_pct
    if stop_on_tol && (y_ci(2) < threshold) && (rel_err <= tol_pct)
        pick_k = k;
        stopped_early = true;
        break;
    end
end

%% choose final window if not stopped early
valid = ~isnan(coef(1,:));
if ~any(valid)
    error('No valid windows; increase data or reduce min_pts.');
end

% --- ensure pick_k is valid ---
if stopped_early
    % pick_k already set by early-stop
else
    % find all bins with valid coef
    valid_idx = find(~isnan(coef(1,:)));
    
    if isempty(valid_idx)
        error('No valid bins found; increase data or reduce min_pts.');
    end
    
    % prefer bins where predicted y lower CI > threshold
    good_idx = valid_idx(y_at_xcut_CI(valid_idx,1) > threshold);
    
    if ~isempty(good_idx)
        pick_k = max(good_idx);   % largest xmax satisfying condition
    else
        % fallback: choose the valid bin with largest y lower CI
        [~,rel] = max(y_at_xcut_CI(valid_idx,1));
        pick_k = valid_idx(rel);
    end
end

% final safety check
if pick_k < 1 || pick_k > numel(xcuts)
    % extreme fallback: pick first valid bin
    pick_k = find(~isnan(coef(1,:)),1,'first');
end

% finalize outputs
best_xmax = xcuts(pick_k);
best_coef = coef(:, pick_k);
best_coef_CI = squeeze(coef_CI(:,:,pick_k));
best_y_at_xmax_CI = y_at_xcut_CI(pick_k,:);
best_b = best_coef(2);
best_b_CI = best_coef_CI(2,:);

% diagnostics
info.xcuts = xcuts;
info.npts = npts;
info.coef = coef;
info.coef_CI = coef_CI;
info.y_at_xcut_CI = y_at_xcut_CI;
info.b_at_xcut_CI = b_at_xcut_CI;
info.rel_err_pct = rel_err_pct;
info.nBins = nBins;
info.B = B;
info.alpha = alpha;
info.min_pts = min_pts;
info.tol_pct = tol_pct;
info.abs_tol = abs_tol;
info.threshold = threshold;
info.stop_on_tol = stop_on_tol;
info.stopped_early = stopped_early;
info.pick_index = pick_k;

end

%% helper: toolbox-free percentile (returns row vector)
function q = percentile(v, p)
v = sort(v(:)); n = numel(v);
if n == 0
    q = NaN(size(p));
    return
end
p_row = p(:).';            % 1 x m
p_row(p_row < 0) = 0;
p_row(p_row > 100) = 100;
r = (n - 1) * (p_row / 100) + 1;   % 1 x m
i1 = floor(r);
i2 = min(i1 + 1, n);
f = r - i1;
v1 = v(i1)'; v2 = v(i2)';
q = v1 .* (1 - f) + v2 .* f;       % 1 x m
end


function q = quantile(v,p)
% QUANTILE(v,p) returns quantiles for vector v at probabilities p (0..1)
v = sort(v(:));
n = numel(v);
p = p(:).';  % row
r = 1 + (n-1).*p;       % positions
i1 = floor(r);
i2 = min(i1+1, n);
f = r - i1;
v1 = v(i1)'; v2 = v(i2)';
q = v1.*(1-f) + v2.*f;
end
