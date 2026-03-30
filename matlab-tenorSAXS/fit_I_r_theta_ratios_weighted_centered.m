function R = fit_I_r_theta_ratios_weighted_centered(r, th, I, weight, use_r3, use_g3)
% Weighted fit with per-ratio precision grades based on CI95 width
% Model:
%   I(r,th) = G(r) + M(r)*cos(2*th)
% G(r) = g0 + g1*r + g2*r^2 (optional + g3*r^3 if use_g3)
% M(r) = m0*r + m1*r^2 (optional + m2*r^3 if use_r3)
%
%   use_r3: include r^3 .* c2 in M-block (default true)
%   use_g3: include rc.^3 in G-block (default false)

% This version:
%   * centers r in the G(r) block to reduce collinearity (safe — model class unchanged)
%   * uses a QR-based weighted least squares (stable; avoids XtWX normal equations)
%   * maps centered coefficients back to the original parameterization for outputs

% defaults
if nargin < 5 || isempty(use_r3), use_r3 = true; end
if nargin < 6 || isempty(use_g3), use_g3 = false; end


% -------- setup / guards -------------------------------------------
warning('error', 'MATLAB:singularMatrix');        % keep as hard errors
warning('error', 'MATLAB:nearlySingularMatrix');

r   = r(:);    th = th(:);    I = I(:);    weight = weight(:);
c2  = cos(2*th);
valid = isfinite(I) &  isfinite(r) & isfinite(th) & isfinite(weight);
if ~any(valid), 
    error('No valid samples.'); 
end

r = r(valid);  I = I(valid);  c2 = c2(valid);  w = weight(valid);

% -------- weighted centering for G(r) only -------------------------
% IMPORTANT: We center r in the *G-block* (1, r, r^2) only.
% Centering the c2* block would introduce a pure c2 column (i.e., change model class).
w_sum = sum(w);
mu_r  = sum(w .* r) / w_sum;      % weighted mean (more appropriate than unweighted here)
rc    = r - mu_r;                 % centered r for G

% % -------- build design (mixed basis: centered G, original M) -------
% % G-part (centered): [1, rc, rc.^2]
% % M-part (original): [r.*c2, r.^2.*c2, r.^3.*c2]
% if use_r3
%     % 6-column model
%     X = [ ones(size(r)), rc, rc.^2, r.*c2, r.^2.*c2, r.^3.*c2 ];
% else
%     % 5-column model
%     X = [ ones(size(r)), rc, rc.^2, r.*c2, r.^2.*c2 ];  %only second order in M
% end
% Build design matrix: G-block first, M-block after
if use_g3
    Gcols = [ ones(size(r)), rc, rc.^2, rc.^3 ];   % 4 cols
else
    Gcols = [ ones(size(r)), rc, rc.^2 ];          % 3 cols
end

if use_r3
    Mcols = [ r.*c2, r.^2.*c2, r.^3.*c2 ];         % 3 cols
else
    Mcols = [ r.*c2, r.^2.*c2 ];                   % 2 cols
end

X = [Gcols, Mcols];
y = I;

% -------- stable weighted least squares via QR ---------------------
% Solve argmin || sqrt(w).*(X*p_c - y) ||_2
sw = sqrt(w);
Xw = X .* sw;
yw = y .* sw;

% Economy QR; handles near-collinearity without forming XtWX
[Q,Rq] = qr(Xw, 0);
% If R is rank-deficient, use SVD least-squares as a fallback
rankX = rank(Rq);
k = size(X,2);
if rankX < k
    % SVD fallback (stable minimum-norm LS solution)
    [U,S,V] = svd(Xw, 'econ');
    s = diag(S);
    tol = max(size(Xw)) * eps(s(1));
    sInv = diag( 1 ./ s .* (s > tol) );
    p_c = V * sInv * (U' * yw);
else
    p_c = Rq \ (Q' * yw);
end

% -------- residuals / variance & covariance (stable via Rq) ---------
yhat = X * p_c;
res  = y - yhat;
N    = size(X,1);
SSE  = sum(w .* (res.^2));
dof  = max(N - k, 1);
s2   = SSE / dof;

% Covariance of p_c using Rq (no explicit inverse of XtWX)
% cov(p_c) = s2 * inv(Rq)' * inv(Rq)
Rinv  = Rq \ eye(k);              % solves Rq*Rinv = I
covPc = s2 * (Rinv * Rinv.');

% % -------- map back to original parameterization --------------------
% % Our p_c is for [1, rc, rc^2, r*c2, r^2*c2, r^3*c2].
% % We need p (original) for [1, r, r^2, r*c2, r^2*c2, r^3*c2].
% % For G block (first 3), with r = rc + mu_r:
% %   g2 = g2c
% %   g1 = g1c - 2*mu_r*g2c
% %   g0 = g0c - mu_r*g1 - mu_r^2*g2
% % M block (last 3) is already in original r-powers.
% T = eye(6);
% T(1,1) = 1;       T(1,2) = -mu_r;     T(1,3) =  mu_r^2;
% T(2,1) = 0;       T(2,2) =  1;        T(2,3) = -2*mu_r;
% T(3,1) = 0;       T(3,2) =  0;        T(3,3) =  1;
% % lower-right 3x3 is identity
% 
% T=T(1:rankX,1:rankX);  %cut T in case we take only 2 M coefficients


% Build transform T mapping centered G-block back to original r-powers
% We'll map G block [1, rc, rc^2, (rc^3)] to [1, r, r^2, (r^3)].
% For rc^3 present: r = rc + mu: r^3 = rc^3 + 3*mu*rc^2 + 3*mu^2*rc + mu^3
% Build Tfull for maximum size 7 (4 G + 3 M). We'll then truncate.
Tfull = eye(7);
% G block mapping (rows = target [1 r r^2 r^3], cols = [1 rc rc^2 rc^3])
% Fill first 4x4:
Tfull(1,1) = 1;    Tfull(1,2) = -mu_r;      Tfull(1,3) =  mu_r^2;       Tfull(1,4) = -mu_r^3;
Tfull(2,1) = 0;    Tfull(2,2) = 1;           Tfull(2,3) = -2*mu_r;      Tfull(2,4) = 3*mu_r^2;
Tfull(3,1) = 0;    Tfull(3,2) = 0;           Tfull(3,3) = 1;            Tfull(3,4) = -3*mu_r;
Tfull(4,1) = 0;    Tfull(4,2) = 0;           Tfull(4,3) = 0;            Tfull(4,4) = 1;
% M block columns are already in original r powers; they map identity in their positions.
% Determine how many columns total:
kG = size(Gcols,2);
kM = size(Mcols,2);
kTot = kG + kM;

% Build final T of size kTot x kTot by taking top kTot rows and cols from Tfull,
% but place the identity for M-block at the right position.
T = zeros(kTot);
% Insert G mapping: target G rows are 1:kG map from centered G cols 1:kG
T(1:kG,1:kG) = Tfull(1:kG,1:kG);
% Insert identity for M-block
T(kG+1:end, kG+1:end) = eye(kM);


p = T * p_c;

% Covariance transform: covP = T * covPc * T'
covP = T * covPc * T.';

% -------- ratios & SEs (same as before, but using covP) -----------
g0=p(1); g1=p(2); g2=p(3);
m0=p(4); m1=p(5); 
try m2=p(6);
catch
    m2=nan;
end
if kG >= 4, g3 = p(4); else g3 = NaN; end

g_ratio = g1 / g0;
m_ratio = m1 / m0;

dg = zeros(1,rankX); dg(1) = -g1/g0^2; dg(2) = 1/g0;
dm = zeros(1,rankX); dm(4) = -m1/m0^2; dm(5) = 1/m0;

% Guards against tiny negatives from roundoff
se_g = sqrt(max(dg * covP * dg.', 0));
se_m = sqrt(max(dm * covP * dm.', 0));

z95  = 1.95996398454005;
g_CI = g_ratio + z95*se_g*[-1 1];
m_CI = m_ratio + z95*se_m*[-1 1];

g_hw = z95*se_g;    m_hw = z95*se_m;
g_w  = 2*g_hw;      m_w  = 2*m_hw;

grade_g = 1 ./ (1 + g_w);
grade_m = 1 ./ (1 + m_w);

% -------- weighted R^2 ---------------------------------------------
ybar_w = sum(w .* y) / sum(w);
SSTw   = sum(w .* (y - ybar_w).^2);
R2w    = 1 - SSE / SSTw;

% -------- pack outputs (same schema) -------------------------------
R.p        = p.';                    % [g0 g1 g2 m0 m1 m2] in ORIGINAL basis
R.g_ratio  = g_ratio;   R.m_ratio  = m_ratio;
R.g_CI95   = g_CI;      R.m_CI95   = m_CI;
R.g_SE     = se_g;      R.m_SE     = se_m;
R.g_CI95W  = g_w;       R.m_CI95W  = m_w;
R.g_CI95HW = g_hw;      R.m_CI95HW = m_hw;
R.grade_g  = grade_g;   R.grade_m  = grade_m;
R.R2w      = R2w;
R.Rq= Rq;
R.covP=covP;

R.ratioG   = R.g_ratio;  R.ratioM  = R.m_ratio;
R.CI95_G   = R.g_CI95;   R.CI95_M  = R.m_CI95;

% -------- optional: 95% CI for g2/g1 (unchanged) -------------------
g1 = p(2); g2 = p(3);
if abs(g1) < eps
    g21_ratio = NaN; g21_SE = NaN; g21_CI = [NaN NaN];
    g21_hw = NaN; g21_w = NaN;
else
    g21_ratio = g2 / g1;
    d21 = zeros(1,rankX); d21(2) = -g2 / (g1^2); d21(3) = 1 / g1;
    q = d21 * covP * d21.'; q = max(q, 0);
    g21_SE = sqrt(q);
    z95 = 1.95996398454005;
    g21_CI = g21_ratio + z95 * g21_SE * [-1 1];
    g21_hw = z95 * g21_SE; g21_w = 2 * g21_hw;
end

R.g21_ratio  = g21_ratio;
R.g21_SE     = g21_SE;
R.g21_CI95   = g21_CI;
R.g21_CI95HW = g21_hw;
R.g21_CI95W  = g21_w;

% -------- optional: new ratios g100 = g1/g0^2 and g210 = g2/(g1*g0) -----------
g0 = p(1); g1 = p(2); g2 = p(3);

% g100 = g1 / g0^2
if abs(g0) < eps
    g100_ratio = NaN; g100_SE = NaN; g100_CI = [NaN NaN];
    g100_hw = NaN; g100_w = NaN;
else
    g100_ratio = g1 / g0^2;
    d100 = zeros(1,rankX); 
    d100(1) = -2*g1 / (g0^3);  % derivative w.r.t g0
    d100(2) = 1 / (g0^2);      % derivative w.r.t g1
    q = d100 * covP * d100.'; q = max(q,0);
    g100_SE = sqrt(q);
    g100_hw = z95 * g100_SE; g100_w = 2 * g100_hw;
    g100_CI = g100_ratio + z95 * g100_SE * [-1 1];
end

% g210 = g2 / (g1 * g0)
if abs(g0*g1) < eps
    g210_ratio = NaN; g210_SE = NaN; g210_CI = [NaN NaN];
    g210_hw = NaN; g210_w = NaN;
else
    g210_ratio = g2 / (g1 * g0);
    d210 = zeros(1,rankX);
    d210(1) = -g2 / (g1 * g0^2);  % derivative w.r.t g0
    d210(2) = -g2 / (g0 * g1^2);  % derivative w.r.t g1
    d210(3) = 1 / (g0 * g1);      % derivative w.r.t g2
    q = d210 * covP * d210.'; q = max(q,0);
    g210_SE = sqrt(q);
    g210_hw = z95 * g210_SE; g210_w = 2 * g210_hw;
    g210_CI = g210_ratio + z95 * g210_SE * [-1 1];
end

% m210 = m2 / (m1 * g0)
if abs(g0*m1) < eps
    m210_ratio = NaN; m210_SE = NaN; m210_CI = [NaN NaN];
    m210_hw = NaN; m210_w = NaN;
else
    m210_ratio = m1 / (m0 * g0);
    d210 = zeros(1,rankX);
    d210(1) = -m1 / (m0 * g0^2);  % derivative w.r.t g0
    d210(2) = -m1 / (g0 * m0^2);  % derivative w.r.t g1
    d210(3) = 1 / (g0 * m0);      % derivative w.r.t g2
    q = d210 * covP * d210.'; q = max(q,0);
    m210_SE = sqrt(q);
    m210_hw = z95 * m210_SE; m210_w = 2 * m210_hw;
    m210_CI = m210_ratio + z95 * m210_SE * [-1 1];
end

% -------- pack into output struct --------------------------------------
R.g100_ratio    = g100_ratio;
R.g100_SE       = g100_SE;
R.g100_CI95     = g100_CI;
R.g100_CI95HW   = g100_hw;
R.g100_CI95W    = g100_w;
R.g3 = g3;

R.g210_ratio    = g210_ratio;
R.g210_SE       = g210_SE;
R.g210_CI95     = g210_CI;
R.g210_CI95HW   = g210_hw;
R.g210_CI95W    = g210_w;

R.m210_ratio    = m210_ratio;
R.m210_SE       = m210_SE;
R.m210_CI95     = m210_CI;
R.m210_CI95HW   = m210_hw;
R.m210_CI95W    = m210_w;


% -------- metadata (could be useful for debugging) -----------------
R.mu_r   = mu_r;          % weighted centering used
R.rankX  = rankX;
R.SSE    = SSE; R.SSTw = SSTw; R.s2 = s2;

end
