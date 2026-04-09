function vq = scat_interp_unscaled(x, y, v, xq, yq, method)
% SCAT_INTERP_UNSCALED Mimics griddata but scales axes to [0, 1] internally.
%
% Usage:
%   vq = scat_interp_unscaled(x, y, v, xq, yq)
%   vq = scat_interp_unscaled(x, y, v, xq, yq, method)
%
% Inputs:
%   x, y   : Vectors of sample point coordinates (e.g., Rg and V)
%   v      : Vector of sample values at those coordinates (e.g., Yg100)
%   xq, yq : Coordinates of the query points
%   method : 'linear' (default), 'nearest', or 'natural'
%
% Note: 'natural' is recommended for SAXS landscapes to avoid "troughs".

    if nargin < 6 || isempty(method)
        method = 'linear';
    end

    % 1. Ensure inputs are column vectors for scatteredInterpolant
    x = x(:); y = y(:); v = v(:);

    % 2. Calculate Scaling Parameters from Sample Points
    xmin = min(x); xmax = max(x);
    ymin = min(y); ymax = max(y);

    % Prevent division by zero if all points have the same coordinate
    dx = xmax - xmin; if dx == 0, dx = 1; end
    dy = ymax - ymin; if dy == 0, dy = 1; end

    % 3. Normalize Sample Points to [0, 1]
    x_norm = (x - xmin) / dx;
    y_norm = (y - ymin) / dy;

    % 4. Create the Interpolant Object
    % We use 'linear' as the extrapolation fallback
    F = scatteredInterpolant(x_norm, y_norm, v, method, 'linear');

    % 5. Normalize Query Points using the SAME parameters
    xq_norm = (xq - xmin) / dx;
    yq_norm = (yq - ymin) / dy;

    % 6. Query the Interpolant
    vq = F(xq_norm, yq_norm);
end