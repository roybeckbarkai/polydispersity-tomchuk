function [q_mat_x,q_mat_y,I_mat,distrp]=Scatter2D(rg,Nois,V,Nu,DETpix,SD_dist,lambda,det_side,PSF0,dist_type, dist_param, Weight_power)
%% simulate a specific scattering case with noise added
% rg Nu V are the scattering parameters (Nu=phi'')
%Nois is the noise level (compared to max signal=1).
% NEGATIVE Nois value provides the number of photons per pixel at the
% center. Then we assume the noise level is the square root of the photon number in each pixel.
%Detpix shows the number of pixels in the detector side, default is 1000
% SD_dist: Distance sample to detector in cm (default 150 cm),
% detector half side is det_side/2 (default 3.5cm),
% wavelength lambda [nm] (default=0.15 nm)
% PSF0 option to incorporate initial measurement PSF (before noise). a matrix which is
% convolved with the simulated scattering. default is ones(1)
% Outside the guinier region the model is invalid, so intensity becomes NaN
%   dist_type  {'normal','lognormal','schulz','exponential','boltzmann',
%               'triangular','uniform'} - calls size_distribution_discrete
%               for the r_g distribution type (mean=rg, rel variance=V)
%  dist_param.N= number of r_g in the distribution
% default distribution- gaussian (N=11)
%
% Weight_power adds a weight to the scattering of each particle: in powers of r 
% default is 0- every r has the same weight (mass): eg IDP
% for solid spheres Weight_power=6: the intensity is proportional to m^2=r^6
% for spherical shells the power is 4: mass=r^2
%
% SPECIAL CASES- if Nu=-1/63 exactly, we use Pedersen 1997 exact formula
% (and not the second order in (q*r)^2 )
%  Nu=0.000666 - thin disks
%  Nu=11/225 - thin rods
%  Nu=-1/45 - spherical shell
%  Nu=1/18 - gaussian chain
%
%
% returns a meshgrid of q on the detector, x- and y-values in q_mat_x and
% q_mat_y (nm-1), respectively, and intensity matrix I_mat
% 
% [q_mat_x,q_mat_y,I_mat]=Scatter2D(rg,Nois,V,Nu,DETpix,SD_dist,lambda,det_side,PSF0)
% usage examples:
%spherical particles (Nu=-1/63) with mean(r_g)= 4 nm; relative variance V, 
% diamond typical PSF and
%noise:
% [q_mat_x,q_mat_y,I_mat]=Scatter2D(4,-1e3,V,-1/63,1000,360,0.1,7,bartlett2d(3,15)); 
% 
%Beck laboratory typical sizes, zero noise, gaussian chain model (Nu=+1/18)  
% [q_mat_x,q_mat_y,I_mat]=Scatter2D(4,0,V,+1/18,1000,150,0.15,7,bartlett2d(25,25))
%
% note that usually the guinier region is small, and most pixels are
% useless (nan). multiply DETpix and det_side by a factor (0.5) to get the
% same results with less effort.

noisemodel=1; %default for constant noise level
try PSF0=PSF0/sum(PSF0(:));
catch PSF0=1;
end
PSF0=makeKernelOddCentered(PSF0); %make it odd so that the center remains at the same place
try rg=abs(rg);
catch
    rg=5;
end
try noisemodel=sign(Nois);
catch
    Nois=0;
    noisemodel=1;
end
try Nu=Nu;
catch
    Nu=-1/63; %spherical default
end
try V=abs(V);
catch
    V=0.1;
end
try r_g=rg;
catch
    r_g=5;
end
try isfinite(Weight_power);
catch
    Weight_power=0;
end

%pixel size = d/L*4pi/lambda=75um/150cm*4pi/lambda= 4e-3 nm^-1
% dqpix=4e-3;

try SD_dist=SD_dist;
catch
    SD_dist=150;  % Sample-detector dist (cm)
end
try lambda=lambda;
catch
    lambda=0.15; %wavelength nm
end
try det_hside=det_side/2;
catch
    det_hside=3.5; %half side (cm)
end

% Distance to detector 150, detector half side is det_hside, lambda=0.15 nm
maxq=4*pi/lambda*det_hside/SD_dist;
try Npix=round(DETpix/2);
catch
    Npix=501;
end
qv=linspace(-maxq,maxq,2*Npix+1);
[qvx,qvy]=meshgrid(qv);
qvr2=(qvx.^2+qvy.^2);
qvr4=qvr2.*qvr2;
qvr=sqrt(qvr2);
nu=[Nu];
ff=@(qr) exp(-qr.^2/3+0.5*nu*qr.^4); %monolithic formfactor (function of unitless q*r_g)
try n_radii=dist_param.N;
catch
    n_radii=11; % number of radii in guinier distribution
end
try [r_vect,p] = size_distribution_discrete(n_radii, V, dist_type, [],0,4);
catch
% [r_vect, p] = gaussian_discrete(n_radii, V); r_vect=r_vect+1;
[r_vect,p] = size_distribution_discrete(n_radii, V, [],'normal',0,4);
end
distrp.r=r_vect*rg;
distrp.p=p;
F=zeros(size(qvr2));
for ii=1:n_radii
    if 0,     F=F+p(ii)*(ff(qvr*r_g*(r_vect(ii)))); %scattering of a simple ensemble
    else
        s = rg * (r_vect(ii));     % scalar scale
        s2 = s * s;                    % s^2
        s4 = s2 * s2;                  % s^4

        % compute exponent using precomputed qvr2 / qvr4
        % exponent = - (s^2/3)*qvr2 + 0.5*nu*(s^4)*qvr4;
        % compute as two scaled arrays added together
        exponent = -(s2/3) .* qvr2;       % MxM
        exponent = exponent + (0.5 * nu * s4) .* qvr4;
        if abs((1/nu-round(1/nu))-0.000666)<1e-6  %add another order
            %example: nu=1/(18+0.000666+0.0123e-8) -> phi'''=0.0123
            ordersix=1e8*(abs(1/nu-round(1/nu))-0.000666); %phi'''
            exponent = exponent + (ordersix/6 * s4 * s2) .* qvr4 .*qvr2;
        end
        if abs(nu-0.000666)<1e-6  %add another order for nu=0 
            %example: nu=(0.000666+0.0123e-8) -> phi'''=0.0123
            ordersix=1e8*(abs(nu-0.000666)); %phi'''
            exponent = exponent + (ordersix/6 * s4 * s2) .* qvr4 .*qvr2;
        end

        exponent(abs(-(s2/3) .* qvr2)< 1 * abs((0.5 * nu * s4) .* qvr4))=nan; % eliminate anomalies outside the guinier region

        if Nu==-1/63 %exact spherical (Pedersen1997)
            GF=sqrt(5/3); %factor from sphere's radius to r_g
            expo=log((3*(sin(qvr*s*GF)-GF*s*qvr.*cos(s*qvr*GF))./qvr./qvr2/s/s2/GF^3).^2);
            expo(abs(qvr*s*GF)<=eps)=0;
            exponent=real(expo).*(~imag(expo));
            Weight_power=6;
        end
        if Nu==+1/18 %exact Debye Gaussian chain (Debye 1949)
            GF=1; %factor from sphere's radius to r_g

            expo=log(2*(exp(-(qvr*s*GF).^2)+(qvr*s*GF).^2-1)./(qvr*s*GF).^4);
            expo(abs(qvr*s*GF)<=eps)=0;
            exponent=real(expo).*(~imag(expo));
            Weight_power=0;
        end

        if Nu==-1/45 %exact shell
            GF=1; %factor from sphere's radius to r_g

            expo=2*log((sin(qvr*s*GF))./(qvr*s*GF));
            expo(abs(qvr*s*GF)<=eps)=0;
            exponent=real(expo).*(~imag(expo));
            Weight_power=4;
        end

        if Nu==11/225 %exact thin rod phi''=11/25, phi'''=1236/99225
            GF=sqrt(12); %factor from rod's length to r_g
            u=qvr*s*GF+10*eps;
            %             expo=log(2*sinint(u)./u-4*sin(u/2).^2./u.^2);
            % High speed version (for x > 0)
            Si = imag(expint(1i * u)) + pi/2;
            expo=log(2*Si./u-4*sin(u/2).^2./u.^2);
            expo(abs(u)<=eps)=0;
            exponent=real(expo).*(~imag(expo));
            Weight_power=2;
        end

        if Nu==11/225+0.000666 % approx thin rod phi''=11/25, phi'''=412/33075
            s = rg * (r_vect(ii));     % scalar scale
            s2 = s * s;                    % s^2
            s4 = s2 * s2;                  % s^4

            nu=11/225;

            % compute exponent using precomputed qvr2 / qvr4
            % exponent = - (s^2/3)*qvr2 + 0.5*nu*(s^4)*qvr4;
            % compute as two scaled arrays added together
            exponent = -(s2/3) .* qvr2;       % MxM
            exponent = exponent + (0.5 * nu * s4) .* qvr4;
            ordersix=412/33075; %phi'''
            exponent = exponent + (ordersix/6 * s4 * s2) .* qvr4 .*qvr2;
            exponent(abs(-(s2/3) .* qvr2)< 1 * abs((0.5 * nu * s4) .* qvr4))=nan; % eliminate anomalies outside the guinier region
        end

        if Nu==0.000666 %exact thin disk  phi''=0, phi'''=1/270
            GF=sqrt(2); %factor from disk's radius to r_g
            u=qvr*s*GF;
            expo=log(2./u.^2.*(1-besselj(1,2*u)./u));
            expo(abs(u)<=eps)=0;
            exponent=real(expo).*(~imag(expo));
            Weight_power=4;
        end
        
        
%         F = F + p(ii) * exp(exponent);
        F = F + r_vect(ii).^Weight_power.*p(ii) * exp(exponent);


    end
end

% Delt=sqrt(V);
% F=ff(qvr*r_g*(1+Delt))+ff(qvr*r_g*(1-Delt)); %scattering of a simple ensemble
% F=F/2; % max F is 1
F=filter2(PSF0,F,"same");
%         F=F+Nois*rand(size(F));  % add noise
if noisemodel<0
    F=F*abs(Nois); % Nois is the number of photons at the peak
    F=F+sqrt(F).*randn(size(F));
    F=F/abs(Nois);


else
    %             F=F+Nois*(1-2*rand(size(F)));  % add not just positive noise
    F=F+Nois*randn(size(F));  % add gaussian noise
end
q_mat_x=qvx;
q_mat_y=qvy;
F(F<0)=0; %remove scarce negatives
I_mat=F;
end

function kernel_out = makeKernelOddCentered(kernel_in)
% makeKernelOddCentered  Expand even-sized kernel to odd while keeping center.
%   kernel_out = makeKernelOddCentered(kernel_in)
%   If rows are even, expands to rows+1 by splitting each original row's
%   weights half/half between adjacent rows. Same for columns.
%
% Examples:
%   makeKernelOddCentered([1 1])        -> [0.5 1 0.5]
%   makeKernelOddCentered([1;1])       -> [0.5; 1; 0.5]
%   makeKernelOddCentered([1 2;3 4])   -> 3x3 result (center preserved)

kernel_out = kernel_in;

% If rows even: create rows+1 and distribute each row half/half
[r, c] = size(kernel_out);
if mod(r,2) == 0
    newr = r + 1;
    tmp = zeros(newr, c);
    % for each original row i, split into tmp(i,:) and tmp(i+1,:)
    for i = 1:r
        tmp(i, :)   = tmp(i, :)   + 0.5 * kernel_out(i, :);
        tmp(i+1, :) = tmp(i+1, :) + 0.5 * kernel_out(i, :);
    end
    kernel_out = tmp;
end

% If cols even: create cols+1 and distribute each column half/half
[r, c] = size(kernel_out);
if mod(c,2) == 0
    newc = c + 1;
    tmp = zeros(r, newc);
    % for each original col j, split into tmp(:,j) and tmp(:,j+1)
    for j = 1:c
        tmp(:, j)   = tmp(:, j)   + 0.5 * kernel_out(:, j);
        tmp(:, j+1) = tmp(:, j+1) + 0.5 * kernel_out(:, j);
    end
    kernel_out = tmp;
end
end

function [x, p] = gaussian_discrete(N, V, alpha)
%GAUSSIAN_DISCRETE  Discrete zero-mean Gaussian with length N and variance V.
%   [x,p] = gaussian_discrete(N, V, alpha)
%     N      : number of points (integer >= 2)
%     V      : target variance (>0)
%     alpha  : (optional) shape parameter for probabilities; default 0.5
%              p_i ∝ exp(-alpha * u_i^2), with symmetric grid u
%
%   Returns:
%     x : Nx1 locations (symmetric about 0), s.t. sum((x-μ)^2 .* p) = V exactly
%     p : Nx1 probabilities (sum(p)=1), mean zero by symmetry
%
%   Works for small N (2..11) and large N. No root finding.

if nargin < 3 || isempty(alpha), alpha = 0.5; end
if ~(isscalar(N) && N==round(N) && N>=2), error('N must be integer >= 2'); end
if ~(isscalar(V) && V>=0), error('V must be positive'); end

if V==0
    x=zeros(N,1);
    p=ones(N,1)/N;
    return
end
% --- symmetric base grid u (unitless) ---
if mod(N,2)==1
    m = (N-1)/2;
    u = (-m:m).';                % includes 0
else
    m = N/2;
    v = ((0:m-1) + 0.5).';
    u = [-flipud(v); v];         % ±0.5, ±1.5, ...
end

% --- probabilities from Gaussian shape on u (independent of scale) ---
w = exp(-alpha * (u.^2));        % unnormalized
% enforce exact symmetry of p for numerical robustness
if mod(N,2)==1
    mid = (N+1)/2;
    for k = 1:mid-1
        wk = 0.5*(w(mid-k) + w(mid+k));
        w(mid-k) = wk; w(mid+k) = wk;
    end
else
    for k = 1:N/2
        wk = 0.5*(w(k) + w(N+1-k));
        w(k) = wk; w(N+1-k) = wk;
    end
end
p = w / sum(w);

% --- choose scale a so that variance equals V exactly ---
% With x = a*u and mean zero by symmetry:
% Var = E[x^2] = a^2 * E[u^2]_p  =>  a = sqrt(V / E[u^2]_p).
Eu2 = sum((u.^2) .* p);
a = sqrt(V / Eu2);

x = a * u;
end

function K = bartlett2d(n, m, mode)
%BARTLETT2D  Normalized 2D triangular (Bartlett-like) kernel with nonzero edges.
%   K = bartlett2d(n, m)          % raised-edge triangular (edges > 0)
%   K = bartlett2d(n, m, 'zero')  % classic Bartlett (edges = 0)
%
%   No toolboxes required.

    if nargin < 3, mode = 'raised'; end

    if nargin == 1, m=n; end

    wy = tri1d(n, mode);   % column
    wx = tri1d(m, mode).'; % row

    K = wy * wx;           % separable outer product
    s = sum(K(:));
    if s > 0, K = K / s; end
end

function w = tri1d(N, mode)
% 1D triangular window; 'raised' keeps nonzero endpoints.

    if N <= 0
        w = zeros(0,1); return
    elseif N == 1
        w = 1; return
    end

    idx = (0:N-1).';
    c   = (N-1)/2;                 % geometric center

    switch lower(mode)
        case 'zero'   % classic Bartlett (endpoints = 0 for N>=3)
            if N == 2
                % With only two taps, use flat weights (normalized later)
                w = [1; 1];
            else
                w = 1 - abs(idx - c) / c;
            end
        otherwise     % 'raised' (nonzero endpoints)
            d = (N+1)/2;           % larger denominator -> nonzero edges
            w = 1 - abs(idx - c) / d;
            w(w < 0) = 0;          % numerical safety for very small N
    end
end
