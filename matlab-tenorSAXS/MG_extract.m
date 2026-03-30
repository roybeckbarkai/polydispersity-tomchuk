function [g_rat,RG2,Pxn,res]=MG_extract(Pxn,q_mat_x,q_mat_y,I_mat,signum,RG2,use_r3,use_g3)
%% Find the M and G coeffs and R_g for a given intensity map
%Pxn is the 4 pixel numbers defining the 2 assymetric PSFs
% [g_rat,RG]=MG_extract(Pxn,q_mat_x,q_mat_y,I_mat)
% input: meshgrid of q on the detector, x- and y-values in q_mat_x and
% signum= default 4; % number of stds in the gaussian filter (PSF)
% q_mat_y (nm-1), respectively, and intensity matrix I_mat
% use_r3 says the fit for M is cubic or quadratic
% use_g3 says the fit for G is cubic or quadratic


% result matrix g_rat shows :
% 1: g1/g0
% 2-3: 95% confidence interval for that
% 4: m2/m1
% 5-6: 95% confidence interval for that
% 7: g2/g1
% 8: estimated Guinier radius- theoretically r_g^2*(1+V). the average of the
% fits for the 2 PSFs
% 9-10: 95% confidence interval for g2/g1
% RG2 is also the estimated square Guinier radius squared- theoretically r_g^2*(1+V). for
% the unfiltered I_mat

% default fit for M is quadratic or cubic
if nargin < 7 || isempty(use_r3)
    use_r3 = true;
end

if nargin < 8 || isempty(use_g3)
    use_g3 = true;
end


qvr=hypot(q_mat_x,q_mat_y);
try 
    RG2=RG2(1);
    if ~isfinite(RG2) || RG2<=0
        RG2=-3*best_origin_quad_b_fast_bins(qvr.^2, log(I_mat)); %r_g^2*(1+V)
    end
catch
    RG2=-3*best_origin_quad_b_fast_bins(qvr.^2, log(I_mat)); %r_g^2*(1+V)
end
if ~exist('signum','var')
    signum=4; % number of stds in the gaussian filter (PSF)
end
try
    Pxn=Pxn(1:4);
    if (max((Pxn-1)/2~=uint8((Pxn-1)/2)) || ... %Pix num should be odd
            (isempty(setdiff(Pxn(1:2),Pxn(3:4)))) || ... % two pixn couples should be different
            ((Pxn(1) == Pxn(2)) && (Pxn(1)*Pxn(2)~=1)) || ...
            ((Pxn(3) == Pxn(4)) && (Pxn(3)*Pxn(4)~=1)) ) % none of the couples should be symmetric (except for [1 1])
        clear Pxn
    end
    % end
catch
    clear Pxn
end
I_mat=single(I_mat);
qrng=[5*min(hypot(q_mat_x(:),q_mat_y(:))) max(hypot(q_mat_x(:),q_mat_y(:)))]; % in 1/nm
% qv=linspace(-1/r_g,1/r_g,101);
qvx=q_mat_x;
qvy=q_mat_y;
qvr=sqrt(qvx.^2+qvy.^2);
maxq=max(qvr(:));
g_rat=[];
pxx=5;
pxy=1;
% PSF pixel nums pxx and pxy must be odd!
pxx=pxx+2*(pxx==pxy);
if exist("Pxn","var")
    pxx=Pxn(3);
    pxy=Pxn(4);
end

%         H=ones(pxx,pxy)';
%         H=exp(-linspace(-2,2,pxy).^2/2)'*exp(-linspace(-2,2,pxx).^2/2); %gaussian
H=exp(-linspace(-signum,signum,pxy).^2/2)'*exp(-linspace(-signum,signum,pxx).^2/2); %gaussian
H=single(H/sum(H(:))); %PSF shape
F2=filter2(H,I_mat,"same"); %applying the PSF
% % Zero-pad kernel to image size once
% KH = fft2( H, size(I_mat,1), size(I_mat,2) );   % H = gy(:) * gx(:).'
% FI = fft2(I_mat);
% F2  = real(ifft2(FI .* KH));
% gx = exp(-linspace(-signum,signum,pxx).^2/2);  gx = gx / sum(gx);
% gy = exp(-linspace(-signum,signum,pxy).^2/2);  gy = gy / sum(gy);
% F2=imfilter(imfilter(I_mat, gy(:),'replicate','same'), ...
%                     gx(:).','replicate','same');


% sigma_pixels_x = (pxx-1)/(2*signum);  % since your grid spans ±signum
% sigma_pixels_y = (pxy-1)/(2*signum);
%
% F2 = imgaussfilt(I_mat, [sigma_pixels_y, sigma_pixels_x], ...
%                 'FilterSize', [pxy pxx], ...
%                 'Padding', 'replicate');
opx=[pxx pxy];
% pxx=pxx+2;
% pxy=pxy+8;

pxx=1;pxy=9;
if exist("Pxn","var")
    pxx=Pxn(1);
    pxy=Pxn(2);
end

H=exp(-linspace(-signum,signum,pxy).^2/2)'*exp(-linspace(-signum,signum,pxx).^2/2); %gaussian
H=single(H/sum(H(:))); %PSF shape
F=filter2(H,I_mat,"same"); %applying the PSF
dqpix=mean(diff((qvy(:,1)))); %q diff between pixels, assuming qy and qx spacing are the same
% sigma_pixels_x = (pxx-1)/(2*signum);  % since your grid spans ±signum
% sigma_pixels_y = (pxy-1)/(2*signum);
%
% F = imgaussfilt(I_mat, [sigma_pixels_y, sigma_pixels_x], ...
%                 'FilterSize', [pxy pxx], ...
%                 'Padding', 'replicate');

% % Zero-pad kernel to image size once
% KH = fft2( H, size(I_mat,1), size(I_mat,2) );   % H = gy(:) * gx(:).'
% FI = fft2(I_mat);
% F  = real(ifft2(FI .* KH));
% gx = exp(-linspace(-signum,signum,pxx).^2/2);  gx = gx / sum(gx);
% gy = exp(-linspace(-signum,signum,pxy).^2/2);  gy = gy / sum(gy);
% F=imfilter(imfilter(I_mat, gy(:),'replicate','same'), ...
%                     gx(:).','replicate','same');
%% extract the guinier radius (r_g^2(1+V))
if 0,
    rng=find(qvr<dqpix*45& qvr>0*qrng(1)/3);
    [~,temp]=sort(qvr(rng));
    rng=rng(temp); %sorted by q
    npoly=2;
    [temp,S]=polyfit(qvr(rng).^2,log(F(rng)),npoly);
    [R, df] = deal(S.R, S.df);


    [temp2,S]=polyfit(qvr(rng).^2,log(F2(rng)),npoly);
    [temp3,S]=polyfit(qvr(rng).^2,log(I_mat(rng)),npoly);
    covp = (S.normr^2 / df) * inv(R)*inv(R)';   % covariance matrix
    alpha = 0.05;                     % 95% CI
    tval = 1.96;
    se_slope = sqrt(covp(end-1,end-1));   % std error of slope coeff
    CI_slope = [temp3(end-1) - tval*se_slope, temp3(end-1) + tval*se_slope];
end
%
%         figure(8)
%         plot(qvr(rng).^2,log(F(rng)),'o',qvr(rng).^2,log(F2(rng)),'o',qvr(rng).^2,polyval(temp,qvr(rng).^2),'-',qvr(rng).^2,polyval(temp2,qvr(rng).^2),'-')
%         hold on
%         plot(qvr(rng).^2,log(I_mat(rng)),'.',qvr(rng).^2,polyval(temp3,qvr(rng).^2),'-')
%         hold off
%         title(['guinier plot (lnI vs q^2) ' num2str(sqrt(-3*[temp(2) temp2(2) temp3(2)]))])
%         disp(['guinier plot (lnI vs q^2) ' num2str(sqrt(-3*[temp(end-1) temp2(end-1) temp3(end-1)]))])


% R_g_est=-3/2*(temp(end-1)+temp2(end-1));  %r_g^2*(1+V)
R_g_est=RG2;
R_g_nofilt=RG2;  %r_g^2*(1+V)
Pxn=[ pxx pxy opx];
% deadpix=sum(Pxn);%max(size(H))*2;
deadpix=2*max(Pxn);%max(size(H))*2;
% disp((max(Pxn)/signum*dqpix*R_g_est)^2*2/3)


qrng=([0*deadpix*dqpix min((maxq-deadpix*dqpix),1.4765/sqrt(R_g_nofilt))]); % relevant range to avoid the boundaries when filtering
qrng=([0*deadpix*dqpix min((maxq-deadpix*dqpix),1.35/sqrt(R_g_nofilt))]); % relevant range to avoid the boundaries when filtering
qrng=([0*deadpix*dqpix min((maxq-deadpix*dqpix),0.9/sqrt(R_g_nofilt))]); % relevant range to avoid the boundaries when filtering
qrng=([0*deadpix*dqpix min((maxq-deadpix*dqpix),0.79/sqrt(R_g_nofilt))]); % relevant range to avoid the boundaries when filtering. Lower upper limit is more accurate for real form-factor (with 3rd order phi''')



if diff(qrng)<0 % not enough pixels
    error('not enough pixels- consider using a smaller slit')
    qrng(2)=0.9/sqrt(R_g_nofilt);
end

% flattening w/o averaging
qvt=atan2(qvy,qvx);

G2f=log(F2(:)./F(:));

% now we fit the ln difference to a function of the form
% g_0+g_1*Q+g_2*Q^2+cos(2theta)*Q*(m_0+m_1*Q+m_2*Q^2)
% where Q=q^2*r_g^2
rng=find(qvr<qrng(2)& qvr>qrng(1));
% res = fit_I_r_theta_ratios(qvr(rng).^2*r_g^2,qvt(rng),G2f(rng));
% res = fit_I_r_theta_ratios(qvr(rng).^2,qvt(rng),G2f(rng));
warning('error', 'MATLAB:singularMatrix');  % treat that warning as error
warning('error', 'MATLAB:nearlySingularMatrix');  % treat that warning as error
RG2=R_g_est;
try
    % res = fit_I_r_theta_ratios_weighted_centered(qvr(rng).^2,qvt(rng),G2f(rng),0*ones(size(rng))+1*sqrt(I_mat(rng)));
    res = fit_I_r_theta_ratios_weighted_centered(qvr(rng).^2,qvt(rng),double(G2f(rng)),double(0*ones(size(rng))+1*sqrt(I_mat(rng))),use_r3,use_g3);
    g_coeff=res.p(1:3);
    m_coeff=res.p(4:5);
    g_rat=double([g_coeff(2)/g_coeff(1), res.CI95_G(1), res.g100_ratio, m_coeff(2)/m_coeff(1), res.CI95_M(1), res.g210_ratio,g_coeff(3)/g_coeff(2), res.g100_CI95(1) res.g21_CI95(1) res.g210_CI95(1)]');
%replacing m with m210
    g_rat=double([g_coeff(2)/g_coeff(1), res.CI95_G(1), res.g100_ratio, res.m210_ratio, res.m210_CI95(1), res.g210_ratio,g_coeff(3)/g_coeff(2), res.g100_CI95(1) res.g21_CI95(1) res.g210_CI95(1)]');

    %     g_rat=double([g_coeff(2)/g_coeff(1), res.CI95_G(1), res.CI95_G(2)*1, m_coeff(2)/m_coeff(1), res.CI95_M(1), res.CI95_M(2)*1,g_coeff(3)/g_coeff(2), R_g_est*1 res.g21_CI95]');
catch
    g_rat=nan*(ones(10,1));
end
RG2=double(R_g_nofilt);
%res.p is the coefficient vector [g0 g1 g2 m0 m1 m2]

if 0
    figure(1)
    r_g=1;
    rng=find(qvr<qrng(2)& qvr>qrng(1));
    p=res.p;
    r0 = linspace(qrng(1),qrng(2),50).^2*r_g^2;
    tm=qvr(rng).^2*r_g^2;
    Gfit = p(1) + p(2)*r0 + p(3)*r0.^2;
    Mfit = p(4) + p(5)*r0 ;
    if length(p)>5
    Mfit = Mfit + p(6)*r0.^2;
    plot3(cos(2*qvt(rng)),qvr(rng).^2*r_g^2,G2f(rng),'+',cos(2*qvt(rng)),tm,p(1) + p(2)*tm + p(3)*tm.^2+tm.*(p(4) + p(5)*tm + p(6)*tm.^2).*cos(2*qvt(rng)),'go',zeros(size(r0)),r0,Gfit,'b',ones(size(r0)),r0,Gfit+r0.*Mfit,'r',-ones(size(r0)),r0,Gfit-r0.*Mfit,'r')
    else
    plot3(cos(2*qvt(rng)),qvr(rng).^2*r_g^2,G2f(rng),'k.',cos(2*qvt(rng)),tm,p(1) + p(2)*tm + p(3)*tm.^2+tm.*(p(4) + p(5)*tm ).*cos(2*qvt(rng)),'go',zeros(size(r0)),r0,Gfit,'b',ones(size(r0)),r0,Gfit+r0.*Mfit,'r',-ones(size(r0)),r0,Gfit-r0.*Mfit,'r')
    end
    xlabel('cos(2\chi)')
    ylabel('Q')
    legend('data','full fit','G(Q) fit','G(Q)\pmM(Q) fit')
    grid on
end


% for M=[0 2]
%     figure(2+M)
%     for i=1:inu
%         h2=plot(ax,g_rat(1+1.5*M,((i-1)*iV+1):(i*iV)),'v:','displayname',['\phi''''=' num2str(nulist(i))]);
%
%         for tmp=1:length(h2)
%             %             h2(tmp).Annotation.LegendInformation.IconDisplayStyle = 'off';
%         end
%         for tmp=1:length(h2)
%             h2(tmp).Color=h2(1).Color;
%         end
%         hold on
%         nu=nulist(i);
%
%         xlabel ('V')
%         if ~M
%             ylabel('g_1/R_0^2/g_0')
%         else
%             ylabel('m_2/R_0^2/m_1')
%
%         end
%         hold on
%     end
%     if  flag_show_legend,
%         for i=[2 4]
%             figure (i)
%             h=legend;
%             %             set(h,'location','bestoutside')
%             set(h,'location','best')
%         end
%     end
% end
end