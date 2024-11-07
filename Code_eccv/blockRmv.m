function out = blockRmv( img, beta)

if nargin<2
    beta = 20;
end

fx = [1, -1];
fy = [1; -1];
[N,M,D] = size(img);
sizeI2D = [N,M];
otfFx = psf2otf(fx,sizeI2D);
otfFy = psf2otf(fy,sizeI2D);
Normin1 = fft2(img);
Denormin2 = abs(otfFx).^2 + abs(otfFy ).^2;

if D>1
    Denormin2 = repmat(Denormin2,[1,1,D]);
end

    Denormin   = 1 + beta*Denormin2;
    
    % h-v subproblem
    u = [diff(img,1,2), img(:,1,:) - img(:,end,:)];
    v = [diff(img,1,1); img(1,:,:) - img(end,:,:)];
    u(:,8:8:end,:) = 0;
    v(8:8:end,:,:) = 0;
   

    % S subproblem
    Normin2 = [u(:,end,:) - u(:, 1,:), -diff(u,1,2)];
    Normin2 = Normin2 + [v(end,:,:) - v(1, :,:); -diff(v,1,1)];
    FS = (Normin1 + beta*fft2(Normin2))./Denormin;
    out = real(ifft2(FS));



