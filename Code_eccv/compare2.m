% clc;

% casePath = 'enhance_data\02' ;

casePath = 'images' ;
compress=80;

%%% ground truth
x_f = im2double(imread(fullfile(casePath,'tone_curve_GT.png')));
%% compressed
y_f_compressed = im2double(imread(fullfile(casePath,'tone_curve_compressed_boost.png')));
[H W D] = size(y_f_compressed);
H = floor(H/8)*8;
W = floor(W/8)*8;
y_f_compressed= y_f_compressed(1:H,1:W,:);

%     MSE_comp = mse (y_f_compressed*255,x_f*255 );
SSIM_comp = ssim_index(mean(y_f_compressed*255,3),mean(x_f*255,3) );
PSNR_comp = psnr (y_f_compressed*255,x_f*255);
%     fprintf('Do-Nothing\t MSE =%4.2f, SSIM =%.4f, PSNR = %4.2f,\n',MSE_comp,SSIM_comp,PSNR_comp);
fprintf('Do-Nothing\t  SSIM =%.4f, PSNR = %4.2f,\n', SSIM_comp,PSNR_comp);



%% ours
y_f_recovery= im2double(imread(fullfile(casePath,'tone_curve_ours.png')));
%     MSE_ours = mse (y_f_recovery*255,x_f*255 );
SSIM_ours = ssim_index(mean(y_f_recovery*255,3),mean(x_f*255,3) );
PSNR_ous = psnr (y_f_recovery*255,x_f*255);
%     fprintf('Our Recovery\t MSE =%4.2f, SSIM =%.4f, PSNR = %4.2f,\n',MSE_ours,SSIM_ours,PSNR_ous);
fprintf('Our Recovery\t   SSIM =%.4f, PSNR = %4.2f,\n' , SSIM_ours,PSNR_ous);

% 
% %% foi_after
% foi = im2double(imread(fullfile(casePath,'06_40b_d_foi_after.png')));   %80b_foi2.png 80_b_d_foi_after.png
% %     MSE_foi = mse (foi*255,x_f*255 );
% SSIM_foi = ssim_index(mean(foi*255,3),mean(x_f*255,3) );
% PSNR_foi = psnr (foi*255,x_f*255);
% %     fprintf('Foi(post)\t MSE =%4.2f, SSIM =%.4f, PSNR = %4.2f,\n',MSE_foi,SSIM_foi,PSNR_foi);
% fprintf('Foi(post)\t   SSIM =%.4f, PSNR = %4.2f,\n' , SSIM_foi,PSNR_foi);
% 
% 
% %%%  foi_before
% foi_before = im2double(imread(fullfile(casePath,'06_40b_d_foi_before.png')));   %80b_foi2.png 80_b_d_foi_after.png
% %     MSE_foi_before = mse (foi_before*255,x_f*255 );
% SSIM_foi_before = ssim_index(mean(foi_before*255,3),mean(x_f*255,3) );
% PSNR_foi_before = psnr (foi_before*255,x_f*255);
% %     fprintf('Foi(before)\t MSE =%4.2f, SSIM =%.4f, PSNR = %4.2f,\n',MSE_foi_before,SSIM_foi_before,PSNR_foi_before);
% fprintf('Foi(before)\t   SSIM =%.4f, PSNR = %4.2f,\n', SSIM_foi_before,PSNR_foi_before);
% 
% 
% 
% 
% sun_res=  im2double(imread(fullfile(casePath,'06_40b_d_Sun.png')));   %80b_foi2.png 80_b_d_foi_after.png
% %     MSE_sun_res = mse (sun_res*255,x_f*255 );
% SSIM_sun_res = ssim_index(mean(sun_res*255,3),mean(x_f*255,3) );
% PSNR_sun_res = psnr (sun_res*255,x_f*255);
% fprintf('Sun \t  SSIM =%.4f, PSNR = %4.2f,\n', SSIM_sun_res,PSNR_sun_res);
% 
% learn1_res=  im2double(imread(fullfile(casePath,'06_40b_d_NN.png')));   %80b_foi2.png 80_b_d_foi_after.png
% %     MSE_sun_res = mse (sun_res*255,x_f*255 );
% SSIM_learn1_res= ssim_index(mean(learn1_res*255,3),mean(x_f*255,3) );
% PSNR_learn1_res= psnr (learn1_res*255,x_f*255);
% fprintf('Learning 1: \t  SSIM =%.4f, PSNR = %4.2f,\n', SSIM_learn1_res,PSNR_learn1_res);