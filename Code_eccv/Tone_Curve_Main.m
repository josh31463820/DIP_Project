close all;
clear 
%% loading data

addpath(genpath('haze-removal'));   % for the function "generateLaplacian2f"

casePath = 'images' ;

%compress=40;

% read the compressed image
y = im2double(imread(fullfile(casePath,'tone_curve_40.jpg')));

% read the uncompressed image
x = im2double(imread(fullfile(casePath,'tone_curve_img.png')));

% crop image to 8X
[H W D] = size(y); 
H = floor(H/8)*8;
W = floor(W/8)*8;

x = x(1:H,1:W,:);
y = y(1:H,1:W,:);


% generate a curve f, g is the first derivative of function f 
  [f g] = curve(100, 0.04);
 
% apply the curve to compressed and uncompressed image

  x_f = f(x*255)/255;
  y_f_compressed = f(y*255)/255;
 

%% structure_texture separation
[y_text y_struct] =  TV_L2_Decomp(y, 0.05) ; %ROF_decomp(y , 0.05, 100, 1) ;
figure,imshow([y_text*10+0.5,y_struct]);
title('the decomposed texture & structure layers')
 
%% boost structure
y_f_struct = f(y_struct*255)/255;
 
%% computer the ratio
%  multi is the factor K in equation  (6)
multi = g(y_struct*255);  %multi = g(y_struct);
y_f_text = multi.*y_text;

figure,imshow([y_f_struct,y_f_struct+multi.*y_text]);
title('boosted structure & boosted structure+texture')


%% compute texture map
w = ones(8,8);
w(1,1) = 0;w(1,2) = 0;w(2,1) = 0;

% define the function for each block 
text_idx_fun = @(block_struct) sum(sum((abs(dct2(block_struct.data)).*w)));
text_idx = [];
for c = 1:3
    text_idx(:,:,c) = blockproc(y_text(:,:,c),[8 8],text_idx_fun);
end
text_idx = max(text_idx,[],3);
text_reg = text_idx>0.1;
text_map = blockproc(text_reg,[1 1],@(block_struct) ones(8,8)*block_struct.data); 


%% refine the initial block-wise map using image matting algorithms
sig=1e-5;
text_map_refined = generateLaplacian2f(y_f_struct, text_map, sig);



%% refine the mask to be almost binary 
thr = 0.5;
ff = curve(thr*255, 0.05);
text_map_refined2 = ff(text_map_refined*255)/255;
figure,imshow([text_map,text_map_refined2]);
title('initial mask  & the refined mask');
%% remove block in texture
y_text_refined = blockRmv(y_text,5);

figure,imshow([10*y_text+0.5,10*y_text_refined+0.5]);
title('texture & the blocked texture');

%% final combination
y_f_recovery = y_f_struct+multi.*y_text_refined.*repmat(text_map_refined2,[1 1 3]);
figure,imshow(y_f_recovery),title('Final result');

 

 
        