 clear  
 close all
 
%% loading data
casePath = 'images' ;


y = im2double(imread(fullfile(casePath,'dehaze_80.jpg')));
addpath('haze-removal');


% crop image to 8X
[H W D] = size(y);
H = floor(H/8)*8;
W = floor(W/8)*8;
y = y(1:H,1:W,:);


[y_text, y_struct] = TV_L2_Decomp(y, 0.03) ;
figure,imshow([y_text*10+0.5,y_struct]);
title('the decomposed texture  &  structure layer');
 
[I ,y_f_struct ,J ,T_est ,T, A] = removeHaze(y_struct,15);

figure,imshow([y_text*10+0.5,y_f_struct]);
title('texture layer  & dehazed__structure layer')

multi = repmat(1./T,[1,1,3]);
w = ones(8,8);
w(1,1) = 0;w(1,2) = 0;w(2,1) = 0;
text_idx_fun = @(block_struct) sum(sum((abs(dct2(block_struct.data)).*w)));
text_idx = [];

for c = 1:3
    text_idx(:,:,c) = blockproc(y_text(:,:,c),[8 8],text_idx_fun);
end
text_idx = max(text_idx,[],3);
text_reg = text_idx>0.2;
text_map = blockproc(text_reg,[1 1],@(block_struct) ones(8,8)*block_struct.data);

sig=1e-5;
text_map_refined = generateLaplacian2f(y_f_struct, text_map,sig);
thr = 0.7;
ff = curve(thr*255, 0.04);
text_map_refined2 = ff(text_map_refined*255)/255;
figure;imshow([text_map text_map_refined2]);
title('the initial mask & the refined mask')

y_text_refined = blockRmv( y_text, 5);
figure,imshow([y_text*10+0.5,y_text_refined*10+0.5]);
title('texture layer & deblocked texture layer')
 

I_d_ours2=y_f_struct+multi.*y_text_refined.*repmat(text_map_refined2,[1 1 3]);
figure,imshow(I_d_ours2);title('final result')


 

 
        