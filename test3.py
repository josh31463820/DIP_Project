import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.fft import fft2, ifft2
from TV_L2_Decomp import TV_L2_Decomp
from blockRMV import blockRmv
from my_blockRMV import blockRmv2






case_path = 'images'
y = cv2.imread(f'{case_path}/reconstructed_image.png')
y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB) / 255.0  # 轉換圖片為 RGB 並標準化到 [0, 1]

y_text, y_struct = TV_L2_Decomp(y, 0.03)

y_text_deblock = blockRmv(y_text)
y_text_deblock_g = blockRmv2(y_text,gaussian_sigma=0.5)

plt.figure(figsize=(40,15))
plt.imshow(np.hstack([(y_text_deblock * 10 + 0.5), (y_text_deblock_g * 10 + 0.5)]), cmap='gray')
plt.title('original v.s gaussian')
plt.show()


