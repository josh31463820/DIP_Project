import numpy as np
import cv2
import matplotlib.pyplot as plt
# from skimage.restoration import (denoise_bilateral,denoise_tv_chambolle)
from TV_L2_Decomp import TV_L2_Decomp
# from laplacian import generate_refined_mask
from blockRMV import blockRmv

# Read the compressed image
case_path = 'images'
y = cv2.imread(f'{case_path}/reconstructed_image.png', cv2.IMREAD_COLOR)
y = y.astype(np.float64) / 255.0  # Normalize to [0, 1]

# Read the uncompressed image
# original_img = 'train.png'
# x = cv2.imread(original_img, cv2.IMREAD_COLOR)
# x = x.astype(np.float64) / 255.0  # Normalize to [0, 1]

# Crop image to be a multiple of 8
H, W, D = y.shape
H = (H // 8) * 8
W = (W // 8) * 8

# x = x[:H, :W, :]
y = y[:H, :W, :]

# Generate a curve f, g is the first derivative of function f

def curve(m, sig):
    # 定義 Sigmoid 函式
    f = lambda x: 255 / (1 + np.exp(sig * (-x + m)))

    # 定義 Sigmoid 函式的導數
    g = lambda x: (sig * 255 * np.exp(sig * (-x + m))) / (1 + np.exp(sig * (-x + m)))**2

    return f, g

f, g = curve(100, 0.04)

# Apply the curve to compressed and uncompressed image
# x_f = f((x * 255).astype(int)) / 255
y_f_compressed = f((y * 255).astype(int)) / 255

# Structure-texture separation (placeholder function)


y_text, y_struct = TV_L2_Decomp(y, 0.03)


#  boost structure
y_f_struct = f(y_struct*255)/255;

#  computer the ratio
#  multi is the factor K in equation  (6)
multi = g(y_struct*255);  #multi = g(y_struct);
y_f_text = multi*y_text;


def custom_blockproc(image, block_size, func, expand_block=False):
    # Initialize the result array
    if expand_block:
        # Expand dimensions to be able to store each block as an 8x8 region
        result = np.zeros((image.shape[0] * block_size[0], image.shape[1] * block_size[1]), dtype=np.float64)
    else:
        result = np.zeros_like(image, dtype=np.float64)

    # Iterate over blocks
    for i in range(0, image.shape[0], block_size[0]):
        for j in range(0, image.shape[1], block_size[1]):
            # Extract the current block
            current_block = image[i:i + block_size[0], j:j + block_size[1]]
            block_result = func(current_block)

            if expand_block:
                # Expand result to 8x8 if `expand_block` is True
                expanded_block = np.ones(block_size) * block_result
                # Assign to the correct location in the result
                result[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]] = expanded_block
            else:
                result[i:i + block_size[0], j:j + block_size[1]] = block_result
    return result

# Weight matrix for DCT-based function
w = np.ones((8, 8))
w[0, 0] = 0
w[0, 1] = 0
w[1, 0] = 0

# Define the function to calculate the texture index for each block
def text_idx_fun(block):
    dct_block = cv2.dct(block.astype(np.float32))
    return np.sum(np.abs(dct_block) * w)

text_idx = []
for c in range(3):  # RGB channels
    text_idx.append(custom_blockproc(y_text[:, :, c], (8, 8), text_idx_fun))
text_idx = np.max(text_idx, axis=0)

# Threshold `text_idx` to create a binary map `text_reg`
text_reg = (text_idx > 0.1).astype(int)

# Expand `text_reg` to an 8x8 block-based map
text_map = custom_blockproc(text_reg, (1, 1), lambda block: block[0, 0], expand_block=True)




# refine the initial block-wise map using image matting algorithms
# text_map_refined = generate_refined_mask(y_f_struct, text_map)


# plt.figure()
# plt.imshow(text_map_refined, cmap='gray')
# plt.title('Refines Texture Map')
# plt.show()

# thr = 0.5
# ff,gg = curve(thr * 255, 0.05)
# text_map_refined2 = ff(text_map * 255) / 255  # Applying the curve function


# Remove blocks in texture
y_text_refined = blockRmv(y_text, 5)
# Final combination
text_map_refined2_repeated = np.repeat(text_map[:, :, np.newaxis], 3, axis=2)
y_f_recovery = y_struct + multi * y_text_refined

# Display final result
# plt.figure()
# plt.imshow(cv2.cvtColor(y_f_recovery.astype(np.float32), cv2.COLOR_BGR2RGB))
# plt.title('Final result')
# plt.show()

cv2.imwrite("result_no_map_no_g.png",y_f_recovery*255)
cv2.imwrite("result_map_no_g.png",y_f_recovery2*255)
