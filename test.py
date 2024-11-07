import numpy as np
import cv2
import matplotlib.pyplot as plt
from skimage.restoration import (denoise_bilateral,denoise_tv_chambolle)
from skimage.util import view_as_blocks

# Read the compressed image
compressed_img = 'test.png'  # Replace with your actual path
y = cv2.imread(compressed_img, cv2.IMREAD_COLOR)
y = y.astype(np.float64) / 255.0  # Normalize to [0, 1]

# Read the uncompressed image
original_img = 'train.png'
x = cv2.imread(original_img, cv2.IMREAD_COLOR)
x = x.astype(np.float64) / 255.0  # Normalize to [0, 1]

# Crop image to be a multiple of 8
H, W, D = y.shape
H = (H // 8) * 8
W = (W // 8) * 8

x = x[:H, :W, :]
y = y[:H, :W, :]

# Generate a curve f, g is the first derivative of function f

def curve(m, sig):
    def f(x):
        return 255 / (1 + np.exp(sig * (-x + m)))

    def g(x):
        exp_term = np.exp(sig * (-x + m))
        return sig * 255 * exp_term / (1 + exp_term) ** 2

    return f, g

f, g = curve(100, 0.04)

# Apply the curve to compressed and uncompressed image
x_f = f((x * 255).astype(int)) / 255
y_f_compressed = f((y * 255).astype(int)) / 255

# Structure-texture separation (placeholder function)


y_struct = denoise_tv_chambolle(y*255.0, weight=10, channel_axis=2)/255.0
y_text = y- y_struct

#print

# Display the results
# plt.figure(figsize=(12, 6))
# plt.imshow(np.hstack([ cv2.cvtColor(y_struct.astype(np.float32), cv2.COLOR_BGR2RGB), cv2.cvtColor(y_text.astype(np.float32)*5+0.5, cv2.COLOR_BGR2RGB)]))
# plt.title('The decomposed structure & texture layers')
# plt.axis('off')
# plt.show()

#  boost structure
y_f_struct = f(y_struct*255)/255;

#  computer the ratio
#  multi is the factor K in equation  (6)
multi = g(y_struct*255);  #multi = g(y_struct);
y_f_text = multi*y_text;

# plt.imshow(np.hstack([ cv2.cvtColor(y_f_struct.astype(np.float32), cv2.COLOR_BGR2RGB), cv2.cvtColor((y_f_struct+y_f_text).astype(np.float32), cv2.COLOR_BGR2RGB)]))
# plt.title('boosted structure & boosted structure+texture')
# plt.axis('off')
# plt.show()




# Define the weight matrix
w = np.ones((8, 8))
w[0, 0] = 0
w[0, 1] = 0
w[1, 0] = 0

# Define the function for each block






# def blockproc(image, block_size, func):
#     blocks = view_as_blocks(image, block_size)
#     result = np.zeros_like(image)
#     for i in range(blocks.shape[0]):
#         for j in range(blocks.shape[1]):
#             block = blocks[i, j]
#             result[i*block_size[0]:(i+1)*block_size[0], j*block_size[1]:(j+1)*block_size[1]] = func(block)
#     return result


# def text_idx_fun(block):
#     # Compute the DCT and apply the weight matrix
#     dct_block = cv2.dct(block.astype(np.float32))
#     return np.sum(np.abs(dct_block) * w)

# text_idx = []
# for c in range(3):
#     text_idx.append(blockproc(y_text[:, :, c], (8, 8), text_idx_fun))
# text_idx = np.max(text_idx, axis=0)
# text_reg = text_idx > 0.1
# text_map = blockproc(text_reg, (1, 1), lambda block: np.ones((8, 8)) * block)

# Compute the texture index map with the block-wise function
# text_idx = []
# for c in range(3):  # Assuming RGB channels
#     text_idx.append(blockproc(y_text[:, :, c], (8, 8), text_idx_fun))
# text_idx = np.max(text_idx, axis=0)

# # Threshold `text_idx` to create a binary texture map
# text_reg = text_idx > 0.1  # This produces a binary map indicating texture

# # Expand `text_reg` to create an 8x8 block for each element in `text_reg`
# text_map = np.kron(text_reg, np.ones((8, 8)))  # This duplicates each `text_reg` element into an 8x8 block


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

# Assuming `y_text` contains the separated texture from the image
text_idx = []
for c in range(3):  # RGB channels
    text_idx.append(custom_blockproc(y_text[:, :, c], (8, 8), text_idx_fun))
text_idx = np.max(text_idx, axis=0)

# Threshold `text_idx` to create a binary map `text_reg`
text_reg = (text_idx > 0.1).astype(int)

# Expand `text_reg` to an 8x8 block-based map
text_map = custom_blockproc(text_reg, (1, 1), lambda block: block[0, 0], expand_block=True)



# Display the texture map
plt.figure()
plt.imshow(text_map, cmap='gray')
plt.title('Texture Map')
plt.show()

