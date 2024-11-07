import numpy as np
from scipy.fftpack import fft2, ifft2
from scipy.signal import convolve2d
import cv2
from skimage.filters import gaussian
from scipy.ndimage import uniform_filter
from scipy.fftpack import dct
from skimage.util import view_as_blocks
import matplotlib.pyplot as plt
from skimage.util import view_as_windows
from scipy.sparse import eye, spdiags, block_diag, csr_matrix, diags, kron
from scipy.sparse.linalg import spsolve
from skimage.color import rgb2gray
from scipy.ndimage import convolve
from scipy.linalg import inv
from scipy.sparse.linalg import inv
from scipy.ndimage import generic_filter
from scipy.ndimage import laplace
from scipy import ndimage
from scipy.fft import fft2, ifft2

def psf2otf(psf, shape):
    """
    Convert a point spread function (PSF) to an optical transfer function (OTF).
    """
    psf_pad = np.zeros(shape)
    psf_pad[:psf.shape[0], :psf.shape[1]] = psf
    
    for axis, axis_size in enumerate(psf.shape):
        psf_pad = np.roll(psf_pad, -int(axis_size / 2), axis=axis)

    otf = fft2(psf_pad)
    n_ops = np.sum(psf_pad.size * np.log2(psf_pad.shape))
    otf[np.abs(otf) < n_ops * np.finfo(otf.dtype).eps] = 0
    return otf

def TV_L2_Decomp(Im, lambda_val=2e-2):
    S = Im.astype(np.float64)
    betamax = 1e5
    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    
    N, M, D = Im.shape
    sizeI2D = (N, M)
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)
    
    Normin1 = fft2(S, axes=(0, 1))
    Denormin2 = np.abs(otfFx)**2 + np.abs(otfFy)**2
    
    if D > 1:
        Denormin2 = np.repeat(Denormin2[:, :, np.newaxis], D, axis=2)
    
    beta = 2 * lambda_val
    
    while beta < betamax:
        lambeta = lambda_val / beta
        Denormin = 1 + beta * Denormin2
        
        # h-v subproblem
        u = np.concatenate([np.diff(S, axis=1), S[:, :1, :] - S[:, -1:, :]], axis=1)
        v = np.concatenate([np.diff(S, axis=0), S[:1, :, :] - S[-1:, :, :]], axis=0)
        u = np.sign(u) * np.maximum(np.abs(u) - lambeta, 0)
        v = np.sign(v) * np.maximum(np.abs(v) - lambeta, 0)
        
        # S subproblem
        Normin2 = np.concatenate([u[:, -1:, :] - u[:, :1, :], -np.diff(u, axis=1)], axis=1)
        Normin2 += np.concatenate([v[-1:, :, :] - v[:1, :, :], -np.diff(v, axis=0)], axis=0)
        
        FS = (Normin1 + beta * fft2(Normin2, axes=(0, 1))) / Denormin
        S = np.real(ifft2(FS, axes=(0, 1)))
        
        beta *= 2
        print('.', end='')
    
    T = Im - S
    print()
    return T, S

def imread_double(imageName):
    # Read image and normalize to range [0,1]
    return cv2.imread(imageName).astype(np.float64) / 255

def make_dark_channel(I, patch_size):
    # Convert to grayscale using minimum across the color channels
    min_channel = np.min(I, axis=2)
    kernel = np.ones((patch_size, patch_size))
    dark_channel = uniform_filter(min_channel, size=patch_size)
    return dark_channel

def estimate_A(I, J, numBrightestPixels):
    # Estimate atmospheric light by choosing the brightest pixels in the dark channel
    dimJ = J.shape
    flat_J = J.flatten()
    flat_I = I.reshape((-1, 3))
    indices = np.argpartition(flat_J, -numBrightestPixels)[-numBrightestPixels:]
    A = np.max(flat_I[indices], axis=0)
    return A

def generate_laplacian(I, T_est, sig):
    # Applying Gaussian filter to refine transmission map
    return gaussian(T_est, sigma=sig)

def bilateral_filter(T, w, sigma):
    # Using OpenCV bilateral filter to smooth transmission map
    return cv2.bilateralFilter(T.astype(np.float32), w, sigma[0], sigma[1])

def remove_haze(imageName, patch_size):
    # Read the input image
    if isinstance(imageName, str):
        I = imread_double(imageName)
    else:
        I = imageName.astype(np.float64)
    
    # Set aerial perspective constant
    aerialPerspective = 0.95
    
    # Ensure image is 3-channel (color)
    if len(I.shape) == 2:
        I = np.stack([I]*3, axis=2)
    
    # Step 1: Compute the dark channel
    J = make_dark_channel(I, patch_size)
    
    # Step 2: Estimate atmospheric light A
    dimJ = J.shape
    numBrightestPixels = int(0.001 * dimJ[0] * dimJ[1])
    A = estimate_A(I, J, numBrightestPixels)
    
    # Step 3: Estimate transmission (Equation 12)
    T_est = 1 - aerialPerspective * make_dark_channel(I / A, patch_size)
    
    # Step 4: Refine the transmission map using Gaussian smoothing
    sig = 1e-4
    T1 = generate_laplacian(I, T_est, sig)
    T1 = np.clip(T1, 0, 1)
    
    # Step 5: Apply bilateral filter to refine the transmission map
    w = 5
    sigma = [3, 0.1]
    T = bilateral_filter(T1, w, sigma)
    
    # Step 6: Dehaze the image (Equation 16)
    dehazed = np.zeros_like(I)
    for c in range(3):
        im = I[:, :, c]
        a = A[c]
        dehazed[:, :, c] = (im - a) / np.maximum(T, 0.1) + a
    
    I_out = np.clip(dehazed, 0, 1)
    
    return I, I_out, J, T_est, T, A

def curve(m, sig):
    # 定義 Sigmoid 函式
    f = lambda x: 255 / (1 + np.exp(sig * (-x + m)))
    
    # 定義 Sigmoid 函式的導數
    g = lambda x: (sig * 255 * np.exp(sig * (-x + m))) / (1 + np.exp(sig * (-x + m)))**2
    
    return f, g

def blockRmv(img, beta=20):
    # 如果輸入影像是單通道，將其擴展為三通道
    if img.ndim == 2:
        img = np.expand_dims(img, axis=2)

    fx = np.array([1, -1])
    fy = np.array([[1], [-1]])
    N, M, D = img.shape
    sizeI2D = [N, M]
    
    # 計算OTF
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)
    
    Normin1 = fft2(img, axes=(0, 1))
    Denormin2 = np.abs(otfFx)**2 + np.abs(otfFy)**2

    if D > 1:
        Denormin2 = np.repeat(Denormin2[:, :, np.newaxis], D, axis=2)

    Denormin = 1 + beta * Denormin2
    
    # h-v 子問題
    u = np.concatenate((np.diff(img, axis=1), img[:, :1, :] - img[:, -1:, :]), axis=1)
    v = np.concatenate((np.diff(img, axis=0), img[:1, :, :] - img[-1:, :, :]), axis=0)
    
    # 每8個像素設置為0
    u[:, 7::8, :] = 0
    v[7::8, :, :] = 0
    
    # S 子問題
    Normin2 = np.concatenate((u[:, -1:, :] - u[:, :1, :], -np.diff(u, axis=1)), axis=1)
    Normin2 += np.concatenate((v[-1:, :, :] - v[:1, :, :], -np.diff(v, axis=0)), axis=0)
    
    # 計算傅立葉轉換後的輸出
    FS = (Normin1 + beta * fft2(Normin2, axes=(0, 1))) / Denormin
    out = np.real(ifft2(FS, axes=(0, 1)))

    return out

def curve(m, sig):

    def f(x):
        return 255.0 / (1 + np.exp(sig * (-x + m)))
    
    def g(x):
        return sig * 255.0 * np.exp(sig * (-x + m)) / (1 + np.exp(sig * (-x + m))) ** 2
    
    return f, g


# 定義DCT索引函數
def text_idx_fun(block, w):
    dct_block = dct(dct(block.T, norm='ortho').T, norm='ortho')  # 計算2D DCT
    return np.sum(np.abs(dct_block) * w)

# 區塊處理
def blockproc(img, block_size, func, *args):
    """
    模拟 MATLAB 中的 blockproc 函数，将图像按块处理。
    :param img: 输入图像（二维）
    :param block_size: 块大小 (行, 列)
    :param func: 处理块的函数
    :param args: 其他参数
    :return: 处理后的图像
    """
    blocks = view_as_blocks(img, block_size)
    output = np.zeros_like(img)

    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block = blocks[i, j]
            result = func(block, *args)
            if result.shape != block.shape:
                # 如果結果形狀不匹配，進行大小調整
                result = np.ones(block.shape) * result
            output[i * block_size[0]:(i + 1) * block_size[0], j * block_size[1]:(j + 1) * block_size[1]] = result

    return output

# 載入圖像
case_path = 'images'  # 圖片所在的資料夾
y = cv2.imread(f'{case_path}/reconstructed_image.png')  # 讀取圖片
y = cv2.cvtColor(y, cv2.COLOR_BGR2RGB) / 255.0  # 轉換圖片為 RGB 並標準化到 [0, 1]

# 將圖片裁剪為 8 的倍數
H, W, D = y.shape
H = (H // 8) * 8  # 高度取整數倍的 8
W = (W // 8) * 8  # 寬度取整數倍的 8
y = y[:H, :W, :]  # 裁剪圖片

# 進行 TV-L2 分解，得到紋理層和結構層
y_text, y_struct = TV_L2_Decomp(y, 0.03)

# 顯示紋理層和結構層
plt.imshow(np.concatenate([y_text * 10 + 0.5, y_struct], axis=1))
plt.title('the decomposed texture  &  structure layer')
plt.show()

# 結構層去霧
I, y_f_struct, J, T_est, T, A = remove_haze(y_struct, 15)

# 顯示去霧結果
plt.imshow(np.concatenate([y_text * 10 + 0.5, y_f_struct], axis=1))
plt.title('texture layer  & dehazed__structure layer')
plt.show()




# 計算紋理遮罩
multi = 1.0 / T  # 根據傳輸圖 T 計算權重
multi = np.repeat(multi[:, :, np.newaxis], 3, axis=2)  # 扩展维度以匹配 RGB 色频

# 初始化 8x8 的權重矩陣
w = np.ones((8, 8))
w[0, 0] = 0
w[0, 1] = 0
w[1, 0] = 0

# 定義 DCT 計算紋理索引的函數
def text_idx_fun(block):
    block_dct = dct(dct(block, axis=0, norm='ortho'), axis=1, norm='ortho')
    return np.sum(np.abs(block_dct) * w)

# 確保 y_text 是8的倍數（填充）
H, W, C = y_text.shape
pad_H = (8 - H % 8) % 8
pad_W = (8 - W % 8) % 8
y_text_padded = np.pad(y_text, ((0, pad_H), (0, pad_W), (0, 0)), mode='edge')

# 區塊處理
text_idx = np.zeros((y_text_padded.shape[0], y_text_padded.shape[1]))

for c in range(3):  # 對 RGB 三個通道進行處理
    blocks = view_as_blocks(y_text_padded[:, :, c], block_shape=(8, 8))
    text_idx_temp = np.zeros((y_text_padded.shape[0], y_text_padded.shape[1]))

    # 所有區塊
    for i in range(blocks.shape[0]):
        for j in range(blocks.shape[1]):
            block = blocks[i, j]
            # 計算紋理索引並填充到對應位置
            text_idx_temp[i * 8:(i + 1) * 8, j * 8:(j + 1) * 8] = text_idx_fun(block)

    # 更新纹理索引，取三个频道中的最大值
    text_idx = np.maximum(text_idx, text_idx_temp)

# 紋理索引閥值篩選
text_reg = text_idx > 0.2

# 將结果重新排列成 8x8 區塊
text_map = np.zeros_like(text_reg)

# 直接填充 text_map
text_map[text_reg] = 1  # 將篩選结果中的 True 設為 1

# 使用 plt.imshow 顯示紋理遮罩
plt.imshow(text_map, cmap='gray')
plt.title("Texture Mask")
plt.axis('off')  
plt.show()


def normalize_mask(mask, min_val=0, max_val=255):
    """
    Normalize mask values to a specific range
    
    Parameters:
    mask (ndarray): Input mask
    min_val (float): Minimum value for normalization (default: 0)
    max_val (float): Maximum value for normalization (default: 255)
    
    Returns:
    ndarray: Normalized mask
    """
    # Convert to float32 for better precision
    mask = mask.astype(np.float32)
    
    # Apply median filter for smoothing
    mask_filtered = ndimage.median_filter(mask, size=3)
    
    # Normalize to [0, 1]
    mask_normalized = (mask_filtered - np.min(mask_filtered)) / \
                     (np.max(mask_filtered) - np.min(mask_filtered))
    
    # Scale to desired range
    mask_scaled = mask_normalized * (max_val - min_val) + min_val
    
    return mask_scaled

# # 使用示例:
# # text_map_refined2 = normalize_mask(text_map_refined) / 255  # 正規化到 [0,1] 範圍# 設定正則化項 sig
# sig = 1e-6

# # 使用拉普拉斯運算生成精細紋理遮罩
# text_map_refined = generate_laplacian_2f(y_struct, text_map, sig)

# # 顯示初始遮罩與精細遮罩的比較
# plt.figure(figsize=(12, 6))

# # 顯示初始紋理遮罩
# plt.subplot(1, 2, 1)
# plt.imshow(text_map, cmap='gray', vmin=np.min(text_map), vmax=np.max(text_map))
# plt.title('初始遮罩')
# plt.axis('off')

# # 顯示精細紋理遮罩
# plt.subplot(1, 2, 2)
# plt.imshow(text_map_refined, cmap='gray', vmin=np.min(text_map_refined), vmax=np.max(text_map_refined))
# plt.title('精細遮罩')
# plt.axis('off')

# plt.show()



# 获取 y_struct 和 text_map 的基本信息
target_H, target_W = 390, 600

def crop_to_target_size(image, target_H, target_W):
    # 获取原始图像的尺寸
    H, W = image.shape[:2]
    
    # 计算需要裁切的像素数
    start_H = (H - target_H) // 2
    start_W = (W - target_W) // 2
    
    # 裁切图像到目标大小
    return image[start_H:start_H + target_H, start_W:start_W + target_W]

text_map_double = text_map.astype(np.double)

# 裁切 y_struct (假设它是 392x600x3 的图像)
y_struct_cropped = crop_to_target_size(y_struct, target_H, target_W)

# 裁切 text_map (假设它是 392x600 的图像)
text_map_cropped = crop_to_target_size(text_map_double, target_H, target_W)

# import numpy as np
# import matplotlib.pyplot as plt

# 假設 y_struct 和 text_map 已經定義為 NumPy 陣列
# 檢查變數的大小和數據類型
# print('y_struct size:')
# print(y_struct.shape)  # 輸出大小
# print('y_struct class:')
# print(y_struct.dtype)  # 輸出數據類型

# print('text_map size:')
# print(text_map_double.shape)  # 輸出大小
# print('text_map class:')
# print(text_map_double.dtype)  # 輸出數據類型

# # 查看值的範圍
# print('y_struct range:')
# print(np.min(y_struct), np.max(y_struct))  # 輸出值的範圍

# print('text_map range:')
# print(np.min(text_map_double), np.max(text_map_double))  # 輸出值的範圍

# # 查看數據的統計信息
# print('y_struct mean and std:')
# print(np.mean(y_struct), np.std(y_struct))  # 輸出均值和標準差

# print('text_map mean and std:')
# print(np.mean(text_map_double), np.std(text_map_double))  # 輸出均值和標準差

# # 顯示圖像
# plt.figure()
# plt.imshow(y_struct, cmap='gray')
# plt.title('y_struct')
# plt.show()

# plt.figure()
# plt.imshow(text_map_double, cmap='gray')
# plt.title('text_map')
# plt.show()




# # 输出 y_struct 的信息
# print("y_struct 信息:")
# print(f"形状: {y_struct_downsampled.shape}")
# print(f"数据类型: {y_struct_downsampled.dtype}")


# # 输出 text_map 的信息
# print("\ntext_map 信息:")
# print(f"形状: {text_map_downsampled.shape}")
# print(f"数据类型: {text_map_downsampled.dtype}")



# 檢查輸入數據
# check_input_data(y_struct_cropped, "y_struct_cropped")
# check_input_data(text_map_cropped, "text_map_cropped")
# # 顯示 y_text 的大小
# print(f'y_text 大小: {y_text.shape}')

# # 顯示數據類型
# print(f'y_text 類型: {y_text.dtype}')

# # 計算數據範圍
# y_min = np.min(y_text)
# y_max = np.max(y_text)
# print(f'y_text 範圍: {y_min:.4f} 至 {y_max:.4f}')

# # 計算均值和標準差
# y_mean = np.mean(y_text)
# y_std = np.std(y_text)
# print(f'y_text 均值和標準差: {y_mean:.4f}, {y_std:.4f}')





def psf2otf(psf, out_size):
    # 確認psf是2D 
    psf = np.atleast_2d(psf)
    
    psf_pad = np.zeros(out_size)
    psf_pad[:psf.shape[0], :psf.shape[1]] = psf
    
    # 將psf移動到中心
    for axis, axis_size in enumerate(psf.shape):
        psf_pad = np.roll(psf_pad, -int(axis_size / 2), axis=axis)
    
    otf = fft2(psf_pad)
    return otf

def blockRmv(img, beta=20):
    fx = np.array([1, -1])
    fy = np.array([[1], [-1]])
    N, M, D = img.shape
    sizeI2D = (N, M)
    
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)
    Normin1 = fft2(img, axes=(0, 1))
    Denormin2 = np.abs(otfFx)**2 + np.abs(otfFy)**2
    
    if D > 1:
        Denormin2 = np.repeat(Denormin2[:, :, np.newaxis], D, axis=2)
    
    Denormin = 1 + beta * Denormin2
    
    # h-v subproblem
    u = np.concatenate((np.diff(img, axis=1), img[:, :1, :] - img[:, -1:, :]), axis=1)
    v = np.concatenate((np.diff(img, axis=0), img[:1, :, :] - img[-1:, :, :]), axis=0)
    
    u[:, 7::8, :] = 0
    v[7::8, :, :] = 0
    
    # S subproblem
    Normin2 = np.concatenate((u[:, -1:, :] - u[:, :1, :], -np.diff(u, axis=1)), axis=1)
    Normin2 += np.concatenate((v[-1:, :, :] - v[:1, :, :], -np.diff(v, axis=0)), axis=0)
    
    FS = (Normin1 + beta * fft2(Normin2, axes=(0, 1))) / Denormin
    out = np.real(ifft2(FS, axes=(0, 1)))
    
    return out

y_text_refined = blockRmv(y_text, 5)  # 將紋理層去區塊化

# 顯示結果
plt.figure()
plt.imshow(np.hstack([(y_text * 10 + 0.5), (y_text_refined * 10 + 0.5)]), cmap='gray')
plt.title('deblock')
plt.show()
