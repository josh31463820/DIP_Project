import numpy as np
from numpy.fft import fft2, ifft2
# import cv2
# import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



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

def blockRmv(img, beta=1):
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

# img = 'text.png'  # Replace with your actual path
# y = cv2.imread(img, cv2.IMREAD_COLOR)
# y = y.astype(np.float64) / 255.0  # Normalize to [0, 1]
# y2 = blockRmv(y)

# plt.figure(figsize=(12, 6))
# plt.imshow(np.hstack([y ,y2]),cmap='gray')
# plt.title('The decomposed structure & texture layers')
# plt.axis('off')
# plt.show()
