import numpy as np
from numpy.fft import fft2, ifft2
# import cv2
# import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter



def blockRmv(img, beta=10, gaussian_sigma=1):

    def psf2otf(psf, shape):
        pad_psf = np.zeros(shape)
        pad_psf[:psf.shape[0], :psf.shape[1]] = psf
        pad_psf = np.roll(pad_psf, -psf.shape[0] // 2, axis=0)
        pad_psf = np.roll(pad_psf, -psf.shape[1] // 2, axis=1)
        return fft2(pad_psf)

    fx = np.array([[1, -1]])  # Filter for x-direction
    fy = np.array([[1], [-1]])  # Filter for y-direction
    N, M, D = img.shape
    sizeI2D = (N, M)

    # Compute OTFs for x and y filters
    otfFx = psf2otf(fx, sizeI2D)
    otfFy = psf2otf(fy, sizeI2D)

    Normin1 = fft2(img)  # Fourier transform of the image
    Denormin2 = np.abs(otfFx)**2 + np.abs(otfFy)**2

    if D > 1:
        Denormin2 = np.repeat(Denormin2[:, :, np.newaxis], D, axis=2)

    Denormin = 1 + beta * Denormin2

    # Horizontal (u) and vertical (v) gradients
    u = np.concatenate((np.diff(img, axis=1), img[:, :1, :] - img[:, -1:, :]), axis=1)
    v = np.concatenate((np.diff(img, axis=0), img[:1, :, :] - img[-1:, :, :]), axis=0)

    # Apply blocking to gradients every 8 pixels
    u[:, 7::8, :] = 0
    v[7::8, :, :] = 0

    # Apply Gaussian blur to reduce blocking artifacts in u and v
    u = gaussian_filter(u, sigma=gaussian_sigma)
    v = gaussian_filter(v, sigma=gaussian_sigma)

    # Construct the S subproblem for Fourier space
    Normin2 = np.concatenate((u[:, -1:, :] - u[:, :1, :], -np.diff(u, axis=1)), axis=1)
    Normin2 += np.concatenate((v[-1:, :, :] - v[:1, :, :], -np.diff(v, axis=0)), axis=0)

    # Fourier space computation of the final result
    FS = (Normin1 + beta * fft2(Normin2, axes=(0, 1))) / Denormin
    out = np.real(ifft2(FS, axes=(0, 1)))

    # # Optional: Normalize to improve visualization
    # out_min, out_max = out.min(), out.max()
    # if out_max > out_min:  # Avoid division by zero
    #     out = (out - out_min) / (out_max - out_min)  # Scale to [0, 1]


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
