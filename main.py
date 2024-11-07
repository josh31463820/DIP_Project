import numpy as np
import cv2
import matplotlib.pyplot as plt

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

def psf2otf(psf, size):
    # Convert the point spread function (PSF) to an optical transfer function (OTF)
    psf_shape = np.array(psf.shape)
    size_shape = np.array(size)
    pad_size = size_shape - psf_shape
    padded_psf = np.pad(psf, [(0, pad_size[0]), (0, pad_size[1])], mode='constant')
    otf = np.fft.fft2(padded_psf)
    return otf



def TV_L2_Decomp1(Im, lambda_val=2e-2):
    # Normalize the input image to [0, 1] range
    S = Im.astype(np.float64) / 255.0

    betamax = 1e5
    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    N, M, D = S.shape

    # Convert PSF to OTF
    def psf2otf(psf, size):
        psf_shape = np.array(psf.shape)
        pad_size = size - psf_shape
        padded_psf = np.pad(psf, [(0, pad_size[0]), (0, pad_size[1])], mode='constant')
        return np.fft.fft2(padded_psf)

    # Optical transfer functions for x and y
    otfFx = psf2otf(fx, (N, M))
    otfFy = psf2otf(fy, (N, M))
    Normin1 = np.fft.fft2(S)

    Denormin2 = np.abs(otfFx)**2 + np.abs(otfFy)**2
    if D > 1:
        Denormin2 = np.repeat(Denormin2[:, :, np.newaxis], D, axis=2)

    beta = 2 * lambda_val

    while beta < betamax:
        lambeta = lambda_val / beta
        Denormin = 1 + beta * Denormin2

        # h-v subproblem (adjusted to match MATLAB wrap-around logic)
        u = np.diff(S, axis=1, append=S[:, :1, :])
        v = np.diff(S, axis=0, append=S[:1, :, :])

        # Apply soft-thresholding
        u = np.sign(u) * np.maximum(np.abs(u) - lambeta, 0)
        v = np.sign(v) * np.maximum(np.abs(v) - lambeta, 0)

        # S subproblem: construct Normin2 with adjustments for boundary conditions
        Normin2 = np.concatenate([
            (u[:, -1, :] - u[:, 0, :])[:, np.newaxis],  # Last - First in the row direction
            -np.diff(u, axis=1)
        ], axis=1)

        Normin2 += np.concatenate([
            (v[-1, :, :] - v[0, :, :])[np.newaxis, :, :],  # Last - First in the column direction
            -np.diff(v, axis=0)
        ], axis=0)

        # Update S using Fourier division
        FS = (Normin1 + beta * np.fft.fft2(Normin2)) / Denormin
        S = np.real(np.fft.ifft2(FS))
        print("S MIN = ", S.min()*255,"S MAX = ", S.max()*255)


        beta *= 2  # Double beta in each iteration

    # Clip S to ensure values stay within [0, 1] before scaling
    # S = np.clip(S, 0, 1)
    T = Im - (S * 255.0).astype(np.uint8)

    # Ensure T is also within [0, 255] by clipping
    # T = np.clip(T, 0, 255).astype(np.uint8)

    return T, (S * 255.0).astype(np.uint8)

def TV_L2_Decomp(Im, lambda_val=2e-2):
    S = Im.astype(np.float64) / 255.0
    betamax = 1e5
    fx = np.array([[1, -1]])
    fy = np.array([[1], [-1]])
    N, M, D = S.shape
    otfFx = psf2otf(fx, (N, M))
    otfFy = psf2otf(fy, (N, M))
    Normin1 = np.fft.fft2(S)
    Denormin2 = np.abs(otfFx)**2 + np.abs(otfFy)**2
    if D > 1:
        Denormin2 = np.repeat(Denormin2[:, :, np.newaxis], D, axis=2)
    beta = 2 * lambda_val
    while beta < betamax:
        lambeta = lambda_val / beta
        Denormin = 1 + beta * Denormin2
        # h-v subproblem
        u = np.diff(S, axis=1, append=S[:, :1, :])
        v = np.diff(S, axis=0, append=S[:1, :, :])
        u = np.sign(u) * np.maximum(np.abs(u) - lambeta, 0)
        v = np.sign(v) * np.maximum(np.abs(v) - lambeta, 0)
        # S subproblem
        Normin2 = np.concatenate([u[:, -1:, :] - u[:, :1, :], -np.diff(u, axis=1)], axis=1)
        Normin2 += np.concatenate([v[-1:, :, :] - v[:1, :, :], -np.diff(v, axis=0)], axis=0)
        FS = (Normin1 + beta * np.fft.fft2(Normin2)) / Denormin
        S = np.real(np.fft.ifft2(FS))
        beta *= 2
        print('.', end='')
    T = Im - (S * 255.0).astype(np.uint8)
    print()
    return T, (S * 255.0).astype(np.uint8)


y_text, y_struct = TV_L2_Decomp(y, 0.05)
y_struct = y - y_text
#print

# Display the results
plt.figure(figsize=(10, 5))
plt.imshow(np.hstack([y_text*10+0.5 , y_struct*10p]))
plt.title('The decomposed texture & structure layers')
plt.axis('off')
plt.show()
