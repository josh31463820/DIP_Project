from scipy.fft import fft2, ifft2
import numpy as np






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
