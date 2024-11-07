import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as splinalg
from skimage.color import rgb2gray
from skimage import img_as_float
# from scipy.sparse.linalg import spilu, LinearOperator
from scipy.sparse import diags


def generate_refined_mask(I_structure_layer, initial_mask, alpha=0.1, epsilon=1e-7, win_size=1, tol=1e-3, maxiter=5000):
    """
    Generate a refined mask using the matting Laplacian method.

    Parameters:
    - I_structure_layer: 2D grayscale structure layer (or converted grayscale image) as input for matting.
    - initial_mask: 2D initial mask estimate (binary or grayscale).
    - alpha: Regularization parameter for balancing initial mask and structure alignment.
    - epsilon: Small constant for stability in Laplacian computation.
    - win_size: Window size for the matting Laplacian (1 = 3x3 window).
    - tol: Tolerance for the GMRES solver.
    - maxiter: Maximum iterations for the GMRES solver.

    Returns:
    - refined_mask: 2D refined mask aligned with structure layer.
    """

    def compute_matting_laplacian(I, epsilon=1e-7, win_size=1):
        """Computes the matting Laplacian matrix for soft matting."""
        H, W = I.shape
        window_area = (2 * win_size + 1) ** 2
        win_pixels = window_area
        L = sp.lil_matrix((H * W, H * W))

        # Traverse each pixel in the image
        for y in range(win_size, H - win_size):
            for x in range(win_size, W - win_size):
                # Extract local window
                window = I[y - win_size:y + win_size + 1, x - win_size:x + win_size + 1]
                window = window.flatten()
                mean = np.mean(window)
                cov = np.cov(window) + (epsilon / win_pixels) * np.eye(win_pixels)

                # Compute the inverse covariance
                cov_inv = np.linalg.inv(cov)

                # Calculate indices
                indices = []
                for dy in range(-win_size, win_size + 1):
                    for dx in range(-win_size, win_size + 1):
                        indices.append((y + dy) * W + (x + dx))

                # Populate L
                for i in range(win_pixels):
                    for j in range(win_pixels):
                        diff_i = window[i] - mean
                        diff_j = window[j] - mean
                        val = (1 + diff_i * (cov_inv[i, j] * diff_j)) / win_pixels
                        L[indices[i], indices[j]] -= val

        return L.tocsr()

    # Step 1: Ensure structure layer is grayscale and float
    I_S = img_as_float(I_structure_layer) if len(I_structure_layer.shape) == 3 else I_structure_layer
    if I_S.ndim == 3:
        I_S = rgb2gray(I_S)

    # Step 2: Compute the matting Laplacian matrix L_s
    L_s = compute_matting_laplacian(I_S, epsilon=epsilon, win_size=win_size)

    # Step 3: Set up the linear system A * m = b
    H, W = I_S.shape
    A_sparse = sp.eye(H * W) + alpha * L_s  # Matrix A in sparse form
    m_hat = initial_mask.flatten()  # Flatten the initial mask to match the vector form



    # Step 4: Solve the linear system using GMRES with preconditioner
    b = m_hat
    m, exit_code = splinalg.gmres(A_sparse, b, tol=tol, maxiter=maxiter)

    if exit_code != 0:
        print(f"GMRES did not converge, exit code: {exit_code}")

    # Step 5: Reshape the result back into the image shape
    refined_mask = m.reshape(H, W)

    return refined_mask


