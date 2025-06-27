import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lsqr
from typing import Tuple

def create_sparse_system_matrix(shape: Tuple[int, int], degradation_sigma: float = 0.1) -> sparse.csr_matrix:
    """Create a sparse system matrix instead of dense"""
    n = shape[0] * shape[1]
    
    # Create sparse identity matrix
    main_diag = sparse.eye(n, format='csr')
    
    # Add sparse random perturbation
    nnz_per_row = 5  # Number of non-zero elements per row
    indices = np.random.randint(0, n, size=(n, nnz_per_row))
    data = np.random.normal(0, degradation_sigma, size=(n, nnz_per_row))
    
    rows = np.repeat(np.arange(n), nnz_per_row)
    cols = indices.flatten()
    data = data.flatten()
    
    perturbation = sparse.csr_matrix((data, (rows, cols)), shape=(n, n))
    
    return main_diag + perturbation

def reconstruct_image(degraded_image: np.ndarray, regularization_param: float) -> np.ndarray:
    """
    Perform constrained image reconstruction with given regularization parameter
    using sparse matrices and chunked processing
    """
    # Convert image to float32 for memory efficiency
    degraded_image = degraded_image.astype(np.float32)
    
    # Create sparse system matrix
    A = create_sparse_system_matrix(degraded_image.shape)
    
    # Reshape image to 1D array
    b = degraded_image.flatten()
    
    # Create sparse regularization matrix
    reg_matrix = regularization_param * sparse.eye(A.shape[1], format='csr')
    
    # Combine matrices efficiently
    A_reg = sparse.vstack([A, reg_matrix])
    b_reg = np.concatenate([b, np.zeros(A.shape[1], dtype=np.float32)])
    
    # Solve the regularized least squares problem
    x, istop, itn, r1norm = lsqr(A_reg, b_reg, atol=1e-6, btol=1e-6)[:4]
    
    # Reshape result back to image dimensions
    reconstructed = x.reshape(degraded_image.shape)
    
    # Ensure non-negative values and convert back to uint8 for image display
    reconstructed = np.clip(reconstructed, 0, 255).astype(np.uint8)
    
    return reconstructed