import numpy as np
from scipy import ndimage
from PIL import Image

def preprocess_image(image: Image.Image, max_size: int = 512) -> np.ndarray:
    """Preprocess image to manageable size and format"""
    # Convert to grayscale if needed
    if image.mode != 'L':
        image = image.convert('L')
    
    # Resize if too large
    if max(image.size) > max_size:
        ratio = max_size / max(image.size)
        new_size = tuple(int(dim * ratio) for dim in image.size)
        image = image.resize(new_size, Image.Resampling.LANCZOS)
    
    return np.array(image)

def extract_quality_metrics(reconstruction: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Extract quality metrics from reconstructed image"""
    metrics = np.zeros(3, dtype=np.float32)
    
    # Ensure same dtype and range
    reconstruction = reconstruction.astype(np.float32)
    reference = reference.astype(np.float32)
    
    # Metric 1: Mean Squared Error
    metrics[0] = np.mean((reconstruction - reference) ** 2)
    
    # Metric 2: Structural similarity (simplified)
    metrics[1] = calculate_structural_similarity(reconstruction, reference)
    
    # Metric 3: Edge preservation
    metrics[2] = calculate_edge_preservation(reconstruction, reference)
    
    return metrics

def calculate_structural_similarity(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate structural similarity between two images"""
    K1, K2 = 0.01, 0.03
    L = 255.0
    
    C1 = (K1 * L) ** 2
    C2 = (K2 * L) ** 2
    
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    sigma1_sq = np.var(img1)
    sigma2_sq = np.var(img2)
    sigma12 = np.mean((img1 - mu1) * (img2 - mu2))
    
    ssim = ((2 * mu1 * mu2 + C1) * (2 * sigma12 + C2)) / \
           ((mu1 ** 2 + mu2 ** 2 + C1) * (sigma1_sq + sigma2_sq + C2))
    
    return float(ssim)

def calculate_edge_preservation(img1: np.ndarray, img2: np.ndarray) -> float:
    """Calculate edge preservation metric"""
    # Use smaller kernel for edge detection
    sobel_x1 = ndimage.sobel(img1, axis=0, mode='reflect')
    sobel_y1 = ndimage.sobel(img1, axis=1, mode='reflect')
    edges1 = np.sqrt(sobel_x1**2 + sobel_y1**2)
    
    sobel_x2 = ndimage.sobel(img2, axis=0, mode='reflect')
    sobel_y2 = ndimage.sobel(img2, axis=1, mode='reflect')
    edges2 = np.sqrt(sobel_x2**2 + sobel_y2**2)
    
    # Calculate correlation efficiently
    correlation = np.corrcoef(edges1.ravel(), edges2.ravel())[0,1]
    return float(correlation)