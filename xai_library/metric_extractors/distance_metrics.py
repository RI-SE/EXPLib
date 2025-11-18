import numpy as np
from scipy.stats import norm, entropy

# Euclidean distance between a point and the mean of a distribution
def euclidean_distance(point, distribution):
    """
    Compute Euclidean distance between a point and the mean of a distribution.
    """
    mean = np.mean(distribution)
    return np.abs(point - mean)  # sqrt((x - mean)^2) simplifies to abs(x - mean)

# Mahalanobis distance between a point and a distribution
def mahalanobis_distance(point, distribution, covariance=None):
    """
    Compute Mahalanobis distance between a point and a distribution.
    If covariance is not provided, compute from the distribution.
    """
    mean = np.mean(distribution)
    diff = np.array(point - mean)
    
    if covariance is None:
        covariance = np.cov(distribution, rowvar=False)
    
    inv_covariance = np.linalg.inv(covariance)
    return np.sqrt(np.dot(np.dot(diff.T, inv_covariance), diff))

# KL Divergence between two normal distributions
def kl_divergence(mean_p, var_p, mean_q, var_q):
    """
    Compute KL divergence between two normal distributions P and Q.
    P ~ N(mean_p, var_p), Q ~ N(mean_q, var_q)
    """
    return 0.5 * (np.log(var_q / var_p) + (var_p + (mean_p - mean_q)**2) / var_q - 1)

# Jensen-Shannon Divergence between two distributions
def js_divergence(p, q):
    """
    Compute Jensen-Shannon divergence between two probability distributions p and q.
    Both p and q should be arrays that sum to 1.
    """
    p = np.asarray(p, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    
    # Normalize to ensure valid probability distributions
    p /= p.sum()
    q /= q.sum()
    
    m = 0.5 * (p + q)
    return 0.5 * (entropy(p, m, base=2) + entropy(q, m, base=2))

