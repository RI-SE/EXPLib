import cv2
import numpy as np
import joblib
import json
import matplotlib.pyplot as plt
from skimage.feature import corner_harris, corner_peaks, local_binary_pattern
from sklearn.mixture import GaussianMixture


### ======================== Feature Extraction Utilities ======================== ###

def compute_keypoint_stats(keypoints):
    """Compute statistical features from keypoints."""
    if len(keypoints) == 0:
        return [0] * 8  # Return zeroed-out stats if no keypoints
    
    keypoints = np.array(keypoints)
    mean_x, mean_y = np.mean(keypoints, axis=0)
    min_x, min_y = np.min(keypoints, axis=0)
    max_x, max_y = np.max(keypoints, axis=0)
    std_x, std_y = np.std(keypoints, axis=0)
    
    return [mean_x, mean_y, min_x, min_y, max_x, max_y, std_x, std_y]


def extract_keypoints(image_path):
    """Extract keypoints using Harris corner detection."""
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Error: Unable to load image {image_path}")
    
    keypoints = corner_peaks(corner_harris(image), min_distance=5, threshold_rel=0.02)
    return np.array(keypoints, dtype=np.float32)


def extract_gmm_features(keypoints):
    """Fit a GMM model to keypoints and extract statistical features."""
    if len(keypoints) < 3:
        return [len(keypoints), 0, 0, 100, 100, 200, 200]  # Handle edge case (low keypoints)

    keypoints = np.array(keypoints, dtype=np.float32)

    # Determine optimal number of clusters
    n_components_range = range(1, min(4, len(keypoints)) + 1)
    bic_scores = [
        GaussianMixture(n, covariance_type="full", random_state=42).fit(keypoints).bic(keypoints)
        for n in n_components_range
    ]

    # Choose the model with the lowest BIC
    best_n_components = n_components_range[np.argmin(bic_scores)]
    gmm = GaussianMixture(n_components=best_n_components, covariance_type="full", random_state=42)
    labels = gmm.fit_predict(keypoints)

    # Find the largest cluster
    largest_cluster = max(set(labels), key=list(labels).count)
    cluster_keypoints = keypoints[labels == largest_cluster]

    # Compute bounding box
    x_min, y_min = np.min(cluster_keypoints, axis=0)
    x_max, y_max = np.max(cluster_keypoints, axis=0)
    cluster_std = np.std(cluster_keypoints, axis=0) if len(cluster_keypoints) > 1 else [0, 0]

    return [len(cluster_keypoints), cluster_std[0], cluster_std[1], x_min, y_min, x_max, y_max]


### ======================== LBP Feature Extraction ======================== ###

def chi_square_distance(hist1, hist2):
    """Computes Chi-Square distance between two LBP histograms."""
    return 0.5 * np.sum(((hist1 - hist2) ** 2) / (hist1 + hist2 + 1e-10))  # Avoid division by zero


def extract_lbp(image, n_points=8, radius=1):
    """Extracts LBP feature map from an image."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return local_binary_pattern(gray, n_points, radius, method="uniform")


def extract_lbp_patches(image, patch_size=20, n_points=8, radius=1):
    """Extracts LBP histograms for non-overlapping patches."""
    lbp = extract_lbp(image, n_points, radius)
    h, w = lbp.shape
    lbp_patches = []

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = lbp[y:y + patch_size, x:x + patch_size]
            hist, _ = np.histogram(patch.ravel(), bins=np.arange(0, n_points + 3), range=(0, n_points + 2))
            hist = hist.astype(np.float32) / hist.sum()  # Normalize histogram
            lbp_patches.append((x, y, hist))

    return lbp_patches


def match_lbp_patches(image, typical_patterns, threshold=0.2, patch_size=20, display=False):
    """Matches LBP patches in an image to the most common patterns."""
    matched_patches = []
    lbp_patches = extract_lbp_patches(image, patch_size)

    for x, y, hist in lbp_patches:
        best_match = min(typical_patterns, key=lambda ref: chi_square_distance(hist, ref))
        distance = chi_square_distance(hist, best_match)
        
        if distance < threshold:  # Lower is better match
            matched_patches.append((x, y))
    
    if display:
        for (x, y) in matched_patches:
            cv2.circle(image, (x + patch_size//2, y + patch_size//2), radius=5, color=(0, 255, 0), thickness=1)

        # Display results
        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("LBP Patch Matches (Green Circles)")
        plt.axis("off")
        plt.show()
    return matched_patches


def detect_LBP_patch(image_path, top_lbp_patterns, patch_size=20, display=False):
    """Detects and visualizes LBP patches that match common patterns."""
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Error: Unable to load image {image_path}")

    matched_patches = match_lbp_patches(image, top_lbp_patterns, patch_size=patch_size)

    if display:
        for x, y in matched_patches:
            cv2.circle(image, (x + patch_size // 2, y + patch_size // 2), radius=5, color=(0, 255, 0), thickness=1)

        plt.figure(figsize=(12, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("LBP Patch Matches (Green Circles)")
        plt.axis("off")
        plt.show()

    return np.array(matched_patches, dtype=np.float32)


### ======================== Feature Preparation & Prediction ======================== ###

def prepare_features(image_path, top_lbp_patterns):
    """Extracts features from an image, including keypoints, LBP matches, and GMM statistics."""
    matched_patches = detect_LBP_patch(image_path, top_lbp_patterns)
    lbp_gmm = extract_gmm_features(matched_patches)
    keypoints = extract_keypoints(image_path)
    keypoint_gmm = extract_gmm_features(keypoints)
    keypoint_stats = compute_keypoint_stats(keypoints)

    return np.hstack([keypoint_gmm, keypoint_stats, lbp_gmm]).reshape(1, -1)


def predict_bbox(image_path, rf_model, top_lbp_patterns):
    """Predicts the bounding box for an input image using a trained model."""
    features = prepare_features(image_path, top_lbp_patterns)
    print(features)
    predicted_bbox = rf_model.predict(features)[0]
    return predicted_bbox  # Returns [x1, y1, x2, y2]

def overlay_bbox(image_path, predicted_bbox, output_path=None, display=True):
    """
    Overlays the predicted bounding box onto the image.
    
    Args:
        image_path (str): Path to the input image.
        predicted_bbox (list): Predicted bounding box [x1, y1, x2, y2].
        output_path (str, optional): Path to save the output image (if provided).
        display (bool): Whether to display the image with bbox.
    
    Returns:
        None
    """
    # Load image
    image = cv2.imread(image_path)
    
    if image is None:
        print(f"Error: Unable to load image {image_path}")
        return
    
    # Unpack bbox coordinates
    x1, y1, x2, y2 = map(int, predicted_bbox)  # Ensure integer values
    
    # Draw bounding box
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box

    # Save the image if output path is provided
    if output_path:
        cv2.imwrite(output_path, image)
        print(f"Saved image with bbox at {output_path}")

    # Display the image with bounding box
    if display:
        plt.figure(figsize=(8, 6))
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title("Predicted Bounding Box")
        plt.axis("off")
        plt.show()