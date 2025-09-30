import numpy as np
import matplotlib.pyplot as plt

import torchvision.transforms as T
from typing import Callable, Optional

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic, watershed, mark_boundaries
import sklearn.linear_model
import os
import time
from PIL import Image
import torch
import torchvision    


def load_model(model_name, weight_path, device):



    # Model definition (get model from torchvision library)
    #name = "fasterrcnn_mobilenet_v3_large_320_fpn" #[tm_v1]"ssdlite320_mobilenet_v3_large_2024Oct" #[tm_v2]fasterrcnn_mobilenet_v3_large_320_fpn
    model = torchvision.models.get_model(model_name,
                                        weights=None,
                                        weights_backbone=None,
                                        num_classes=2)

    # Load the trained model
    checkpoint = torch.load(weight_path)
    model.load_state_dict(checkpoint)
    model.to(device)
    model.eval()
    return model

def detection_model(image, model, device):
    
    """Runs object detection and returns the max confidence score."""
    # Define the transformation for inputs
    transform = T.Compose([
            T.Resize((320, 320)),
            T.ToTensor(),
    ])
    # Ensure image is in PIL format
    if isinstance(image, np.ndarray):
        image = Image.fromarray(image)  

    # Apply transformations
    image = transform(image).unsqueeze(0).to(device)

    # Perform inference
    with torch.no_grad():
        outputs = model(image)
    
    # Extract highest detection score
    if len(outputs[0]["scores"]) > 0:
        result = outputs[0]["scores"].max().unsqueeze(0)
    else:
        result = torch.tensor([0.0], device=device)

    return result.cpu().numpy()[0]  # Convert tensor to scalar NumPy value


def _make_detection_callable(model: torch.nn.Module,
                             device: torch.device) -> Callable[[Image.Image], float]:
    """
    Returns a *zero‑argument* callable that can be passed to `compute_lime_values`.

    The callable closes over *model* and *device* and simply forwards the PIL image.
    """
    def _call(img: Image.Image) -> float:
        return detection_model(img, model, device)
    return _call


def load_image(img_path):
    image_pil = Image.open(img_path).convert("RGB")  
    return image_pil, np.array(image_pil)  # Return both PIL and NumPy formats

def test_slic(img_path, n_segment=50):
    img = Image.open(img_path)

    img_np = np.array(img.convert("RGB"))
    start_time = time.time()
    segments_slic = slic(img_np, n_segments=n_segment, compactness=10, sigma=1, start_label=1)
    print("segments_slic--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()
    gradient = sobel(rgb2gray(img_np))
    segments_watershed = watershed(gradient, markers=n_segment, compactness=0.001)
    print("segments_watershed--- %s seconds ---" % (time.time() - start_time))
    start_time = time.time()

    print(f'Watershed number of segments: {len(np.unique(segments_watershed))}')

    plt.imshow(mark_boundaries(img_np, segments_watershed))
    plt.title('Compact watershed')
    plt.show()

    plt.imshow(mark_boundaries(img_np, segments_slic))
    plt.title('SLIC')
    plt.show()

def segment_image(image_np, num_segments=50, is_slic=True):
    if is_slic:
        result = slic(image_np, n_segments=num_segments, compactness=10, sigma=1, start_label=1)
    else:
        gradient = sobel(rgb2gray(image_np))
        result = watershed(gradient, markers=num_segments, compactness=0.001)
    return result

def compute_lime_values(
    image_np: np.ndarray,
    segments: np.ndarray,
    detection_model: Callable[[Image.Image], float],
    num_segments: Optional[int] = 50,
    num_samples: int = 200,
    kernel_width: float = 0.25
) -> np.ndarray:
    """
    Compute LIME (Local Interpretable Model-agnostic Explanations) values for image regions.

    Args:
        image_np (np.ndarray): Input image as a numpy array.
        segments (np.ndarray): Segmentation mask with integers [0, num_segments-1].
        baseline_score (float): Base detection score for the original image.
        detection_model (Callable): Function that takes a PIL Image and returns a detection score.
        num_segments (Optional[int]): Number of segments in the image. If None, computed from segments.
        num_samples (int): Number of perturbed samples used for LIME computation.
        kernel_width (float): Width of the exponential kernel used for weighting samples.

    Returns:
        np.ndarray: LIME values for each segment.

    Raises:
        ValueError: If input parameters are invalid.
        TypeError: If input types are incorrect.
    """
    torch.cuda.manual_seed_all(42)
    # Input validation
    if not isinstance(image_np, np.ndarray):
        raise TypeError("image_np must be a numpy array")
    if not isinstance(segments, np.ndarray):
        raise TypeError("segments must be a numpy array")
    if image_np.shape[:2] != segments.shape:
        raise ValueError("image and segments shapes must match")

    # Determine the number of unique segments if not provided
    if num_segments is None:
        num_segments = len(np.unique(segments))
    
    # Data storage
    feature_matrix = np.zeros((num_samples, num_segments))  # Binary masks for perturbed images
    scores = np.zeros(num_samples)  # Model's predictions on perturbed images
    distances = np.zeros(num_samples)  # Distance of perturbed images from the original

    # Generate perturbed images and get their model scores
    for sample_idx in range(num_samples):
        # Generate a random perturbation mask (0 = remove segment, 1 = keep segment)
        mask = np.random.choice([0, 1], size=num_segments, p=[0.5, 0.5])
        perturbed_image = np.copy(image_np)

        # Apply perturbation (replace removed segments with mean color)
        for i in range(num_segments):
            if mask[i] == 0:
                perturbed_image[segments == i] = np.mean(image_np[segments == i], axis=0)

        # Compute model score
        try:
            new_score = detection_model(Image.fromarray(perturbed_image.astype('uint8')))
        except Exception as e:
            print(f"Warning: Detection model failed for sample {sample_idx}: {str(e)}")
            continue

        # Store data
        feature_matrix[sample_idx] = mask  # Store the segment mask
        scores[sample_idx] = new_score  # Store the model's prediction

        # Compute distance from the original image (Hamming distance in binary space)
        distances[sample_idx] = np.linalg.norm(mask - np.ones(num_segments)) / num_segments

    # Compute sample weights using an exponential kernel
    weights = np.exp(-distances ** 2 / kernel_width ** 2)

    # Fit a weighted linear regression model to estimate feature importance
    try:
        lime_model = sklearn.linear_model.Ridge(alpha=1e-5)
        lime_model.fit(feature_matrix, scores, sample_weight=weights)
        lime_values = lime_model.coef_
    except Exception as e:
        raise RuntimeError(f"Error fitting LIME model: {str(e)}")

    return lime_values

# Compute baseline detection score
def compute_lime(detection_model, img_path, num_segments=50):
    image_pil, image_np = load_image(img_path)
    # Compute LIME values
    segments = segment_image(image_np, num_segments=num_segments, is_slic=False)
    lime_values = compute_lime_values(image_np, segments, detection_model)

    # Create LIME heatmap
    lime_map = np.zeros_like(image_np, dtype=np.float32)
    for i in range(len(lime_values)):
        lime_map[segments == i] = lime_values[i]

    lime_map_min = lime_map.min()
    lime_map_max = lime_map.max()
    lime_map_range = lime_map_max - lime_map_min
        
    if lime_map_range > 0:
        lime_map = (lime_map - lime_map_min) / lime_map_range
    else:
        # If all values are the same, set lime_map to zeros or a constant
        lime_map = np.zeros_like(lime_map)
    return lime_map, image_np

def plot_lime(image_np, lime_map):
    # Normalize SHAP values (if not done yet)
    lime_map = lime_map[:,:,0]

    # Create a figure and use gridspec to manage layout
    fig, ax = plt.subplots(1, 2, figsize=(15, 8))

    # Plot the original image
    ax[0].imshow(image_np)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Overlay the LIME heatmap on the original image
    ax[1].imshow(lime_map, cmap="plasma", alpha=0.5)  # Use 'plasma' for better color contrast
    ax[1].set_title("LIME Heatmap")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
