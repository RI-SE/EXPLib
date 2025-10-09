import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import torch
import torchvision
import torchvision.transforms as T
from typing import Callable, Optional, Tuple
from shap.plots import colors

from skimage.color import rgb2gray
from skimage.filters import sobel
from skimage.segmentation import slic, watershed, mark_boundaries
import time
from PIL import Image


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

def _make_detection_wrapper(
    model: torch.nn.Module, device: torch.device, size: Tuple[int, int] = (320, 320)
) -> Callable[[Image.Image], float]:
    """
    Convert a generic detection model into the simple
    ``detection_model(image: PIL) -> float`` callable that
    ``compute_shap()`` expects.

    Returns
    -------
    Callable[[PIL.Image], float]
        A function that returns the *highest* detection confidence
        for a given image.
    """
    transform = T.Compose([T.Resize(size), T.ToTensor()])

    def wrapper(image: Image.Image) -> float:
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        img_tensor = transform(image).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img_tensor)
        # ``outputs`` is a list of dicts – use the first element
        scores = outputs[0]["scores"]
        return float(scores.max()) if len(scores) > 0 else 0.0

    return wrapper


def load_image(img_path):
    image_pil = Image.open(img_path).convert("RGB")  
    return image_pil, np.array(image_pil)  # Return both PIL and NumPy formats


# Apply SLIC segmentation
def segment_image(image_np, num_segments=50):
    return slic(image_np, n_segments=num_segments, compactness=10, sigma=1, start_label=1)

# Compute SHAP values for image regions
def compute_shap_values(
    image_np: np.ndarray,
    segments: np.ndarray,
    baseline_score: float,
    detection_model: Callable[[Image.Image], float],
    num_segments: Optional[int] = 50,
    num_samples: int = 200
) -> np.ndarray:
    """
    Compute SHAP (SHapley Additive exPlanations) values for image regions.
    
    Args:
        image_np (np.ndarray): Input image as a numpy array.
        segments (np.ndarray): Segmentation mask with integers [0, num_segments-1].
        baseline_score (float): Base detection score for the original image.
        detection_model (Callable): Function that takes a PIL Image and returns a detection score.
        num_segments (Optional[int]): Number of segments in the image. If None, computed from segments.
        num_samples (int): Number of random samples to use for SHAP computation.
    
    Returns:
        np.ndarray: SHAP values for each segment.
        
    Raises:
        ValueError: If input parameters are invalid.
        TypeError: If input types are incorrect.
    """
    # Input validation
    if not isinstance(image_np, np.ndarray):
        raise TypeError("image_np must be a numpy array")
    if not isinstance(segments, np.ndarray):
        raise TypeError("segments must be a numpy array")
    if image_np.shape[:2] != segments.shape:
        raise ValueError("image and segments shapes must match")
    
    # If num_segments is None, compute from segments
    if num_segments is None:
        num_segments = len(np.unique(segments))
    else:
        actual_segments = len(np.unique(segments))
        if actual_segments > num_segments:
            raise ValueError(f"num_segments ({num_segments}) is less than actual number of segments ({actual_segments})")
    
    # Initialize SHAP values and contribution counter
    shap_values = np.zeros(num_segments)
    contribution_count = np.zeros(num_segments)
    
    try:
        for _ in range(num_samples):
            # Create random mask (coalition)
            mask = np.random.choice([0, 1], size=num_segments, p=[0.5, 0.5])
            perturbed_image = np.copy(image_np)

            # Apply masking with vectorized operations where possible
            for i in range(num_segments):
                if mask[i] == 0:
                    segment_mask = (segments == i)
                    if segment_mask.any():  # Check if segment exists
                        # Compute mean color for each channel
                        mean_color = np.mean(image_np[segment_mask], axis=0)
                        perturbed_image[segment_mask] = mean_color

            # Compute new score
            try:
                new_score = detection_model(Image.fromarray(perturbed_image.astype('uint8')))
            except Exception as e:
                print(f"Warning: Detection model failed for sample {_}: {str(e)}")
                continue

            # Update SHAP values and count contributions
            score_diff = new_score - baseline_score
            shap_values += score_diff * mask
            contribution_count += mask

        # Normalize SHAP values by actual contribution counts
        # Add small epsilon to avoid division by zero
        epsilon = 1e-10
        normalized_shap = shap_values / (contribution_count + epsilon)
        
        return normalized_shap

    except Exception as e:
        raise RuntimeError(f"Error computing SHAP values: {str(e)}")
    

def compute_shap(img_path, detection_model, num_segments=30):
    """
    Compute a SHAP heatmap for a single image.

    Parameters
    ----------
    img_path : str
        Path to the image file.
    detection_model : callable
        A callable that accepts a PIL image and returns the model score
    num_segments : int, default 30
        Number of super-pixel segments used for SHAP estimation.

    Returns
    -------
    shap_map : np.ndarray
        Normalised heatmap with the same shape as the input image.
    image_np : np.ndarray
        The original image as a NumPy array
    """
    image_pil, image_np = load_image(img_path)

    baseline_score = detection_model(image_pil)

    segments = segment_image(image_np, num_segments=num_segments)

    shap_values = compute_shap_values(
        image_np, segments, baseline_score, detection_model
    )

    shap_map = np.zeros_like(image_np, dtype=np.float32)
    for seg_idx in range(len(shap_values)):
        shap_map[segments == seg_idx] = shap_values[seg_idx]

    shap_min, shap_max = shap_map.min(), shap_map.max()
    shap_range = shap_max - shap_min
    if shap_range > 0:
        shap_map = (shap_map - shap_min) / shap_range
    else:
        shap_map = np.zeros_like(shap_map)

    return shap_map, image_np


def plot_shap(image_np, shap_map):
    shap_map = shap_map[:,:,1]

    fig, ax = plt.subplots(1, 2, figsize=(15, 8))

    # Plot the original image
    ax[0].imshow(image_np)
    ax[0].set_title("Original Image")
    ax[0].axis("off")

    # Overlay the SHAP heatmap on the original image
    ax[1].imshow(shap_map, cmap=colors.red_blue, alpha=0.5)  # Use 'plasma' for better color contrast
    ax[1].set_title("SHAP Heatmap")
    ax[1].axis("off")

    plt.tight_layout()
    plt.show()
