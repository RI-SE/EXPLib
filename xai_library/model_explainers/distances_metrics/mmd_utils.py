import os
from typing import List, Dict, Optional, Union

import torch
import torchvision
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm


# -------------------- Core MMD Functions --------------------
def gaussian_kernel(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute Gaussian kernel between x and y.
    k(x, y) = exp(-(x-y)^2 / (2*sigma^2))
    Shapes: x: [n, d], y: [m, d]
    """
    # Ensure float32 for numerical stability and performance
    x = x.float()
    y = y.float()
    x_norm = (x ** 2).sum(dim=1, keepdim=True)  # [n, 1]
    y_norm = (y ** 2).sum(dim=1, keepdim=True).transpose(0, 1)  # [1, m]
    dist_matrix = x_norm + y_norm - 2.0 * (x @ y.T)             # [n, m]
    return torch.exp(-dist_matrix / (2.0 * (sigma ** 2)))


def compute_mmd(x: torch.Tensor, y: torch.Tensor, sigma: float = 1.0) -> torch.Tensor:
    """
    Compute Maximum Mean Discrepancy (MMD) between two sets of features.
    """
    xx = gaussian_kernel(x, x, sigma)
    yy = gaussian_kernel(y, y, sigma)
    xy = gaussian_kernel(x, y, sigma)
    return xx.mean() + yy.mean() - 2.0 * xy.mean()


def estimate_sigma_median_heuristic(x: torch.Tensor, y: torch.Tensor, subsample: int = 1000) -> float:
    """
    Median heuristic for RBF bandwidth: median of pairwise distances.
    Helpful when feature scales vary.

    If subsample is large, it randomly samples to reduce O(n^2).
    """
    with torch.no_grad():
        X = x
        Y = y
        if subsample is not None:
            if X.size(0) > subsample:
                X = X[torch.randperm(X.size(0))[:subsample]]
            if Y.size(0) > subsample:
                Y = Y[torch.randperm(Y.size(0))[:subsample]]
        # Compute pairwise |x - y|^2
        x_norm = (X ** 2).sum(dim=1, keepdim=True)
        y_norm = (Y ** 2).sum(dim=1, keepdim=True).transpose(0, 1)
        d2 = x_norm + y_norm - 2.0 * (X @ Y.T)
        # Convert to distances (sqrt), take median (avoid zeros)
        d = torch.sqrt(torch.clamp(d2, min=1e-12)).reshape(-1)
        med = torch.median(d).item()
        # Avoid sigma=0
        return max(med, 1e-6)


# -------------------- Hooks --------------------
def get_activation(activations: Dict[str, torch.Tensor], name: str):
    """
    Hook function to capture layer activations (flattened).
    """
    def hook(module, input, output):
        # Some modules may output tuples; handle tensors only
        if isinstance(output, torch.Tensor):
            activations[name] = output.detach().view(output.size(0), -1).cpu()
    return hook


def register_hooks(model: torch.nn.Module,
                   layer_types: Optional[Union[type, List[type]]] = torch.nn.Conv2d) -> Dict[str, torch.Tensor]:
    """
    Register forward hooks for layers of specific types.

    layer_types can be a single type or a list of types, e.g.,
    [torch.nn.Conv2d, torch.nn.Linear, torch.nn.BatchNorm2d]
    """
    if isinstance(layer_types, type):
        layer_types = [layer_types]
    activations: Dict[str, torch.Tensor] = {}
    for name, layer in model.named_modules():
        if any(isinstance(layer, t) for t in layer_types):
            layer.register_forward_hook(get_activation(activations, name))
    return activations


# -------------------- Image Loading --------------------
def preprocess_image(image_path: str, device: torch.device,
                     resize_height: int = 320, resize_width: int = 320,
                     normalize: Optional[Dict[str, List[float]]] = None) -> torch.Tensor:
    """
    Preprocess a single image. Optionally apply normalization (mean/std).
    """
    transforms = [T.Resize((resize_height, resize_width)), T.ToTensor()]
    if normalize is not None:
        transforms.append(T.Normalize(mean=normalize["mean"], std=normalize["std"]))
    transform = T.Compose(transforms)

    image = Image.open(image_path).convert("RGB")
    image = transform(image).unsqueeze(0).to(device, non_blocking=True)
    return image


def load_images_from_folder(folder_path: str, device: torch.device,
                            resize_height: int = 320, resize_width: int = 320,
                            limit: Optional[int] = None,
                            normalize: Optional[Dict[str, List[float]]] = None) -> List[torch.Tensor]:
    """
    Load and preprocess all images from a folder.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Folder does not exist: {folder_path}")

    files = sorted([f for f in os.listdir(folder_path)
                    if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
    if limit is not None:
        files = files[:limit]

    if len(files) == 0:
        raise ValueError(f"No image files found in: {folder_path}")

    images: List[torch.Tensor] = []
    for f in tqdm(files, desc=f"Loading images from {folder_path}"):
        img_tensor = preprocess_image(os.path.join(folder_path, f), device,
                                      resize_height, resize_width, normalize)
        images.append(img_tensor)
    return images


# -------------------- Feature Collection --------------------
def collect_features(model: torch.nn.Module, images: List[torch.Tensor], activations: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Run the model on images to collect activations for layers with hooks.
    """
    layer_activations: Dict[str, List[torch.Tensor]] = {}
    model.eval()
    with torch.no_grad():
        for img in images:
            _ = model(img)  # triggers hooks, fills 'activations' with latest pass
            for layer_name, activation in activations.items():
                layer_activations.setdefault(layer_name, []).append(activation)

    # Concatenate per layer to shape [N_images, feat_dim]
    concatenated: Dict[str, torch.Tensor] = {}
    for layer_name, acts in layer_activations.items():
        concatenated[layer_name] = torch.cat(acts, dim=0)
    return concatenated


def collect_flattened_images(images: List[torch.Tensor]) -> torch.Tensor:
    """
    Flatten all images into vectors (on CPU).
    """
    all_images: List[torch.Tensor] = []
    for img in images:
        img_flat = img.detach().cpu().view(img.size(0), -1)
        all_images.append(img_flat)
    return torch.cat(all_images, dim=0)


# -------------------- MMD Computation --------------------
def compute_layer_mmd(pretrained_features: Dict[str, torch.Tensor],
                      new_features: Dict[str, torch.Tensor],
                      sigma: Optional[float] = 1.0,
                      use_median_heuristic_if_none: bool = True) -> Dict[str, float]:
    """
    Compute MMD distances for each layer. If sigma is None and use_median_heuristic_if_none=True,
    estimate sigma per layer using the median heuristic.
    """
    layer_mmd_distances: Dict[str, float] = {}
    for layer_name in pretrained_features.keys():
        if layer_name not in new_features:
            # Skip if not present in the second feature set
            continue
        X = pretrained_features[layer_name]
        Y = new_features[layer_name]
        if sigma is None and use_median_heuristic_if_none:
            s = estimate_sigma_median_heuristic(X, Y)
        else:
            s = float(sigma)
        mmd_distance = compute_mmd(X, Y, sigma=s)
        layer_mmd_distances[layer_name] = float(mmd_distance.item())
    return layer_mmd_distances


# -------------------- Visualization --------------------
def plot_mmd(layer_mmd_distances: Dict[str, float], exclude_keys: Optional[List[str]] = None,
             title: str = "MMD Distances per Layer"):
    """
    Plot MMD distances per layer.
    """
    if exclude_keys is None:
        exclude_keys = []
    filtered = {k: v for k, v in layer_mmd_distances.items() if not any(ex in k for ex in exclude_keys)}

    layers = list(filtered.keys())
    values = list(filtered.values())

    plt.figure(figsize=(12, 6))
    plt.plot(layers, values, marker="o", color="b", linestyle="-", linewidth=1, markersize=4)
    plt.xticks(rotation=90, fontsize=8)
    plt.xlabel("Layer")
    plt.ylabel("MMD Distance")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()






# -------------------- Main Pipeline --------------------
def run_mmd_analysis(dataset1_path: str, dataset2_path: str,
                     model_name: str, weight_path: str, weight_file: str,
                     num_classes: int, device: Optional[Union[str, torch.device]] = None,
                     sigma: Optional[float] = 1.0, limit: int = 50,
                     resize_height: int = 320, resize_width: int = 320,
                     normalize: Optional[Dict[str, List[float]]] = None,
                     hook_layer_types: Optional[Union[type, List[type]]] = torch.nn.Conv2d):
    """
    Full pipeline: load model, read images, compute MMD for layers and raw images, and plot results.

    Set sigma=None to use the median heuristic per layer.
    """
    device = _resolve_device(device)

    print("Loading model...")
    model = load_and_configure_model(model_name, weight_path, weight_file, num_classes, device=device)

    print("Registering hooks...")
    activations = register_hooks(model, layer_types=hook_layer_types)

    print("Loading images...")
    images1 = load_images_from_folder(dataset1_path, device,
                                      resize_height=resize_height, resize_width=resize_width,
                                      limit=limit, normalize=normalize)
    images2 = load_images_from_folder(dataset2_path, device,
                                      resize_height=resize_height, resize_width=resize_width,
                                      limit=limit, normalize=normalize)

    print("Collecting features...")
    pretrained_features = collect_features(model, images1, activations)
    new_features = collect_features(model, images2, activations)

    print("Computing layer-wise MMD...")
    layer_mmd_distances = compute_layer_mmd(pretrained_features, new_features, sigma=sigma)

    print("Computing raw image MMD...")
    pretrained_images = collect_flattened_images(images1)
    new_images = collect_flattened_images(images2)

    if sigma is None:
        sigma_images = estimate_sigma_median_heuristic(pretrained_images, new_images)
    else:
        sigma_images = float(sigma)

    dataset_mmd_distance = compute_mmd(pretrained_images, new_images, sigma=sigma_images)

    print(f"MMD Distance between datasets (raw images): {dataset_mmd_distance.item():.4f}")

    print("Plotting results...")
    plot_mmd(layer_mmd_distances, title="MMD Distances per Layer")

    return layer_mmd_distances, dataset_mmd_distance