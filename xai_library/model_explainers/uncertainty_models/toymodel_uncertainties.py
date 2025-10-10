import torch
import torch.nn as nn
import torchvision
from typing import List, Dict, Optional, Tuple
from collections import OrderedDict
import cv2
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import seaborn as sns

class SSDWithDropout(nn.Module):
    def __init__(self, base_ssd, dropout_prob=0.3):
        super(SSDWithDropout, self).__init__()
        self.base_ssd = base_ssd
        self.dropout = nn.Dropout(p=dropout_prob)

    def forward(
        self, images: List[torch.Tensor], targets: Optional[List[Dict[str, torch.Tensor]]] = None
    ) -> Tuple[Dict[str, torch.Tensor], List[Dict[str, torch.Tensor]]]:
        
        if self.training:
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for target in targets:
                    boxes = target["boxes"]
                    if isinstance(boxes, torch.Tensor):
                        torch._assert(
                            len(boxes.shape) == 2 and boxes.shape[-1] == 4,
                            f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.",
                        )
                    else:
                        torch._assert(False, f"Expected target boxes to be of type Tensor, got {type(boxes)}.")
        
        # get the original image sizes
        original_image_sizes: List[Tuple[int, int]] = []
        for img in images:
            val = img.shape[-2:]
            torch._assert(
                len(val) == 2,
                f"expecting the last two dimensions of the Tensor to be H and W instead got {img.shape[-2:]}",
            )
            original_image_sizes.append((val[0], val[1]))

        # transform the input
        images, targets = self.base_ssd.transform(images, targets)

        # Check for degenerate boxes
        if targets is not None:
            for target_idx, target in enumerate(targets):
                boxes = target["boxes"]
                degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
                if degenerate_boxes.any():
                    bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                    degen_bb: List[float] = boxes[bb_idx].tolist()
                    torch._assert(
                        False,
                        "All bounding boxes should have positive height and width."
                        f" Found invalid box {degen_bb} for target at index {target_idx}.",
                    )

        # get the features from the backbone
        features = self.base_ssd.backbone(images.tensors)
        if isinstance(features, torch.Tensor):
            features = OrderedDict([("0", features)])
        
        features = [self.dropout(f) for f in list(features.values())]

        # compute the ssd heads outputs using the features
        head_outputs = self.base_ssd.head(features)

        # create the set of anchors
        anchors = self.base_ssd.anchor_generator(images, features)

        losses = {}
        detections: List[Dict[str, torch.Tensor]] = []
        if self.training:
            matched_idxs = []
            if targets is None:
                torch._assert(False, "targets should not be none when in training mode")
            else:
                for anchors_per_image, targets_per_image in zip(anchors, targets):
                    if targets_per_image["boxes"].numel() == 0:
                        matched_idxs.append(
                            torch.full(
                                (anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device
                            )
                        )
                        continue

                    match_quality_matrix = torchvision.ops.box_iou(targets_per_image["boxes"], anchors_per_image)
                    matched_idxs.append(self.base_ssd.proposal_matcher(match_quality_matrix))

                losses = self.base_ssd.compute_loss(targets, head_outputs, anchors, matched_idxs)
        else:
            detections = self.base_ssd.postprocess_detections(head_outputs, anchors, images.image_sizes)
            detections = self.base_ssd.transform.postprocess(detections, images.image_sizes, original_image_sizes)

        if torch.jit.is_scripting():
            if not self._has_warned:
                warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
                self._has_warned = True
            return losses, detections
        return self.base_ssd.eager_outputs(losses, detections)

def predict_with_uncertainty(model, inputs, n_iter=10):
    all_outputs = []
    model.eval()
    for m in model.modules():
        if m.__class__.__name__.startswith('Dropout'):
            m.train()
    for _ in range(n_iter):
        with torch.no_grad():
            outputs = model(inputs)
            all_outputs.append(outputs)

    model.eval()   # Set back to evaluation mode
    
    return all_outputs

def calculate_uncertainties(predictions):
    uncertainties = []
    
    for pred in predictions:
        boxes_mean = pred[0]['boxes'].mean(dim=0)
        boxes_std = pred[0]['boxes'].std(dim=0)
        
        scores_mean = pred[0]['scores'].mean(dim=0)
        scores_std = pred[0]['scores'].std(dim=0)
        
        uncertainties.append({
            'boxes_mean': boxes_mean,
            'boxes_std': boxes_std,
            'scores_mean': scores_mean,
            'scores_std': scores_std
        })
        
    return uncertainties

def rotate_image(image, angle):
    # Get image dimensions
    height, width = image.shape[:2]
    
    # Compute the rotation matrix
    rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1)
    
    # Compute the rotated image
    rotated_image = cv2.warpAffine(image, rotation_matrix, (width, height), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    
    return rotated_image, rotation_matrix

def compute_statistics_and_plot(outputs):
    if isinstance(outputs, torch.Tensor):
        outputs = outputs.cpu().detach().numpy()
    # Extract bounding box coordinates
    bounding_boxes = [output[0]['boxes'][0] for output in outputs]

    # Compute mean bounding box
    mean_bbox = np.mean(bounding_boxes, axis=0)

    # Compute mean center
    mean_center = [(mean_bbox[0] + mean_bbox[2]) / 2, (mean_bbox[1] + mean_bbox[3]) / 2]

    # Compute deviation per edges
    deviation_edges = np.std(np.array(bounding_boxes)[:, [2, 3]] - np.array(bounding_boxes)[:, [0, 1]], axis=0)

    # Compute deviation per x and y
    deviation_x = np.std(np.array(bounding_boxes)[:, [0, 2]], axis=0)
    deviation_y = np.std(np.array(bounding_boxes)[:, [1, 3]], axis=0)

    # Plot the mean bounding box and deviations
    plt.figure(figsize=(8, 6))
    plt.plot([mean_bbox[0], mean_bbox[2], mean_bbox[2], mean_bbox[0], mean_bbox[0]],
             [mean_bbox[1], mean_bbox[1], mean_bbox[3], mean_bbox[3], mean_bbox[1]],
             label='Mean Bounding Box', color='blue')

    plt.errorbar(mean_center[0], mean_center[1],
                 xerr=[[deviation_x[0]], [deviation_x[1]]],
                 yerr=[[deviation_y[0]], [deviation_y[1]]],
                 fmt='o', color='red', label='Deviation per x, y')

    plt.errorbar(mean_bbox[0], mean_bbox[1],
                 xerr=deviation_edges[0], yerr=deviation_edges[1],
                 fmt='o', color='green', label='Deviation per edges')

    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('Mean Bounding Box with Deviations')
    plt.legend()
    plt.grid(True)
    plt.show()
def convert_output_to_cpu(outputs):
    for output in outputs:
        for item in output:
            for key, value in item.items():
                if isinstance(value, torch.Tensor):
                    item[key] = value.cpu()
    return outputs


def setup_parallel_models(base_model, n_models):
    import copy
    """
    Creates multiple models for parallel Toymodel MC

    Args:
        base_model: The base PyTorch model to replicate.
        n_models: The number of models to create.
    
    Returns:
        A list of models
    """
    
    models = []
    for i in range(n_models):
        model_copy = copy.deepcopy(base_model)

        model_copy.eval()
        for m in model_copy.modules():
            if m.__class__.__name__.startswith('Dropout'):
                m.train()
        
        models.append(model_copy)

    return models

def predict_with_parallel_models(models, inputs, n_iter=10):
    import numpy as np
    from concurrent.futures import ThreadPoolExecutor
    """
    Performs uncertainty predictions using a list of prepared models in parallel,
    
    Args:
        models: A list of created parallel models 
        inputs: The inputs to the models (image tensor)
        n_iter: Number of stochastic forward passes for each model.
    
    Returns:
        A dictionary containing:
            - 'bbox_mean': The mean of predicted bboxes
            - 'bbox_std': The standard deviation of bboxes
            - 'bbox_CI95': A tuple containing the lower and upper 95% CI bounds of bboxes
    """
    def single_model_pass(model):
        with torch.no_grad():
            return [model(inputs) for _ in range(n_iter)]

    # Run  models in parallel
    with ThreadPoolExecutor() as executor:
        all_outputs = list(executor.map(single_model_pass, models))

    all_boxes = [ # Thanh fixed this for case where not exactly one bbox is found (20241229)
        torch.cat([output['boxes'][0] for output in iteration_outputs if 'boxes' in output and output['boxes'].numel() > 0], dim=0)
        for model_outputs in all_outputs
        for iteration_outputs in model_outputs
        if any('boxes' in output and output['boxes'].numel() > 0 for output in iteration_outputs)
    ]

    all_boxes_tensor = torch.stack(all_boxes, dim=0)  

    bbox_mean = torch.mean(all_boxes_tensor, dim=0)  
    bbox_std = torch.std(all_boxes_tensor, dim=0)    

    z_score = 1.96  # For 95% CI
    CI95 = z_score * (bbox_std)
    device = bbox_mean.device  # Get the device of bbox_mean
    CI95_adjustments = torch.tensor([CI95[0], CI95[1], -CI95[2], -CI95[3]], device=device)

    bbox_CI95_lower = bbox_mean + CI95_adjustments
    bbox_CI95_upper = bbox_mean - CI95_adjustments

    return {
        'bbox_mean': bbox_mean,  # Mean bboxes
        'bbox_std': bbox_std,  # Std of bboxes
        'bbox_all': all_boxes_tensor,
        'bbox_CI95': (bbox_CI95_lower, bbox_CI95_upper)  # Tensors with lower and upper CI bounds
    }


def plot_image_with_bboxes(image_tensor, bbox_CI95, bbox_mean):

    image = image_tensor[0, 0].numpy()
    plt.figure(figsize=(6, 6))
    plt.imshow(image, cmap='gray')
    plt.axis('off')
    
    ci_lower, ci_upper = bbox_CI95

    rect_ci95_lower = Rectangle(
        (ci_lower[0], ci_lower[1]),  # Top-left corner (x1, y1)
        ci_lower[2] - ci_lower[0],  # Width
        ci_lower[3] - ci_lower[1],  # Height
        linewidth=2, edgecolor='green', facecolor='none', label='CI95_lower'
    )
    plt.gca().add_patch(rect_ci95_lower)
    
    rect_ci95_upper = Rectangle(
        (ci_upper[0], ci_upper[1]),  # Top-left corner (x1, y1)
        ci_upper[2] - ci_upper[0],  # Width
        ci_upper[3] - ci_upper[1],  # Height
        linewidth=2, edgecolor='blue', facecolor='none', label='CI95_upper'
    )
    plt.gca().add_patch(rect_ci95_upper)
    plt.legend()
    plt.show()


# Plot the PDFs for bbox parameters
def plot_bbox_pdfs(all_boxes_tensor):
    params = ['x1', 'y1', 'x2', 'y2']
    plt.figure(figsize=(12, 6))
    
    for i, param in enumerate(params):
        plt.subplot(1, 4, i + 1)
        sns.kdeplot(all_boxes_tensor[:, i].numpy(), fill=True, color='blue')
        plt.title(f'PDF of {param}')
        plt.xlabel(param)
        plt.ylabel('Density')
    
    plt.tight_layout()
    plt.show()

