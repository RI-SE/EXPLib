import warnings
from functools import partial
from typing import Any, Callable, Dict, List, Optional, Tuple, OrderedDict

import torch
from torch import nn, Tensor
import torch.nn.functional as F

from torchvision._internally_replaced_utils import load_state_dict_from_url
from torchvision.models import mobilenet
from torchvision.models.detection import _utils as det_utils
from torchvision.models.detection.anchor_utils import DefaultBoxGenerator
from torchvision.ops import boxes as box_ops
from torchvision.models.detection.backbone_utils import _validate_trainable_layers
from torchvision.models.detection.ssd import SSD, SSDScoringHead
from torchvision.models.detection.ssdlite import _prediction_block, _normal_init, _mobilenet_extractor
from torchvision.models.detection.ssdlite import SSDLiteClassificationHead, SSDLiteRegressionHead
from PIL import ImageDraw
import torchvision.transforms as T
import matplotlib.pyplot as plt



class SSDLiteRegressionUncertaintyHead(SSDScoringHead):
    def __init__(self, in_channels: List[int], num_anchors: List[int], norm_layer: Callable[..., nn.Module]):
        uncertainty_reg = nn.ModuleList()
        for channels, anchors in zip(in_channels, num_anchors):
            # Predicting 4 values per anchor (uncertainties)
            uncertainty_reg.append(_prediction_block(channels, 4 * anchors, 3, norm_layer))
        _normal_init(uncertainty_reg)
        super().__init__(uncertainty_reg, 4)  # (4 uncertainty)

    def forward(self, x: Tensor) -> Tensor:
        raw_uncertainty = super().forward(x)  # raw_output includes uncertainty
        return raw_uncertainty


class SSDLiteHeadUncertainty(nn.Module):
    def __init__(
        self, in_channels: List[int], num_anchors: List[int], num_classes: int, norm_layer: Callable[..., nn.Module]
    ):
        super().__init__()
        # Classification head (class probabilities)
        self.classification_head = SSDLiteClassificationHead(in_channels, num_anchors, num_classes, norm_layer)
        # Regression head (bbox coordinates)
        self.regression_head = SSDLiteRegressionHead(in_channels, num_anchors, norm_layer)
        # THANH: Newly added Regression Uncertainty head (bbox uncertainty)
        self.regression_uncertainty_head = SSDLiteRegressionUncertaintyHead(in_channels, num_anchors, norm_layer)

    def forward(self, x: Tensor) -> Dict[str, Tensor]:
        # THANH: Add to Return for both regular outputs and uncertainty output
        return {
            "bbox_regression": self.regression_head(x),
            "cls_logits": self.classification_head(x),
            "bbox_regression_uncertainty": self.regression_uncertainty_head(x),  # Output for uncertainty
        }

def postprocess_detections_with_unc(self, head_outputs: Dict[str, Tensor], image_anchors: List[Tensor], image_shapes: List[Tuple[int, int]]) -> List[Dict[str, Tensor]]:
    # THANH: Modify this function to use within SSD class to output also the uncertainty besides existing output
    bbox_regression = head_outputs["bbox_regression"]
    bbox_regression_uncertainty = head_outputs["bbox_regression_uncertainty"]
    pred_scores = F.softmax(head_outputs["cls_logits"], dim=-1)

    num_classes = pred_scores.size(-1)
    device = pred_scores.device

    detections: List[Dict[str, Tensor]] = []

    for boxes, scores, anchors, image_shape, bbox_unc in zip(
        bbox_regression, pred_scores, image_anchors, image_shapes, bbox_regression_uncertainty
    ):
        # Decode and clip boxes
        boxes = self.box_coder.decode_single(boxes, anchors)
        boxes = box_ops.clip_boxes_to_image(boxes, image_shape)

        image_boxes = []
        image_scores = []
        image_labels = []
        image_uncertainty = []  # To store bbox_regression_uncertainty

        for label in range(1, num_classes):
            score = scores[:, label]
            keep_idxs = score > self.score_thresh
            score = score[keep_idxs]
            box = boxes[keep_idxs]
            unc = bbox_unc[keep_idxs]  # Filter uncertainty

            # Keep only top-k scoring predictions
            num_topk = min(self.topk_candidates, score.size(0))
            score, idxs = score.topk(num_topk)
            box = box[idxs]
            unc = unc[idxs]  # Select top-k uncertainty

            image_boxes.append(box)
            image_scores.append(score)
            image_labels.append(torch.full_like(score, fill_value=label, dtype=torch.int64, device=device))
            image_uncertainty.append(unc)

        image_boxes = torch.cat(image_boxes, dim=0)
        image_scores = torch.cat(image_scores, dim=0)
        image_labels = torch.cat(image_labels, dim=0)
        image_uncertainty = torch.cat(image_uncertainty, dim=0)

        # Non-maximum suppression
        keep = box_ops.batched_nms(image_boxes, image_scores, image_labels, self.nms_thresh)
        keep = keep[: self.detections_per_img]

        detections.append(
            {
                "boxes": image_boxes[keep],
                "scores": image_scores[keep],
                "labels": image_labels[keep],
                "bbox_uncertainty": image_uncertainty[keep],  # Add uncertainty to output
            }
        )
    return detections

def compute_loss_with_unc( ##THANH: Add NLL loss for uncertainty head
        self,
        targets: List[Dict[str, Tensor]],
        head_outputs: Dict[str, Tensor],
        anchors: List[Tensor],
        matched_idxs: List[Tensor],
    ) -> Dict[str, Tensor]:
    
    bbox_regression = head_outputs["bbox_regression"]
    bbox_regression_uncertainty = head_outputs["bbox_regression"]
    cls_logits = head_outputs["cls_logits"]

    num_foreground = 0
    bbox_loss = []
    uncertainty_loss = []
    cls_targets = []
    
    for (
        targets_per_image,
        bbox_regression_per_image,
        bbox_regression_uncertainty_per_image,  # Uncertainty per image
        cls_logits_per_image,
        anchors_per_image,
        matched_idxs_per_image,
    ) in zip(targets, bbox_regression, bbox_regression_uncertainty, cls_logits, anchors, matched_idxs):
        
        # Match ground truth boxes with default anchors
        foreground_idxs_per_image = torch.where(matched_idxs_per_image >= 0)[0]
        foreground_matched_idxs_per_image = matched_idxs_per_image[foreground_idxs_per_image]
        num_foreground += foreground_matched_idxs_per_image.numel()

        # Calculate regression loss with uncertainty (Gaussian NLL)
        matched_gt_boxes_per_image = targets_per_image["boxes"][foreground_matched_idxs_per_image]
        bbox_regression_per_image = bbox_regression_per_image[foreground_idxs_per_image, :]
        anchors_per_image = anchors_per_image[foreground_idxs_per_image, :]
        
        target_regression = self.box_coder.encode_single(matched_gt_boxes_per_image, anchors_per_image)
        
        if bbox_regression_uncertainty is not None:
            # Extract uncertainty (log variance) and calculate variance
            uncertainty_per_image = bbox_regression_uncertainty_per_image[foreground_idxs_per_image, :]
            sigma_squared = torch.exp(uncertainty_per_image)  # Convert log variance to variance

            # Gaussian NLL loss for bbox coordinates
            gaussian_nll = 0.5 * (torch.pow(bbox_regression_per_image - target_regression, 2) / sigma_squared + uncertainty_per_image)
            uncertainty_loss.append(gaussian_nll.sum())  # Append the loss based on uncertainty
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
            )
        else:
            # If uncertainty is not available, use standard Smooth L1 loss 
            bbox_loss.append(
                torch.nn.functional.smooth_l1_loss(bbox_regression_per_image, target_regression, reduction="sum")
            )

        # Classification loss target preparation
        gt_classes_target = torch.zeros(
            (cls_logits_per_image.size(0),),
            dtype=targets_per_image["labels"].dtype,
            device=targets_per_image["labels"].device,
        )
        gt_classes_target[foreground_idxs_per_image] = targets_per_image["labels"][
            foreground_matched_idxs_per_image
        ]
        cls_targets.append(gt_classes_target)

    bbox_loss = torch.stack(bbox_loss)
    uncertainty_loss = torch.stack(uncertainty_loss) if uncertainty_loss else torch.tensor(0.0, device=bbox_loss.device)
    cls_targets = torch.stack(cls_targets)

    # Classification loss (cross entropy)
    num_classes = cls_logits.size(-1)
    cls_loss = F.cross_entropy(cls_logits.view(-1, num_classes), cls_targets.view(-1), reduction="none").view(cls_targets.size())

    # Hard negative mining for classification loss
    foreground_idxs = cls_targets > 0
    num_negative = self.neg_to_pos_ratio * foreground_idxs.sum(1, keepdim=True)
    negative_loss = cls_loss.clone()
    negative_loss[foreground_idxs] = -float("inf")
    values, idx = negative_loss.sort(1, descending=True)
    background_idxs = idx.sort(1)[1] < num_negative

    N = max(1, num_foreground)
    return {
        "bbox_regression": bbox_loss.sum() / N,
        "classification": (cls_loss[foreground_idxs].sum() + cls_loss[background_idxs].sum()) / N,
        "uncertainty_loss": uncertainty_loss.sum() / N,  # Include uncertainty loss
    }

def forward_with_unc( ## THANH: Use this to replace standard forward function in SSD class
    self, images: List[Tensor], targets: Optional[List[Dict[str, Tensor]]] = None
) -> Tuple[Dict[str, Tensor], List[Dict[str, Tensor]]]:
    if self.training and targets is None:
        raise ValueError("In training mode, targets should be passed")

    if self.training:
        assert targets is not None
        for target in targets:
            boxes = target["boxes"]
            if isinstance(boxes, torch.Tensor):
                if len(boxes.shape) != 2 or boxes.shape[-1] != 4:
                    raise ValueError(f"Expected target boxes to be a tensor of shape [N, 4], got {boxes.shape}.")
            else:
                raise ValueError(f"Expected target boxes to be of type Tensor, got {type(boxes)}.")

    # Get the original image sizes
    original_image_sizes: List[Tuple[int, int]] = []
    for img in images:
        val = img.shape[-2:]
        assert len(val) == 2
        original_image_sizes.append((val[0], val[1]))

    # Transform the input
    images, targets = self.transform(images, targets)

    # Check for degenerate boxes
    if targets is not None:
        for target_idx, target in enumerate(targets):
            boxes = target["boxes"]
            degenerate_boxes = boxes[:, 2:] <= boxes[:, :2]
            if degenerate_boxes.any():
                bb_idx = torch.where(degenerate_boxes.any(dim=1))[0][0]
                degen_bb: List[float] = boxes[bb_idx].tolist()
                raise ValueError(
                    "All bounding boxes should have positive height and width."
                    f" Found invalid box {degen_bb} for target at index {target_idx}."
                )

    # Get the features from the backbone
    features = self.backbone(images.tensors)
    if isinstance(features, torch.Tensor):
        features = OrderedDict([("0", features)])

    features = list(features.values())

    # Compute the SSD heads outputs using the features
    head_outputs = self.head(features)

    # Create the set of anchors
    anchors = self.anchor_generator(images, features)

    losses = {}
    detections: List[Dict[str, Tensor]] = []
    if self.training:
        assert targets is not None

        matched_idxs = []
        for anchors_per_image, targets_per_image in zip(anchors, targets):
            if targets_per_image["boxes"].numel() == 0:
                matched_idxs.append(
                    torch.full((anchors_per_image.size(0),), -1, dtype=torch.int64, device=anchors_per_image.device)
                )
                continue

            match_quality_matrix = box_ops.box_iou(targets_per_image["boxes"], anchors_per_image)
            matched_idxs.append(self.proposal_matcher(match_quality_matrix))

        losses = self.compute_loss(targets, head_outputs, anchors, matched_idxs)
    else:
        detections = self.postprocess_detections(head_outputs, anchors, images.image_sizes)
        detections = self.transform.postprocess(detections, images.image_sizes, original_image_sizes)

    if torch.jit.is_scripting():
        if not self._has_warned:
            warnings.warn("SSD always returns a (Losses, Detections) tuple in scripting")
            self._has_warned = True
        return losses, detections
    return self.eager_outputs(losses, detections)


def ssdlite320_mobilenet_v3_large_unc(# Define SSDlite uncertainty model to use the newly added head 
    pretrained: bool = False,
    weights_name = "ssd_model_with_uncertainty_epoch_70",
    progress: bool = True,
    num_classes: int = 2,
    pretrained_backbone: bool = False,
    trainable_backbone_layers: Optional[int] = None,
    norm_layer: Optional[Callable[..., nn.Module]] = None,
    **kwargs: Any,
):
    if "size" in kwargs:
        warnings.warn("The size of the model is already fixed; ignoring the argument.")

    trainable_backbone_layers = _validate_trainable_layers(
        pretrained or pretrained_backbone, trainable_backbone_layers, 6, 6
    )

    if pretrained:
        pretrained_backbone = False

    # Enable reduced tail if no pretrained backbone is selected.
    reduce_tail = not pretrained_backbone

    if norm_layer is None:
        norm_layer = partial(nn.BatchNorm2d, eps=0.001, momentum=0.03)

    # Create the MobileNetV3 Large backbone
    backbone = mobilenet.mobilenet_v3_large(
        weights=None if not pretrained_backbone else "IMAGENET1K_V1", progress=progress, norm_layer=norm_layer, reduced_tail=reduce_tail, **kwargs
    )
    if not pretrained_backbone:
        _normal_init(backbone)
    
    # Create the feature extractor from the backbone
    backbone = _mobilenet_extractor(
        backbone,
        trainable_backbone_layers,
        norm_layer,
    )

    # Define the input size and anchor generator
    size = (320, 320)
    anchor_generator = DefaultBoxGenerator([[2, 3] for _ in range(6)], min_ratio=0.2, max_ratio=0.95)
    out_channels = det_utils.retrieve_out_channels(backbone, size)
    num_anchors = anchor_generator.num_anchors_per_location()
    assert len(out_channels) == len(anchor_generator.aspect_ratios)

    # Default model configuration
    defaults = {
        "score_thresh": 0.001,
        "nms_thresh": 0.55,
        "detections_per_img": 300,
        "topk_candidates": 300,
        "image_mean": [0.5, 0.5, 0.5],
        "image_std": [0.5, 0.5, 0.5],
    }
    kwargs = {**defaults, **kwargs}

    # Create the final SSD model with the updated head
    model = SSD(
        backbone,
        anchor_generator,
        size,
        num_classes,
        head=SSDLiteHeadUncertainty(out_channels, num_anchors, num_classes, norm_layer),
        **kwargs,
    )

    model.postprocess_detections = postprocess_detections_with_unc.__get__(model, SSD)
    model.forward = forward_with_unc.__get__(model, SSD)
    model.compute_loss = compute_loss_with_unc.__get__(model, SSD)

    # Load pre-trained weights if specified
    if pretrained:
        #weights_name = "ssd_model_with_uncertainty" # This is copied from the trained weight
        weights_name = weights_name
        weights = torch.load(weights_name + '.pth')
        model.load_state_dict(weights)

    return model

def compute_ci95_bboxes(bbox, log_var):
    ##Compute CI95 bounding boxes given bbox coordinates and log variances.

    bbox = torch.tensor(bbox, dtype=torch.float32, device=bbox.device) if not isinstance(bbox, torch.Tensor) else bbox.clone().detach().requires_grad_(True)
    std_dev = torch.exp(0.5 * log_var.clone().detach().to(dtype=torch.float32, device=bbox.device))

    # Compute CI95 bounds
    direction = torch.tensor([1, 1, -1, -1], dtype=torch.float32, device=bbox.device)
    ci_lower = bbox + 1.96 * std_dev * direction 
    ci_upper = bbox - 1.96 * std_dev * direction 

    return ci_lower.tolist(), ci_upper.tolist()

def visualize_max_score_bbox(prediction, image, ssd_alea_unc_utils, threshold=0, save_output=False, output_path="result.png"):
    ###  Visualize bounding box with the highest score and its CI95 bounding boxes.
    
    boxes = prediction[0]['boxes']
    labels = prediction[0]['labels']
    scores = prediction[0]['scores']
    uncertainty = prediction[0]['bbox_uncertainty']
    
    # Get the box with the highest score
    max_score_index = torch.argmax(scores)
    max_score = scores[max_score_index]

    if max_score <= threshold:
        print("No bounding boxes above the threshold.")
        return

    # Convert grayscale image to RGB
    transform_to_PIL = T.ToPILImage()
    img = transform_to_PIL(image[0]).convert("RGB")

    # Draw the highest-scoring bounding box
    draw = ImageDraw.Draw(img)
    max_score_box = boxes[max_score_index]
    draw.rectangle(
        [(max_score_box[0], max_score_box[1]), (max_score_box[2], max_score_box[3])],
        outline=(255, 255, 255),
        width=3
    )

    # Compute CI95 bounding boxes
    CI95bboxes = ssd_alea_unc_utils.compute_ci95_bboxes(boxes[max_score_index], uncertainty[max_score_index])
    lower_bound, upper_bound = CI95bboxes
    print("CI95 Bounding Boxes:", CI95bboxes)

    # Draw CI95 bounding boxes
    ci95_img = transform_to_PIL(image[0]).convert("RGB")
    ci95_draw = ImageDraw.Draw(ci95_img)
    ci95_draw.rectangle(
        [(lower_bound[0], lower_bound[1]), (lower_bound[2], lower_bound[3])],
        outline=(0, 0, 255),
        width=2
    )
    ci95_draw.rectangle(
        [(upper_bound[0], upper_bound[1]), (upper_bound[2], upper_bound[3])],
        outline=(255, 0, 0),
        width=2
    )

    # Use matplotlib for display
    plt.figure(figsize=(6, 6))
    plt.imshow(ci95_img)
    plt.title("CI95 Bounding Boxes")
    plt.axis("off")

    # Add a custom legend
    plt.text(10, 20, "CI95 Lower Bound", color='blue', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))
    plt.text(10, 40, "CI95 Upper Bound", color='red', fontsize=10, bbox=dict(facecolor='white', alpha=0.6))

    plt.show()

    # Optionally save output
    if save_output:
        ci95_img.save(output_path)
        print(f"Image saved to: {output_path}")