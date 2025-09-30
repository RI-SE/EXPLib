import torch.nn as nn
import torch
import torch.nn.functional as F

def aleatoric_loss(y_pred_boxes_mean, y_pred_boxes_var, y_pred_classes_mean, y_pred_classes_var, 
                   y_true_boxes, y_true_classes, lambda_class=1.0, lambda_box=1.0):
    """
    Calculates the aleatoric uncertainty loss for object detection.

    Arguments:
    - y_pred_boxes_mean: Predicted bounding box coordinates (means).
    - y_pred_boxes_var: Predicted variance for bounding box coordinates.
    - y_pred_classes_mean: Predicted class logits (means).
    - y_pred_classes_var: Predicted variance for class logits.
    - y_true_boxes: Ground truth bounding box coordinates.
    - y_true_classes: Ground truth classes (integer labels).
    - lambda_class: Weight for classification loss component.
    - lambda_box: Weight for bounding box loss component.

    Returns:
    - Combined loss accounting for aleatoric uncertainty.
    """

    # 1. Bounding Box Loss with Aleatoric Uncertainty
    box_residual = (y_true_boxes - y_pred_boxes_mean) ** 2
    box_loss = torch.mean(box_residual / (2 * y_pred_boxes_var) + 0.5 * torch.log(y_pred_boxes_var))

    # 2. Classification Loss with Aleatoric Uncertainty
    # Use cross-entropy loss for class mean predictions, scaled by class variance
    class_loss = F.cross_entropy(y_pred_classes_mean, y_true_classes, reduction='none')
    class_loss = torch.mean(class_loss / y_pred_classes_var + 0.5 * torch.log(y_pred_classes_var))

    # Combine losses with respective weights
    total_loss = lambda_box * box_loss + lambda_class * class_loss

    return total_loss



class SSDLiteMobileNetV3WithAleatoric(nn.Module):
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.base_model = base_model
        
        # use feature extraction backbone, 
        # and output head for aleatoric uncertainty
        self.output_layer = nn.Linear(1024, (num_classes + 4) * 2)  # Double output size for variance
        
    def forward(self, x):
        features = self.base_model(x)  # Extract features from base model
        outputs = self.output_layer(features)  # Pass through custom output layer
        
        # Split outputs into means and variances
        y_pred_boxes_mean = outputs[:, :4]                  # Bounding box mean (x, y, width, height)
        y_pred_boxes_var = torch.exp(outputs[:, 4:8])       # Bounding box variance (log-variance for stability)
        
        y_pred_classes_mean = outputs[:, 8:8+num_classes]   # Class scores mean
        y_pred_classes_var = torch.exp(outputs[:, 8+num_classes:])  # Class scores variance (log-variance for stability)
        
        return y_pred_boxes_mean, y_pred_boxes_var, y_pred_classes_mean, y_pred_classes_var

#####
#### Training
# for inputs, (y_true_boxes, y_true_classes) in dataloader:
#     optimizer.zero_grad()
    
#     y_pred_boxes_mean, y_pred_boxes_var, y_pred_classes_mean, y_pred_classes_var = model(inputs)
    
#     loss = aleatoric_loss(
#         y_pred_boxes_mean, y_pred_boxes_var,
#         y_pred_classes_mean, y_pred_classes_var,
#         y_true_boxes, y_true_classes,
#         lambda_class=1.0, lambda_box=1.0
#     )
    
#     loss.backward()
#     optimizer.step()