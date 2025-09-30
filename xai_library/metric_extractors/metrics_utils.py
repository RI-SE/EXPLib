### Examples metric extractors, read groundtruth and prediction from json file with the same format
# Groundtruth assume positive values for all entries found
# Examples json are identical except that one item is added to predictions
# TODO: Fix error for AP computations (sorting of classes and AP curve)


import json
import numpy as np
import torch
from sklearn.metrics import precision_recall_curve, average_precision_score

def calculate_iou(box1, box2):
    # Calculate IoU between two bounding boxes
    x1, y1, w1, h1 = box1
    x2, y2, w2, h2 = box2
    
    # Calculate intersection coordinates
    x_intersection = max(x1, x2)
    y_intersection = max(y1, y2)
    x_intersection_end = min(x1 + w1, x2 + w2)
    y_intersection_end = min(y1 + h1, y2 + h2)
    
    # Calculate intersection area
    intersection_area = max(0, x_intersection_end - x_intersection) * max(0, y_intersection_end - y_intersection)
    
    # Calculate union area
    box1_area = w1 * h1
    box2_area = w2 * h2
    union_area = box1_area + box2_area - intersection_area
    
    # Calculate IoU
    iou = intersection_area / union_area
    
    return iou

def parse_json(json_file):
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def map_category_to_class_id(ground_truth):
    category_ids = set()
    for annotation in ground_truth:
        category_ids.add(annotation["category_id"])
    category_to_class_id = {category_id: class_id for class_id, category_id in enumerate(sorted(category_ids))}
    return category_to_class_id

def compute_precision_recall(predictions, ground_truth, category_to_class_id, score_threshold):
    num_classes = len(category_to_class_id)
    # Initialize variables to store TP, FP, FN for each class
    precision = np.zeros(num_classes)
    recall = np.zeros(num_classes)
    
    for category_id, class_id in category_to_class_id.items():
        # Filter predictions for the current class and score threshold
        pred_class = [pred for pred in predictions if pred["category_id"] == category_id and pred["score"] >= score_threshold]
        
        # Filter ground truth annotations for the current class
        gt_class = [gt for gt in ground_truth if gt["category_id"] == category_id]
        num_gt = len(gt_class)
        
        # Compute true positives and false positives
        true_positives = 0
        false_positives = len(pred_class)
        
        for pred_bbox in pred_class:
            # Check if there is any ground truth annotation overlapping with the prediction
            for gt_bbox in gt_class:
                iou = calculate_iou(pred_bbox["bbox"], gt_bbox["bbox"])
                if iou >= 0.5:
                    true_positives += 1
                    false_positives -= 1
                    break  # Break the loop since we found a matching ground truth annotation
        
        # Update precision and recall arrays
        if num_gt > 0:
            recall[class_id] = true_positives / num_gt
        if true_positives + false_positives > 0:
            precision[class_id] = true_positives / (true_positives + false_positives)
    
    return precision, recall

def compute_ap(precision, recall):
    # Compute AP using the precision-recall curve 
    # Some BUGS HERE
    ap = 0
    for i in range(len(recall) - 1):
        ap += (recall[i + 1] - recall[i]) * precision[i + 1]
    return ap

def evaluate_object_detection(predictions, ground_truth, category_to_class_id, score_threshold):
    num_classes = len(category_to_class_id)
    # Compute precision and recall for each class
    precision, recall = compute_precision_recall(predictions, ground_truth, category_to_class_id, score_threshold)
    
    # Calculate mAP
    average_precision = []
    for class_id in range(num_classes):
        # Filter predictions for the current class and score threshold
        pred_class = [pred for pred in predictions if pred["category_id"] == class_id and pred["score"] >= score_threshold]
        if pred_class:
            # Compute precision-recall curve
            sorted_indices = np.argsort([-bbox["score"] for bbox in pred_class])
            sorted_predictions = [pred_class[i] for i in sorted_indices]
            sorted_gt = [1] * len(sorted_predictions)
            precision_, recall_, _ = precision_recall_curve(sorted_gt, [-bbox["score"] for bbox in sorted_predictions])
            # Compute AP for the current class
            ap = compute_ap(precision_, recall_)
            average_precision.append(ap)
        else:
            average_precision.append(0)  # If there are no predictions for the class, AP is zero
    
    # Compute mAP
    mAP = np.mean(average_precision)
    
    return precision, recall, average_precision, mAP

def bbox_iou(box1, box2):
    # Calculate Intersection
    inter_rect_x1 = torch.max(box1[..., 0], box2[..., 0])
    inter_rect_y1 = torch.max(box1[..., 1], box2[..., 1])
    inter_rect_x2 = torch.min(box1[..., 2], box2[..., 2])
    inter_rect_y2 = torch.min(box1[..., 3], box2[..., 3])
    inter_area = torch.clamp(inter_rect_x2 - inter_rect_x1 + 1, min=0) * torch.clamp(inter_rect_y2 - inter_rect_y1 + 1, min=0)

    # Calculate Union
    b1_area = (box1[..., 2] - box1[..., 0] + 1) * (box1[..., 3] - box1[..., 1] + 1)
    b2_area = (box2[..., 2] - box2[..., 0] + 1) * (box2[..., 3] - box2[..., 1] + 1)
    union_area = b1_area + b2_area - inter_area

    return inter_area / union_area

