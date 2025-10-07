import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
from scipy.optimize import linear_sum_assignment

def load_coco_predictions(file_path: str | Path) -> List[dict]:
    
    ##Load a COCO-style predictions JSON file.
    with open(file_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "annotations" in data:
        return data["annotations"]
    return data


def bbox_to_xyxy(bbox: List[float]) -> Tuple[float, float, float, float]:
    ## Convert  [x, y, w, h] → [x1, y1, x2, y2] if needed
    x, y, w, h = bbox
    return x, y, x + w, y + h


def compute_iou(box1: Tuple[float, float, float, float],
                box2: Tuple[float, float, float, float]) -> float:
    ## Compute the Intersection over Union of two xyxy boxes
    x1_min, y1_min, x1_max, y1_max = box1
    x2_min, y2_min, x2_max, y2_max = box2

    inter_x_min = max(x1_min, x2_min)
    inter_y_min = max(y1_min, y2_min)
    inter_x_max = min(x1_max, x2_max)
    inter_y_max = min(y1_max, y2_max)

    inter_area = max(0.0, inter_x_max - inter_x_min) * max(0.0, inter_y_max - inter_y_min)

    area1 = (x1_max - x1_min) * (y1_max - y1_min)
    area2 = (x2_max - x2_min) * (y2_max - y2_min)

    union = area1 + area2 - inter_area
    return inter_area / union if union > 0.0 else 0.0


def build_index(preds: List[dict]) -> Dict[int, Dict[int, List[Tuple[float, float, float, float]]]]:
    
    ## Build an index: image_id → category_id → list_of_xyxy_boxes
    index: Dict[int, Dict[int, List[Tuple[float, float, float, float]]]] = defaultdict(lambda: defaultdict(list))
    for ann in preds:
        image_id = ann["image_id"]
        cat_id   = ann["category_id"]
        bbox     = bbox_to_xyxy(ann["bbox"])
        index[image_id][cat_id].append(bbox)
    return index

def compute_fidelity(
    main_json: str | Path,
    surrogate_json: str | Path,
    *,
    min_iou: float = 0.3
) -> Tuple[float, int, int]:
    """
    Compute a fidelity score F = (sum IoU of all boxes) / (max(#main, #surrogate) per object class)

    Parameters
    ----------
    main_json : Path to the main model's predictions JSON.
    surrogate_json : Path to the surrogate predictions JSON.
    min_iou : Minimum IoU required for a match.  

    Returns
    -------
    fidelity : Value in [0,1] 
    n_matches : Number of matched pairs that contributed to the IoU sum.
    n_possible : Total number of boxes that could be matched
    """
    main_preds = load_coco_predictions(main_json)
    surrogate_preds = load_coco_predictions(surrogate_json)

    main_idx = build_index(main_preds)
    sur_idx  = build_index(surrogate_preds)

    total_iou   = 0.0
    n_matches   = 0
    n_possible  = 0   

    for img_id in set(main_idx) & set(sur_idx):
        main_cats   = main_idx[img_id]
        sur_cats    = sur_idx[img_id]

        for cat_id in set(main_cats) & set(sur_cats):
            m_boxes = main_cats[cat_id]
            s_boxes = sur_cats[cat_id]

            if not m_boxes and not s_boxes:
                continue

            n_main = len(m_boxes)
            n_sur  = len(s_boxes)
            n_max  = max(n_main, n_sur)

            iou_mat = np.zeros((n_main, n_sur), dtype=np.float64)
            for i, b1 in enumerate(m_boxes):
                for j, b2 in enumerate(s_boxes):
                    iou_mat[i, j] = compute_iou(b1, b2)

            iou_mat[iou_mat < min_iou] = 0.0

            padded = np.pad(iou_mat, ((0, n_max - n_main), (0, n_max - n_sur)),
                            constant_values=0.0)

            # Hungarian matching
            row_ind, col_ind = linear_sum_assignment(-padded)

            for r, c in zip(row_ind, col_ind):
                if r < n_main and c < n_sur:
                    n_matches   += 1
                    total_iou   += padded[r, c]   
            n_possible += n_max   

    if n_possible == 0:
        fidelity = 1.0
    else:
        fidelity = total_iou / n_possible   # 1.0 → perfect match

    return fidelity, n_matches, n_possible
