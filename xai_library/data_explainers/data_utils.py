import os
import numpy as np
from typing import Tuple
from torchvision.datasets import CocoDetection
import matplotlib.pyplot as plt
import json

### Dataset wrapper for Toy Model ###

class ToyModelDataset:
    def __init__(self, root_dir:str, images_subdir: str, annotation_file_path: str):
        """
        :param root_dir:                Where the data is stored.
        :param images_subdir:           Local path to directory that includes all the images. Path relative to `root_dir`.
        :param annotation_file_path:    Local path to annotation file. Path relative to `root_dir`.
        """

        self.base_dataset = CocoDetection(
            root=os.path.join(root_dir, images_subdir),
            annFile=os.path.join(root_dir, annotation_file_path),
        )
        self.class_ids = self.base_dataset.coco.getCatIds()

        categories = self.base_dataset.coco.loadCats(self.class_ids)
        self.class_names = {class_id: category["name"] for class_id, category in zip(self.class_ids, categories)}

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __iter__(self) -> Tuple[np.ndarray, np.ndarray]:
        for i in range(len(self)):
            yield self[i]

    def __getitem__(self, index: int) -> Tuple[np.ndarray, np.ndarray]:
        image, annotations = self.base_dataset[index]
        image = np.array(image)
        labels = []
        for annotation in annotations:
            class_id = annotation["category_id"]
            x1, y1, x2, y2 = annotation["bbox"]
            labels.append((int(class_id), float(x1), float(y1), float(x2-x1), float(y2-y1)))
        labels = np.array(labels, dtype=np.float32).reshape(-1, 5) # , dtype=np.float32

        return image, labels

### JSON parsing ###

def parse_json(bbox_file):
    def calculate_iou(box1, box2):
        # box format (xmin, ymin, xmax, ymax)
        x1_inter = max(box1[0], box2[0])
        y1_inter = max(box1[1], box2[1])
        x2_inter = min(box1[2], box2[2])
        y2_inter = min(box1[3], box2[3])
        
        intersection_area = max(0, x2_inter - x1_inter + 1) * max(0, y2_inter - y1_inter + 1)
        
        box1_area = (box1[2] - box1[0] + 1) * (box1[3] - box1[1] + 1)
        box2_area = (box2[2] - box2[0] + 1) * (box2[3] - box2[1] + 1)
        
        union_area = box1_area + box2_area - intersection_area
        iou = intersection_area / union_area
        return iou
    
    bounding_boxes=[]
    with open(bbox_file, 'r') as f:
        prediction_json = json.load(f)

    for annotation in prediction_json['annotations']:
        xmin = annotation['bbox'][0]
        ymin = annotation['bbox'][1]
        xmax = annotation['bbox'][2]
        ymax = annotation['bbox'][3]

        center_x = (xmin + xmax) / 2
        center_y = (ymin + ymax) / 2
        width = xmax - xmin
        height = ymax - ymin
        area = width * height
        ratio = width / height

        xmin_pred = annotation['pred_bbox'][0]
        ymin_pred = annotation['pred_bbox'][1]
        xmax_pred = annotation['pred_bbox'][2]
        ymax_pred = annotation['pred_bbox'][3]
        pred_score= annotation['pred_score']

        center_x_pred = (xmin_pred + xmax_pred) / 2
        center_y_pred = (ymin_pred + ymax_pred) / 2
        width_pred = xmax_pred - xmin_pred
        height_pred = ymax_pred - ymin_pred
        area_pred = width_pred * height_pred
        if height_pred == 0:
            ratio_pred=1
        else:
            ratio_pred = width_pred / height_pred
        
        bounding_boxes.append({
            'center_x': center_x,
            'center_y': center_y,
            'width': width,
            'height': height,
            'area': area,
            'ratio': ratio,
            'center_x_pred': center_x_pred,
            'center_y_pred': center_y_pred,
            'width_pred': width_pred,
            'height_pred': height_pred,
            'area_pred': area_pred,
            'ratio_pred': ratio_pred,
            'pred_score': pred_score,
            'iou': calculate_iou(annotation['bbox'], annotation['pred_bbox'])
        })

    return bounding_boxes


### Plot functions ###
def plot_heatmap(data, xlabel, ylabel, title, outputdir):
    x, y = zip(*data)

    fig, ax = plt.subplots()
    h = ax.hist2d(x, y, bins=(50, 50), cmap=plt.cm.YlOrRd, cmin=1)
    plt.colorbar(h[3], ax=ax)

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)   

    save_path= outputdir + 'bbox_heatmap.png' 
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

def plot_distribution(data, xlabel, title, outputdir):
    import matplotlib.pyplot as plt
    plt.hist(data, bins=50, alpha=0.7, color='blue', edgecolor='black')
    plt.xlabel(xlabel)
    plt.ylabel('Frequency')
    plt.title(title)
    
    save_path= outputdir + 'bbox_hist.png' 
    if save_path is not None:
        plt.savefig(save_path, bbox_inches='tight')
    plt.show()

### Annotation analysis ###

def analyze_bboxes(bboxes, target_object_name='ground truth satellite'):
    center_positions = []
    dimensions = []
    areas = []
    ratios = []

    for box in bboxes:
        center_positions.append((box['center_x'], box['center_y']))
        dimensions.append((box['width'], box['height']))
        areas.append(box['area'])
        ratios.append(box['ratio'])

    plot_heatmap(center_positions, 'Center X', 'Center Y', f'Center Position Heatmap ({target_object_name})', './')
    plot_heatmap(dimensions, 'Width', 'Height', f'Bounding Box Dimensions Heatmap ({target_object_name})', './')
    plot_distribution(areas, 'Bounding Box Area', f'Bounding Box Area Distribution ({target_object_name})', './')
    plot_distribution(ratios, 'Bounding Box Ratio', f'Bounding Box Ratio Distribution ({target_object_name})', './')

### IoU metric helper functions ###

def compute_iou(bb):
    """
    Compute the Intersection over Union (IoU) of two bounding boxes.
    """
    x1, y1, w1, h1 = bb['center_x'], bb['center_y'], bb['width'], bb['height']
    x2, y2, w2, h2 = bb['center_x_pred'], bb['center_y_pred'], bb['width_pred'], bb['height_pred']

    x_left = max(x1 - w1 / 2, x2 - w2 / 2)
    y_top = max(y1 - h1 / 2, y2 - h2 / 2)
    x_right = min(x1 + w1 / 2, x2 + w2 / 2)
    y_bottom = min(y1 + h1 / 2, y2 + h2 / 2)

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)

    bb1_area = w1 * h1
    bb2_area = w2 * h2

    union_area = bb1_area + bb2_area - intersection_area

    iou = intersection_area / union_area

    return iou

def analyze_iou_correlations(bboxes):
    iou_values = [item['iou'] for item in bboxes]
    
    variables = ['center_x', 'center_y', 'width', 'height', 'area', 'ratio', 
                 'center_x_pred', 'center_y_pred', 'width_pred', 'height_pred', 
                 'area_pred', 'ratio_pred', 'pred_score']
    
    correlations = {}
    for var in variables:
        var_values = [item[var] for item in bboxes]
        corr = np.corrcoef(iou_values, var_values)[0, 1]
        correlations[var] = corr
        print(f"Correlation between IoU and {var}: {corr}")
    
    for var in variables:
        var_values = [item[var] for item in bboxes]
        plt.scatter(var_values, iou_values)
        plt.xlabel(var)
        plt.ylabel('IoU')
        plt.title(f'IoU vs {var}')
        plt.show()