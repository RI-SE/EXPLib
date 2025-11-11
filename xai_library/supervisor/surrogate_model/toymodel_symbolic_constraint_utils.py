import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from torchvision.datasets import CocoDetection
import torchvision.transforms as T
import torch.nn.functional as F
import numpy as np
from pycocotools.coco import COCO
from sklearn.linear_model import LinearRegression
import math, sys
from torch.amp import autocast
from torchvision.models.detection import ssdlite320_mobilenet_v3_large


def collate_fn(batch):
    """
    Convert COCO [x1,y1,x2,y2] annotations into targets expected by torchvision detectors
    """
    images, annos = zip(*batch)
    images = list(images)

    targets = []
    for anno in annos:
        if len(anno) == 0:
            targets.append({'boxes': torch.zeros((0, 4), dtype=torch.float32),
                            'labels': torch.zeros((0,), dtype=torch.int64)})
            continue

        boxes = []
        labels = []
        for obj in anno:
            x_min, y_min, x_max, y_max = obj['bbox']

            if (x_max <= x_min) or (y_max <= y_min):
                continue

            boxes.append([x_min, y_min, x_max, y_max])
            labels.append(int(obj['category_id']))

        if len(boxes) == 0:
            targets.append({'boxes': torch.zeros((0, 4), dtype=torch.float32),
                            'labels': torch.zeros((0,), dtype=torch.int64)})
        else:
            boxes = torch.tensor(boxes, dtype=torch.float32)
            labels = torch.tensor(labels, dtype=torch.int64)
            targets.append({'boxes': boxes, 'labels': labels})

    return images, targets


def create_dataloaders(dataset_path, coco_annotations, batch_size=8, num_workers=0, split=(0.9, 0.05, 0.05)):
    transform = T.Compose([T.ToTensor()])
    dataset = CocoDetection(root=dataset_path, annFile=coco_annotations, transform=transform)

    total_size = len(dataset)
    if total_size == 0:
        raise RuntimeError("Dataset is empty — check dataset_path and coco_annotations.")

    train_size = int(split[0] * total_size)
    val_size = int(split[1] * total_size)
    test_size = total_size - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    data_loader_train = DataLoader(train_dataset, batch_size=batch_size, sampler=RandomSampler(train_dataset),
                                   num_workers=num_workers, collate_fn=collate_fn)
    data_loader_val = DataLoader(val_dataset, batch_size=batch_size, sampler=SequentialSampler(val_dataset),
                                 num_workers=num_workers, collate_fn=collate_fn)
    data_loader_test = DataLoader(test_dataset, batch_size=batch_size, sampler=SequentialSampler(test_dataset),
                                  num_workers=num_workers, collate_fn=collate_fn)

    return data_loader_train, data_loader_val, data_loader_test


def compute_bbox_constraints_from_coco_json(coco_json_path):
    coco = COCO(coco_json_path)
    bbox_areas, bbox_ratios, bbox_centers, center_area_data, center_ratio_data = [], [], [], [], []

    for ann in coco.dataset.get("annotations", []):
        if "bbox" not in ann:
            continue
        x, y, w, h = ann["bbox"]
        if w <= 0 or h <= 0:
            continue
        x2, y2 = x + w, y + h
        area = w * h
        ratio = h / w
        cx, cy = (x + x2) / 2, (y + y2) / 2

        bbox_areas.append(area)
        bbox_ratios.append(ratio)
        bbox_centers.append((cx, cy))
        center_area_data.append([cx, cy, area])
        center_ratio_data.append([cx, cy, ratio])

    bbox_areas, bbox_ratios = np.array(bbox_areas), np.array(bbox_ratios)
    area_min, area_max = np.percentile(bbox_areas, [2, 98])
    ratio_min, ratio_max = np.percentile(bbox_ratios, [2, 98])

    area_model = LinearRegression().fit(np.array(bbox_centers), bbox_areas)
    ratio_model = LinearRegression().fit(np.array(bbox_centers), bbox_ratios)

    return {
        "area_range": (float(area_min), float(area_max)),
        "ratio_range": (float(ratio_min), float(ratio_max)),
        "area_model": area_model,
        "ratio_model": ratio_model,
    }


def class_occurrence_constraint(labels_list, class_id=1, device="cpu"):
    penalty = 0.0
    for labels in labels_list:
        if labels is None or labels.numel() == 0:
            continue
        counts = (labels == class_id).sum().item()
        if counts > 1:
            penalty += (counts - 1)
    return torch.tensor(penalty, dtype=torch.float32, device=device)


def bbox_area_constraint(boxes_list, area_constraints, device="cpu"):
    a_min, a_max = area_constraints
    penalties = []
    for boxes in boxes_list:
        if boxes is None or boxes.numel() == 0:
            continue
        w, h = (boxes[:, 2] - boxes[:, 0]).clamp(min=0), (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        areas = w * h
        penalties.append((F.relu(a_min - areas) + F.relu(areas - a_max)).sum())
    return torch.stack(penalties).sum() if penalties else torch.tensor(0.0, device=device)


def bbox_ratio_constraint(boxes_list, ratio_constraints, device="cpu"):
    r_min, r_max = ratio_constraints
    penalties = []
    for boxes in boxes_list:
        if boxes is None or boxes.numel() == 0:
            continue
        w, h = (boxes[:, 2] - boxes[:, 0]).clamp(min=1e-6), (boxes[:, 3] - boxes[:, 1]).clamp(min=0)
        ratios = h / w
        penalties.append((F.relu(r_min - ratios) + F.relu(ratios - r_max)).sum())
    return torch.stack(penalties).sum() if penalties else torch.tensor(0.0, device=device)


def symbolic_loss(preds, class_id, constraints, area_w=1e-3, ratio_w=1e-3, occ_w=1e-2, device="cpu"):
    boxes_list, labels_list = [], []
    for p in preds:
        boxes_list.append(p.get("boxes", torch.empty(0, 4)).to(device))
        labels_list.append(p.get("labels", torch.empty(0, dtype=torch.int64)).to(device))

    area_pen = bbox_area_constraint(boxes_list, constraints["area_range"], device) * area_w
    ratio_pen = bbox_ratio_constraint(boxes_list, constraints["ratio_range"], device) * ratio_w
    class_pen = class_occurrence_constraint(labels_list, class_id, device) * occ_w

    return area_pen + ratio_pen + class_pen


def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=50, scaler=None, bbox_constraints=None, utils=None):
    model.train()

    logger_iter = enumerate(data_loader)
    for i, (images, targets) in logger_iter:
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()
        with autocast('cuda',enabled=(scaler is not None)):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

            # Get preds safely in eval mode
            model.eval()
            with torch.no_grad():
                preds = model(images)
            model.train()

            symbolic_penalty = symbolic_loss(preds, class_id=1, constraints=bbox_constraints, device=device)
            total_loss = losses + symbolic_penalty

        loss_value = total_loss.item()
        if not math.isfinite(loss_value):
            print(f"Non-finite loss {loss_value}, stopping")
            sys.exit(1)

        if scaler is not None:
            scaler.scale(total_loss).backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        if i % print_freq == 0:
            print(f"[Epoch {epoch} | Iter {i}] Loss: {total_loss.item():.4f}")

    return


def create_ssdlite(num_classes=2, device=None):
    model = ssdlite320_mobilenet_v3_large(progress=True, num_classes=num_classes)
    if device:
        model.to(device)
    return model