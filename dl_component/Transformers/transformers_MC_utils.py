import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt

from tqdm import tqdm
from transformers import DetrForObjectDetection, AutoImageProcessor
import warnings
from torchvision.ops import box_iou

def force_dropout_in_detr(model, p=0.3, backbone_p=0.1):
    """
    Add dropout to DETR transformer
    """

    for name, module in model.model.named_modules():
        # Attention output dropout
        if hasattr(module, "self_attn") and hasattr(module.self_attn, "out_proj"):
            module.self_attn.out_proj = nn.Sequential(
                module.self_attn.out_proj,
                nn.Dropout(p)
            )

        # FFN output dropout
        if hasattr(module, "linear2"):
            module.linear2 = nn.Sequential(
                module.linear2,
                nn.Dropout(p)
            )

    backbone = model.model.backbone

    if hasattr(backbone, "body"):
        for name, child in backbone.body.named_children():
            backbone.body._modules[name] = nn.Sequential(child, nn.Dropout2d(backbone_p))
    else:
        for name, child in backbone.named_children():
            if isinstance(child, nn.Sequential) or isinstance(child, nn.Conv2d):
                backbone._modules[name] = nn.Sequential(child, nn.Dropout2d(backbone_p))

    return model


def enable_dropout(model):
    for m in model.modules():
        if isinstance(m, (nn.Dropout, nn.Dropout2d)):
            m.train()



def mc_dropout_bboxes(model, image, feature_extractor, n_iter=30, conf_thresh=0.9, iou_thresh=0.5):
    ###Monte Carlo Dropout 
    
    enable_dropout(model)

    all_boxes, all_probs = [], []

    for i in tqdm(range(n_iter), desc="MC Dropout Runs"):
        encoding = feature_extractor(images=image, return_tensors="pt")
        with torch.no_grad():
            outputs = model(**encoding)

        probs = outputs.logits.softmax(-1)[0, :, :-1]
        keep = probs.max(-1).values > conf_thresh

        target_sizes = torch.tensor([image.size[::-1]])
        postprocessed = feature_extractor.post_process_object_detection(
            outputs, threshold=conf_thresh, target_sizes=target_sizes
        )[0]

        boxes = postprocessed["boxes"]
        scores = postprocessed["scores"]
        probs_kept = probs[keep]

        if boxes.numel() == 0:
            continue

        all_boxes.append(boxes.cpu())
        all_probs.append(probs_kept.cpu())

    num_valid = len(all_boxes)
    if num_valid == 0:
        print("No valid detections found in any MC iteration.")
        return np.zeros((0, 4)), np.zeros((0, 4)), np.zeros((0, model.config.num_labels))

    if num_valid < n_iter / 3:
        print(f"Only {num_valid}/{n_iter} iterations had detections; "
              f"consider lowering conf_thresh (currently {conf_thresh}).")

    ref_boxes = all_boxes[0]
    n_ref = ref_boxes.shape[0]

    matched_boxes = [torch.zeros((num_valid, 4)) for _ in range(n_ref)]
    matched_probs = [torch.zeros((num_valid, all_probs[0].shape[1])) for _ in range(n_ref)]

    for t in range(num_valid):
        boxes_t = all_boxes[t]
        probs_t = all_probs[t]

        ious = box_iou(ref_boxes, boxes_t)

        for i in range(n_ref):
            best_idx = torch.argmax(ious[i])
            if ious[i, best_idx] > iou_thresh:
                matched_boxes[i][t] = boxes_t[best_idx]
                matched_probs[i][t] = probs_t[best_idx]
            else:
                matched_boxes[i][t] = ref_boxes[i]
                matched_probs[i][t] = all_probs[0][i]

    mean_boxes = torch.stack([b.mean(0) for b in matched_boxes]).numpy()
    std_boxes = torch.stack([b.std(0) for b in matched_boxes]).numpy()
    mean_probs = torch.stack([p.mean(0) for p in matched_probs]).numpy()

    return mean_boxes, std_boxes, mean_probs

def plot_bboxes_with_ci95(pil_img, mean_boxes, std_boxes, probs, labels, colors=None):
    
    ## Plot  inner and outer bounding boxes with 95% confidence intervals (CI95).
    
    if colors is None:
        COLORS = [[0.000, 0.447, 0.741], [0.850, 0.325, 0.098],
                  [0.929, 0.694, 0.125], [0.494, 0.184, 0.556],
                  [0.466, 0.674, 0.188], [0.301, 0.745, 0.933]] * 100
    else:
        COLORS = colors

    plt.figure(figsize=(16, 10))
    plt.imshow(pil_img)
    ax = plt.gca()

    z = 1.96  # for 95% confidence interval

    for i, (mean_box, std_box, prob) in enumerate(zip(mean_boxes, std_boxes, probs)):
        xmin, ymin, xmax, ymax = mean_box
        dxmin, dymin, dxmax, dymax = std_box
        color = COLORS[i % len(COLORS)]

        # # --- Mean bounding box ---
        # ax.add_patch(plt.Rectangle(
        #     (xmin, ymin),
        #     xmax - xmin,
        #     ymax - ymin,
        #     fill=False, color=color, linewidth=2, label='mean box' if i == 0 else "")
        # )

        # Outer CI95
        ax.add_patch(plt.Rectangle(
            (xmin - z*dxmin, ymin - z*dymin),
            (xmax - xmin) + z*(dxmin + dxmax),
            (ymax - ymin) + z*(dymin + dymax),
            fill=False, linestyle='--', edgecolor=color, linewidth=1, alpha=0.8, label='95% CI outer' if i == 0 else "")
        )

        # Inner CI95 
        ax.add_patch(plt.Rectangle(
            (xmin + z*dxmin, ymin + z*dymin),
            (xmax - xmin) - z*(dxmin + dxmax),
            (ymax - ymin) - z*(dymin + dymax),
            fill=False, linestyle=':', edgecolor=color, linewidth=1, alpha=0.9, label='95% CI inner' if i == 0 else "")
        )

        cl = prob.argmax()
        text = f'{labels[cl.item()]}: {prob[cl]:.2f}'
        ax.text(xmin, ymin - 5, text, fontsize=12, color='black',
                bbox=dict(facecolor='yellow', alpha=0.5, edgecolor='none'))

    plt.axis("off")
    plt.title("Bounding Boxes with 95% Confidence Intervals", fontsize=16)
    handles, labels_plot = ax.get_legend_handles_labels()
    if i == 0:
        plt.legend(handles, labels_plot, loc='upper right')
    plt.show()