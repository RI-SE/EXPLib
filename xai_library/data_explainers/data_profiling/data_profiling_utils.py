import os
import json
import numpy as np
import pandas as pd
from collections import defaultdict
from PIL import Image
import matplotlib.pyplot as plt


def load_annotations(annotations_file):
    """Load COCO-like annotations from a JSON file."""
    with open(annotations_file, 'r') as f:
        return json.load(f)


def compute_image_statistics(annotations, images_dir):
    """Compute detailed image statistics."""
    image_stats = []

    for image_info in annotations['images']:
        image_id = image_info['id']
        image_path = os.path.join(images_dir, image_info['file_name'])

        try:
            with Image.open(image_path) as img:
                img_array = np.array(img.convert('RGB'))
                width, height = img.size
                aspect_ratio = width / height

                # Compute pixel-level stats
                mean_pixel = img_array.mean()
                std_pixel = img_array.std()
                brightness = np.mean(np.max(img_array, axis=2))
                contrast = img_array.std(axis=(0,1)).mean()

                image_stats.append({
                    'image_id': image_id,
                    'file_name': image_info['file_name'],
                    'width': width,
                    'height': height,
                    'aspect_ratio': aspect_ratio,
                    'mean_pixel': mean_pixel,
                    'std_pixel': std_pixel,
                    'brightness': brightness,
                    'contrast': contrast
                })
        except Exception as e:
            print(f"Error processing {image_info['file_name']}: {e}")

    return pd.DataFrame(image_stats)


def xyxy2ywh(bbox):
    x1, y1, x2, y2 = bbox
    return [float(x1), float(y1), float(x2 - x1), float(y2 - y1)]


def compute_bbox_statistics(annotations, bbox_convert=True):
    """Compute statistics of bounding boxes and bbox-image relationships."""
    bbox_stats = []

    image_sizes = {img['id']: (img['width'], img['height']) for img in annotations['images']}

    for ann in annotations['annotations']:
        image_id = ann['image_id']
        category_id = ann['category_id']
        bbox = ann['bbox']  # [x, y, width, height]
        if bbox_convert:
            x, y, w, h = xyxy2ywh(bbox)
        else:
            x, y, w, h = bbox

        if image_id not in image_sizes:
            continue

        img_w, img_h = image_sizes[image_id]
        area = w * h
        aspect_ratio = w / h if h > 0 else 0

        rel_area = area / (img_w * img_h)
        rel_w = w / img_w
        rel_h = h / img_h
        rel_x = x / img_w
        rel_y = y / img_h

        bbox_stats.append({
            'image_id': image_id,
            'category_id': category_id,
            'bbox_width': w,
            'bbox_height': h,
            'bbox_area': area,
            'bbox_aspect_ratio': aspect_ratio,
            'rel_area': rel_area,
            'rel_width': rel_w,
            'rel_height': rel_h,
            'rel_x': rel_x,
            'rel_y': rel_y
        })

    return pd.DataFrame(bbox_stats)


def compute_label_distribution(annotations):
    """Compute the distribution of labels (categories)."""
    category_counts = defaultdict(int)
    for annotation in annotations['annotations']:
        category_counts[annotation['category_id']] += 1

    category_map = {cat['id']: cat['name'] for cat in annotations['categories']}
    category_distribution = {category_map[k]: v for k, v in category_counts.items()}
    return category_distribution


def plot_image_statistics(image_stats_df):
    """Plot distributions of image-related statistics."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    axes[0].hist(image_stats_df['width'], bins=20, alpha=0.7)
    axes[0].set_title('Image Width Distribution')

    axes[1].hist(image_stats_df['height'], bins=20, alpha=0.7)
    axes[1].set_title('Image Height Distribution')

    axes[2].hist(image_stats_df['aspect_ratio'], bins=20, alpha=0.7)
    axes[2].set_title('Aspect Ratio Distribution')

    axes[3].hist(image_stats_df['mean_pixel'], bins=20, alpha=0.7)
    axes[3].set_title('Mean Pixel Intensity')

    axes[4].hist(image_stats_df['std_pixel'], bins=20, alpha=0.7)
    axes[4].set_title('Pixel Std Dev (Contrast)')

    axes[5].scatter(image_stats_df['width'], image_stats_df['height'], alpha=0.5)
    axes[5].set_title('Image Width vs Height')
    axes[5].set_xlabel('Width')
    axes[5].set_ylabel('Height')

    plt.tight_layout()
    plt.show()


def plot_bbox_statistics(bbox_df):
    """Plot bounding box distributions and interactions."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    axes = axes.ravel()

    axes[0].hist(bbox_df['bbox_area'], bins=30, alpha=0.7)
    axes[0].set_title('Bounding Box Area Distribution')

    axes[1].hist(bbox_df['bbox_aspect_ratio'], bins=30, alpha=0.7)
    axes[1].set_title('Bounding Box Aspect Ratio Distribution')

    axes[2].scatter(bbox_df['bbox_width'], bbox_df['bbox_height'], alpha=0.4)
    axes[2].set_title('BBox Width vs Height')
    axes[2].set_xlabel('Width')
    axes[2].set_ylabel('Height')

    axes[3].hist(bbox_df['rel_area'], bins=30, alpha=0.7)
    axes[3].set_title('Relative BBox Area (to Image)')

    axes[4].scatter(bbox_df['rel_x'], bbox_df['rel_y'], alpha=0.4)
    axes[4].set_title('BBox Center Position Distribution')
    axes[4].set_xlabel('Relative X')
    axes[4].set_ylabel('Relative Y')

    axes[5].scatter(bbox_df['bbox_aspect_ratio'], bbox_df['rel_area'], alpha=0.4)
    axes[5].set_title('BBox Aspect Ratio vs Relative Area')
    axes[5].set_xlabel('BBox Aspect Ratio')
    axes[5].set_ylabel('Rel Area')

    plt.tight_layout()
    plt.show()
