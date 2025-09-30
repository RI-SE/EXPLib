import json
import pandas as pd
import numpy as np

def compute_metrics(groundtruth_json = "toymodel_output_for_ensemble.json"):
    with open(groundtruth_json, "r") as file: 
        json_data = json.load(file)
    areas = []
    bbox_ratios = []
    anomaly_scores = {"input": [], "output": [], "model": []}

    for item in json_data:
        # Check if any of the anomaly scores are greater than 10
        if (
            item["anomaly_scores"]["input"] > 10
            or item["anomaly_scores"]["output"] > 10
            or item["anomaly_scores"]["model"] > 10
        ):
            continue  # Skip this item if the condition is met

        bbox = item["bbox"]
        width = bbox[2] - bbox[0]
        height = bbox[3] - bbox[1]
        area = width * height
        bbox_ratio = min(width, height) / max(width, height)

        areas.append(area)
        bbox_ratios.append(bbox_ratio)

        anomaly_scores["input"].append(item["anomaly_scores"]["input"])
        anomaly_scores["output"].append(item["anomaly_scores"]["output"])
        anomaly_scores["model"].append(item["anomaly_scores"]["model"])

    # Calculate mean and std for anomaly scores
    metrics = {
        "input": {
            "mean": np.mean(anomaly_scores["input"]),
            "std": np.std(anomaly_scores["input"]),
        },
        "output": {
            "mean": np.mean(anomaly_scores["output"]),
            "std": np.std(anomaly_scores["output"]),
        },
        "model": {
            "mean": np.mean(anomaly_scores["model"]),
            "std": np.std(anomaly_scores["model"]),
        },
        "areas": {
            "mean": np.mean(areas),
            "std": np.std(areas),
        },
        "bbox_ratio": {
            "mean": np.mean(bbox_ratios),
            "std": np.std(bbox_ratios),
        },
    }
    data = {
        "input": anomaly_scores["input"],
        "output": anomaly_scores["output"],
        "model": anomaly_scores["model"],
        "area": areas,
        "bbox_ratio": bbox_ratios,
    }
    df = pd.DataFrame(data)

    # Compute the correlation matrix for next experiments (anomaly scores as functions of bbox)
    corr_matrix = df.corr()

    return metrics, corr_matrix

def toymodel_ensemble(new_item, metrics, z_threshold=2.0):
    # Extract bbox features
    bbox = new_item["bbox"]
    width = bbox[2] - bbox[0]
    height = bbox[3] - bbox[1]
    area = width * height
    bbox_ratio = min(width, height) / max(width, height)

    # Compute z-scores for anomaly scores
    anomaly_scores = new_item["anomaly_scores"]
    z_scores = {
        "input": (anomaly_scores["input"] - metrics["input"]["mean"])
        / metrics["input"]["std"],
        "output": (anomaly_scores["output"] - metrics["output"]["mean"])
        / metrics["output"]["std"],
        "model": (anomaly_scores["model"] - metrics["model"]["mean"])
        / metrics["model"]["std"],
        "area": (area - metrics["areas"]["mean"]) / metrics["areas"]["std"],
        "bbox_ratio": (bbox_ratio - metrics["bbox_ratio"]["mean"])
        / metrics["bbox_ratio"]["std"],
    }

    # Identify anomalies (features where |z| > z_threshold)
    anomalies = {feature: z for feature, z in z_scores.items() if abs(z) > z_threshold}

    # Assess overall anomaly
    is_anomalous = len(anomalies) > 0

    return {
        "z_scores": z_scores,
        "is_anomalous": is_anomalous,
        "anomalies": anomalies,  # Details of where the anomalies are
    }

def example_run_code (new_item, groundtruth_json = "data/toymodel_output_for_ensemble.json"):
    ## this should only be loaded once outside of threading
    metrics, _ = compute_metrics(groundtruth_json)
    
    # This code is using inside the thread to read the output from other module
    result = toymodel_ensemble(new_item, metrics)
    print(result)
    return result