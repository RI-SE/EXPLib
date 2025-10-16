
import os
import json
from tqdm import tqdm

import numpy as np
import pandas as pd
from PIL import Image
from skimage.color import rgb2gray
from skimage.feature import local_binary_pattern, hog
from skimage.filters import sobel
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.inspection import PartialDependenceDisplay, partial_dependence
import matplotlib.pyplot as plt
from PyALE import ale

import xgboost as xgb
from sklearn.base import BaseEstimator, RegressorMixin

NUM_HOG_BINS = 16
NUM_LBP_BINS = 16
NUM_EDGE_BINS = 16
SEED = 42

def get_feature_names():
    feature_names = (
            ["brightness", "contrast"]
            + [f"edge_bin_{i}" for i in range(NUM_EDGE_BINS)]
            + [f"lbp_bin_{i}" for i in range(NUM_LBP_BINS)]
            + [f"hog_bin_{i}" for i in range(NUM_HOG_BINS)]
        )
    return feature_names

def load_json(path):
    with open(path, 'r') as f:
        return json.load(f)

def save_json(obj, path):
    with open(path, 'w') as f:
        json.dump(obj, f, indent=2)

def compute_features_for_crop(img_pil):
    """
    img_pil: PIL.Image (RGB)
    returns: dict with brightness, edge_hist (list), lbp_hist (list), hog_hist (list)
    """
    # Resize to small fixed size for consistent features (keeps textures)
    crop = img_pil.resize((128, 128))
    arr = np.array(crop).astype(np.float32) / 255.0  
    gray = rgb2gray(arr)  

    brightness = arr.mean()
    contrast = arr.std()

    # Sobel edge histogram 
    edge_mag = sobel(gray)  
    edge_hist, _ = np.histogram(edge_mag.ravel(), bins=NUM_EDGE_BINS, range=(0.0, 1.0))
    if edge_hist.sum() > 0:
        edge_hist = edge_hist.astype(float) / edge_hist.sum()
    else:
        edge_hist = np.zeros_like(edge_hist, dtype=float)

    # LBP histogram
    lbp = local_binary_pattern(gray, P=8, R=1, method="uniform")
    lbp_hist, _ = np.histogram(lbp.ravel(), bins=NUM_LBP_BINS, range=(0, NUM_LBP_BINS))
    if lbp_hist.sum() > 0:
        lbp_hist = lbp_hist.astype(float) / lbp_hist.sum()
    else:
        lbp_hist = np.zeros_like(lbp_hist, dtype=float)

    # HOG feature histogram
    hog_feats = hog(gray,
                    pixels_per_cell=(16, 16),
                    cells_per_block=(2, 2),
                    visualize=False,
                    feature_vector=True)
    hog_hist, _ = np.histogram(hog_feats.ravel(), bins=NUM_HOG_BINS)
    if hog_hist.sum() > 0:
        hog_hist = hog_hist.astype(float) / hog_hist.sum()
    else:
        hog_hist = np.zeros_like(hog_hist, dtype=float)

    # Concatenated feature
    feats = {
        "brightness": float(brightness),
        "contrast": float(contrast),
        "edge_hist": edge_hist.tolist(),
        "lbp_hist": lbp_hist.tolist(),
        "hog_hist": hog_hist.tolist()
    }
    return feats

def compute_and_append_features(json_path, images_dir, out_json_path):
    data = load_json(json_path)

    images_list = data.get("images", [])
    annotations = data.get("annotations", None)

    if annotations is None:
        if isinstance(data, list):
            annotations = data
        else:
            raise ValueError("JSON must contain 'annotations' list or be a list of annotations.")

    id2file = {}
    for img in images_list:
        if 'id' in img and 'file_name' in img:
            id2file[img['id']] = img['file_name']

    np.random.seed(SEED)

    num_done = 0
    for ann in tqdm(annotations, desc="processing annotations"):
        # Skip if already computed
        if 'computed_features' in ann:
            num_done += 1
            continue

        image_id = ann.get("image_id") or ann.get("imageId") or ann.get("image")
        if image_id is None:
            print("Warning: annotation without image_id; skipping")
            continue
        file_name = id2file.get(image_id)
        if file_name is None:
            file_name = ann.get("file_name")
        if file_name is None:
            print(f"Warning: no filename for image_id {image_id}; skipping")
            continue

        image_path = os.path.join(images_dir, file_name)
        if not os.path.exists(image_path):
            print(f"Warning: image file not found: {image_path}; skipping")
            continue

        # get predicted bbox
        bbox = ann.get("pred_bbox") or ann.get("pred_box") or ann.get("bbox")
        if bbox is None:
            print(f"Warning: no bbox for annotation id {ann.get('id')}; skipping")
            continue

        # bbox format expected [x1,y1,x2,y2] 
        x1, y1, x2, y2 = bbox[:4]
        
        try:
            with Image.open(image_path) as img:
                crop = img.crop((x1, y1, x2, y2)).convert("RGB")
                feats = compute_features_for_crop(crop)
                ann['computed_features'] = feats
                num_done += 1
        except Exception as e:
            print(f"Error processing image {image_path} annotation {ann.get('id')}: {e}")
            continue

    # Save augmented JSON (with computed_features)
    if images_list:
        out_obj = {"images": images_list, "annotations": annotations}
    else:
        out_obj = annotations

    save_json(out_obj, out_json_path)
    print(f"Done. Computed features for {num_done} annotations. Saved to {out_json_path}")
    return out_json_path

def build_dataset_from_json(json_with_features):
    data = load_json(json_with_features)
    annotations = data.get("annotations", data if isinstance(data, list) else [])

    rows = []
    for ann in annotations:
        feats = ann.get("computed_features")
        if feats is None:
            continue
        # flatten histogram features 
        edge = np.array(feats["edge_hist"], dtype=float)
        lbp = np.array(feats["lbp_hist"], dtype=float)
        hogh = np.array(feats["hog_hist"], dtype=float)
        brightness = float(feats.get("brightness", 0.0))
        contrast = float(feats.get("contrast", 0.0))
        vec = np.concatenate([[brightness, contrast], edge, lbp, hogh])
        rows.append({
            "annotation_id": ann.get("id"),
            "image_id": ann.get("image_id"),
            "feature_vector": vec,
            "pred_score": float(ann.get("pred_score", ann.get("score", 0.0)))
        })

    df = pd.DataFrame(rows)
    if df.empty:
        raise RuntimeError("No annotations with computed_features found.")
    X = np.vstack(df['feature_vector'].values)
    y = df['pred_score'].values
    return df, X, y

def train_xgb_surrogate(X, y):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=SEED)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "seed": SEED,
        "verbosity": 1
    }
    num_round = 200
    evallist = [(dtrain, 'train'), (dtest, 'eval')]
    bst = xgb.train(params, dtrain, num_round, evals=evallist, early_stopping_rounds=20, verbose_eval=False)
    
    y_pred_train = bst.predict(dtrain)
    y_pred_test = bst.predict(dtest)

    print("Train R2:", r2_score(y_train, y_pred_train))
    print("Test  R2:", r2_score(y_test, y_pred_test))
    print("Test MAE:", mean_absolute_error(y_test, y_pred_test))

    return bst, X_train, X_test, y_train, y_test

class BoosterWrapper(BaseEstimator, RegressorMixin):
    from sklearn.base import BaseEstimator, RegressorMixin
    """Sklearn-compatible wrapper around an XGBoost Booster."""
    def __init__(self, booster):
        self.booster = booster
        self._estimator_type = "regressor"

    def fit(self, X, y=None):
        self.n_features_in_ = X.shape[1]
        self.is_fitted_ = True
        return self

    def predict(self, X):
        if not hasattr(self, "is_fitted_"):
            raise RuntimeError("BoosterWrapper must be 'fit' first.")
        d = xgb.DMatrix(X)
        return self.booster.predict(d)

class ALEModelWrapper:
    def __init__(self, booster):
        import xgboost as xgb
        self.booster = booster
    def predict(self, X):
        import xgboost as xgb
        d = xgb.DMatrix(X)
        return self.booster.predict(d)
    

def feature_importance_dict(
        booster: xgb.Booster,
        feature_names: list[str],
        importance_type: str = "gain"
    ) -> np.ndarray:
    """
    Convert the dictionary returned by `booster.get_score()` into a NumPy array
    aligned with `feature_names`.
    """
    fmap = booster.get_score(importance_type=importance_type)
    imp = np.zeros(len(feature_names))
    for k, v in fmap.items():
        # Keys are either "f0", "f1", … or simply "0", "1", …
        idx = int(k[1:]) if k.startswith("f") else int(k)
        if idx < len(imp):
            imp[idx] = v
    return imp

def plot_feature_importances(
        feature_names: list[str],
        importances: np.ndarray,
        top_k: int = 6,
        figsize: tuple[int, int] = (8, 5)
    ) -> None:
    """Bar plot of the top-k feature importances."""
    idx = np.argsort(importances)[::-1][:top_k]
    plt.figure(figsize=figsize)
    plt.bar([feature_names[i] for i in idx], importances[idx])
    plt.xticks(rotation=45, ha="right")
    plt.title("Top feature importances (XGBoost gain)")
    plt.tight_layout()
    plt.show()

def plot_partial_dependence(
        estimator: BoosterWrapper,
        X: np.ndarray,
        feature_names: list[str],
        top_features: list[str],
        grid_resolution: int = 20,
        figsize: tuple[int, int] = (4, 4)
    ) -> None:

    n = len(top_features)
    if n == 0:
        return
    fig, axes = plt.subplots(1, n, figsize=(figsize[0] * n, figsize[1]))
    if n == 1:
        axes = [axes]          

    for ax, feat in zip(axes, top_features):
        PartialDependenceDisplay.from_estimator(
            estimator=estimator,
            X=X,
            features=[feat],
            feature_names=feature_names,
            ax=ax,
            grid_resolution=grid_resolution,
        )
    plt.tight_layout()
    plt.show()

def plot_ale(
        X_df: pd.DataFrame,
        ale_model: ALEModelWrapper,
        feature_names: list[str],
        top_features: list[str],
        grid_size: int = 15
    ) -> None:
    for feat_name in top_features:
        ale_eff = ale(X_df, model=ale_model, feature=[feat_name], grid_size=grid_size)
