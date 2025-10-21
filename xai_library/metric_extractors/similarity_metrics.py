import os
import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
from typing import Tuple
import cv2
import torchvision


def compute_stats(feats: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Compute mean and covariance."""
    mu = feats.mean(axis=0)
    if feats.shape[0] < 2:
        sigma = np.zeros((feats.shape[1], feats.shape[1]), dtype=np.float32)
    else:
        sigma = np.cov(feats, rowvar=False)
    return mu, sigma


def matrix_sqrt(mat: np.ndarray):
    eigvals, eigvecs = np.linalg.eigh(mat)
    eigvals = np.clip(eigvals, a_min=0, a_max=None)
    sqrt_eigvals = np.sqrt(eigvals)
    return eigvecs @ np.diag(sqrt_eigvals) @ eigvecs.T


def compute_fvd(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    
    ### Fréchet Video Distance (FVD) between video features computed from two video datasets (e.g. I3D embeddings)
    
    mu_fake, sigma_fake = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)

    mean_diff = mu_fake - mu_real
    mean_dist = np.dot(mean_diff, mean_diff)

    cov_prod = sigma_fake @ sigma_real
    covmean = matrix_sqrt(cov_prod)

    if not np.isfinite(covmean).all():
        covmean = np.nan_to_num(covmean)

    fvd = mean_dist + np.trace(sigma_fake + sigma_real - 2 * covmean)

    return float(np.real(fvd))


def read_video_frames(video_path: str, max_frames: int = None, resize: tuple = None):
    ## Read video frames from file to np array
    cap = cv2.VideoCapture(video_path)
    frames = []
    count = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  
        
        if resize:
            frame = cv2.resize(frame, resize)
        
        frame = frame.astype(np.float32) / 255.0  
        
        frames.append(frame)
        count += 1

        if max_frames and count >= max_frames:
            break
    
    cap.release()
    return np.array(frames)


def load_videos_from_folder(folder_path: str, max_videos: int = None, **kwargs):
    ## Load mp4 videos from a folder.
    
    video_files = [os.path.join(folder_path, f) 
                   for f in os.listdir(folder_path) 
                   if f.lower().endswith(".mp4")]
    
    if max_videos:
        video_files = video_files[:max_videos]
    
    videos = []
    for path in video_files:
        frames = read_video_frames(path, **kwargs)
        if frames.size > 0:
            videos.append(frames)
    
    return videos

def build_video_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    ## Load pretrained R(2+1)D-18 model and remove the classification head for computing embeddings
    model = torchvision.models.video.r2plus1d_18(weights="KINETICS400_V1")
    model = nn.Sequential(*list(model.children())[:-1])  # remove classification head
    model.eval().to(device)
    return model


def extract_video_embedding(model, video: np.ndarray, device="cuda"):
    ## Extract a feature embedding from one video ([T, H, W, 3]).
    
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((112, 112)),
        T.Normalize(mean=[0.43216, 0.394666, 0.37645],
                    std=[0.22803, 0.22145, 0.216989]),
    ])

    # Apply to each frame
    frames = torch.stack([transform(frame) for frame in video])  # [T, 3, H, W]
    frames = frames.permute(1, 0, 2, 3).unsqueeze(0).to(device)  # [1, 3, T, H, W]

    with torch.no_grad():
        feat = model(frames)
    
    return feat.squeeze().cpu().numpy()  # [512]


def extract_embeddings_for_videos(model, videos, device="cuda"):
    
    ### Extract embeddings for a list of videos.  Returns array [N, D]
    
    embeddings = []
    for vid in videos:
        emb = extract_video_embedding(model, vid, device=device)
        embeddings.append(emb)
    return np.stack(embeddings)


def load_images_from_folder(folder: str, max_images=None, image_size=(299, 299)):
    transform = T.Compose([
        T.Resize(image_size),
        T.CenterCrop(image_size),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406],
                    std=[0.229, 0.224, 0.225]),
    ])
    images = []
    files = [os.path.join(folder, f) for f in os.listdir(folder)
             if f.lower().endswith((".jpg", ".jpeg", ".png"))]
    if max_images:
        files = files[:max_images]

    for f in files:
        try:
            img = Image.open(f).convert("RGB")
            images.append(transform(img))
        except Exception as e:
            print(f"Skipping {f}: {e}")
    return images


def build_inception_model(device="cuda" if torch.cuda.is_available() else "cpu"):
    ## Load pretrained InceptionV3 for image feature extraction.
    model = torchvision.models.inception_v3(weights="IMAGENET1K_V1", transform_input=False)
    model.fc = nn.Identity()  # remove classification head
    model.eval().to(device)
    return model


def extract_image_features(model, images, device="cuda", batch_size=16):
    features = []
    with torch.no_grad():
        for i in range(0, len(images), batch_size):
            batch = torch.stack(images[i:i+batch_size]).to(device)
            emb = model(batch)
            features.append(emb.cpu().numpy())
    return np.concatenate(features, axis=0)  # [N, D]



def compute_fid(feats_fake: np.ndarray, feats_real: np.ndarray) -> float:
    mu_fake, sigma_fake = compute_stats(feats_fake)
    mu_real, sigma_real = compute_stats(feats_real)
    diff = mu_fake - mu_real
    covmean = matrix_sqrt(sigma_fake @ sigma_real)
    fid = diff.dot(diff) + np.trace(sigma_fake + sigma_real - 2 * covmean)
    return float(np.real(fid))


def compute_mmd(feats_fake: np.ndarray, feats_real: np.ndarray, sigma=1.0):
    ### Maximum Mean Discrepancy (Gaussian kernel).
    def kernel(x, y):
        xx = np.sum(x ** 2, axis=1, keepdims=True)
        yy = np.sum(y ** 2, axis=1, keepdims=True)
        xy = np.dot(x, y.T)
        dist = xx + yy.T - 2 * xy
        return np.exp(-dist / (2 * sigma ** 2))

    k_xx = kernel(feats_fake, feats_fake)
    k_yy = kernel(feats_real, feats_real)
    k_xy = kernel(feats_fake, feats_real)
    mmd = k_xx.mean() + k_yy.mean() - 2 * k_xy.mean()
    return float(mmd)


def compute_kid(feats_fake: np.ndarray, feats_real: np.ndarray, num_subsets=100, subset_size=1000):
    ### Kernel Inception Distance (unbiased MMD)
    rng = np.random.default_rng(42)
    mmds = []
    n_fake, n_real = len(feats_fake), len(feats_real)
    for _ in range(num_subsets):
        idx_fake = rng.choice(n_fake, subset_size, replace=n_fake < subset_size)
        idx_real = rng.choice(n_real, subset_size, replace=n_real < subset_size)
        mmds.append(compute_mmd(feats_fake[idx_fake], feats_real[idx_real]))
    return float(np.mean(mmds))


def compare_image_folders(real_folder, fake_folder, max_images=None, device="cuda"):
    ## Compute FID, KID, and MMD for two image folders.
    print("Loading images...")
    real_imgs = load_images_from_folder(real_folder, max_images=max_images)
    fake_imgs = load_images_from_folder(fake_folder, max_images=max_images)
    print(f"Loaded {len(real_imgs)} images from {real_folder} and {len(fake_imgs)} images from {fake_folder}.")

    model = build_inception_model(device=device)

    print("Extracting features...")
    real_feats = extract_image_features(model, real_imgs, device=device)
    fake_feats = extract_image_features(model, fake_imgs, device=device)

    print("Computing metrics...")
    fid = compute_fid(fake_feats, real_feats)
    kid = compute_kid(fake_feats, real_feats)
    mmd = compute_mmd(fake_feats, real_feats)

    return {"FID": fid, "KID": kid, "MMD": mmd}

def compare_video_folders(real_folder, fake_folder, max_videos=5, max_frames=32, resize=(224, 224), device="cuda"):
    real_videos = load_videos_from_folder(real_folder, max_videos=max_videos, max_frames=max_frames, resize=resize)
    fake_videos = load_videos_from_folder(fake_folder, max_videos=max_videos, max_frames=max_frames, resize=resize)

    print(f"Loaded {len(real_videos)} videos from {real_folder} and {len(fake_videos)} videos from {fake_folder}.")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_video_model(device=device)

    real_feats = extract_embeddings_for_videos(model, real_videos, device=device)
    fake_feats = extract_embeddings_for_videos(model, fake_videos, device=device)

    fvd_score = compute_fvd(fake_feats, real_feats)
    return {"FVD": fvd_score}
