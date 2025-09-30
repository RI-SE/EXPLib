import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import torchvision.transforms as T
import numpy as np
from sklearn.decomposition import NMF
from sklearn.preprocessing import normalize
from sklearn.preprocessing import MinMaxScaler
import cv2

def list_images_recursive(folder, img_extensions):
    """
    Lists all image files recursively within a folder and its subfolders.

    Args:
        folder (str): The path to the root folder.
        img_extensions (list): A list of image file extensions (e.g., ['.jpg', '.png']).

    Returns:
        list: A list of paths to image files.
    """
    image_files = []
    for root, _, files in os.walk(folder):
        for f in files:
            if any(f.endswith(ext) for ext in img_extensions):
                image_files.append(os.path.join(root, f))
    return image_files

# Create a custom dataset class for image loading
class CustomDataset(Dataset):
    def __init__(self, folder, transform=None, img_extensions=['.png', '.jpg', '.jpeg', '.bmp']):
        self.image_files =  list_images_recursive(folder, img_extensions)
        self.transform = transform
    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert("RGB")
        if self.transform:
            image = self.transform(image)
        return image

class VAE(nn.Module):
    def __init__(self, input_shape=(3, 160, 160), latent_dim=1024):
        super(VAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(256 * (input_shape[1] // 8) * (input_shape[2] // 8), 512), #replace 512 to 2048
            nn.ReLU(),
            nn.Linear(512, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 256 * (input_shape[1] // 8) * (input_shape[2] // 8)),
            nn.ReLU(),
            nn.Unflatten(1, (256, input_shape[1] // 8, input_shape[2] // 8)),
            nn.ConvTranspose2d(256, 128, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  
            nn.ReLU(),
            nn.ConvTranspose2d(64, input_shape[0], kernel_size=4, stride=2, padding=1),  
            nn.Sigmoid()
        )
        self.activations = None
        self.gradients = None
    
    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def save_gradients(self, grad):
        self.gradients = grad

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, log_var = torch.chunk(mu_logvar, 2, dim=1)
        z = self.reparameterize(mu, log_var)

        self.activations = self.decoder[6](z)
        self.activations.requires_grad = True
        self.activations.register_hook(self.save_gradients)
        
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var
    
    def get_activations_gradient(self):
        return self.gradients

    def get_activations(self):
        return self.activations

def generate_gradcam_heatmap(model, input_image, transform):
    device = next(model.parameters()).device
    image_tensor = transform(input_image).unsqueeze(0).to(device)
    
    model.eval()
    with torch.no_grad():
        reconstructed_image, mu, log_var = model(image_tensor)
    
    # Re-enable gradient calculation
    model.train()
    image_tensor.requires_grad = True
    
    loss = vae_loss(reconstructed_image, image_tensor, mu, log_var)
    
    model.zero_grad()
    loss.backward(retain_graph=True)
    
    gradients = model.get_activations_gradient()
    pooled_gradients = torch.mean(gradients, dim=[0, 2, 3])
    
    activations = model.get_activations().detach()
    
    for i in range(activations.shape[1]):
        activations[:, i, :, :] *= pooled_gradients[i]
    
    heatmap = torch.mean(activations, dim=1).squeeze()
    heatmap = F.relu(heatmap)
    
    heatmap -= heatmap.min()
    heatmap /= heatmap.max()
    
    heatmap = heatmap.cpu().numpy()
    
    return heatmap

class VAE_small(nn.Module):
    def __init__(self, input_shape=(1, 20, 20), latent_dim=10):
        super(VAE_small, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_shape[0], 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(64 * (input_shape[1] // 4) * (input_shape[2] // 4), 128),
            nn.ReLU(),
            nn.Linear(128, latent_dim * 2)
        )

        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 64 * (input_shape[1] // 4) * (input_shape[2] // 4)),
            nn.ReLU(),
            nn.Unflatten(1, (64, input_shape[1] // 4, input_shape[2] // 4)),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(32, input_shape[0], kernel_size=4, stride=2, padding=1),
            nn.Sigmoid()
        )

        self.latent_dim = latent_dim

    def reparameterize(self, mu, log_var):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def forward(self, x):
        mu_logvar = self.encoder(x)
        mu, log_var = torch.chunk(mu_logvar, 2, dim=1)
        z = self.reparameterize(mu, log_var)
        x_reconstructed = self.decoder(z)
        return x_reconstructed, mu, log_var

def vae_loss(recon_x, x, mu, log_var):
    criterion = nn.MSELoss(reduction='sum')
    BCE = criterion(recon_x, x)
    KLD = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    return BCE + KLD

def create_optimizer_and_scheduler(model, lr=0.001, step_size=5, gamma=0.5):
    optimizer = optim.Adam(model.parameters(), lr=lr)
    #scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min')
    return optimizer, scheduler

def train_vae(model, train_loader, optimizer, scheduler, num_epochs=10):
    model.train()
    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        for batch_idx, data in enumerate(train_loader):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss = vae_loss(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

        average_loss = train_loss / len(train_loader.dataset)
        scheduler.step(average_loss)  # Update learning rate
        print(f'Epoch {epoch}/{num_epochs} - Average loss: {average_loss:.6f}')


def detect_anomaly(model, image_path, resized_height=160, resized_width=160):
    transform = transforms.Compose([
        transforms.Resize((resized_height, resized_width)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    device = next(model.parameters()).device
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed_image, _, _ = model(image_tensor)
    
    mse_loss = nn.functional.mse_loss(reconstructed_image, image_tensor)
    return reconstructed_image.cpu().squeeze(0), mse_loss.item()



def load_vae_model(model, model_path, device):
    # Map storage to the CPU if CUDA is not available
    map_location = torch.device('cpu') if not torch.cuda.is_available() else None
    model.load_state_dict(torch.load(model_path, map_location=map_location))
    model.eval()
    model.to(device)
    return model


# For computing NMF prototypes
def find_prototype_image_indices(W, top_k=3):
    """
    Find the indices of images that best match each prototype.

    Args:
        W (np.array): NMF weight matrix (num_samples x num_prototypes)
        top_k (int): Number of top images per prototype to retrieve

    Returns:
        dict: Dictionary mapping prototype index to a list of top-k image indices
    """
    num_prototypes = W.shape[1]
    prototype_indices = {}

    for proto_idx in range(num_prototypes):
        top_images_idx = np.argsort(-W[:, proto_idx])[:top_k]  # Find top-k images
        prototype_indices[proto_idx] = top_images_idx.tolist()

    return prototype_indices

def find_image_filename(coco_annotations, image_index):
    image_info = coco_annotations["images"][image_index]
    return image_info["file_name"]

def find_latent_prototypes(mu_vectors, num_prototypes=10):
    nmf = NMF(n_components=num_prototypes, init='nndsvd', random_state=42)
    W = nmf.fit_transform(mu_vectors)  # Weight matrix (importance of each prototype)
    H = nmf.components_  # Basis components (prototypes in latent space)
    return W, H

def ensure_positive_latent(mu_vectors):
    scaler = MinMaxScaler(feature_range=(0, 1))  # Shift to [0,1]
    return scaler.fit_transform(mu_vectors)


def find_activated_latent_regions(H_array, latent_shape=(256, 16, 16), patch_size_factor=2, top_k=3):
    """
    Identifies the most activated regions in the latent space for a given patch size factor.

    Args:
        H (np.array): NMF components (num_prototypes x latent_dim)
        latent_shape (tuple): (C, H, W) shape of the latent space
        patch_size_factor (int): Factor to scale the patch size (e.g., 2 for 20x20, 3 for 30x30)
        top_k (int): Number of most activated latent regions to retrieve

    Returns:
        dict: {prototype_idx: [(channel, y, x), ...] }
    """
    C, H, W = latent_shape
    num_prototypes = H_array.shape[0]
    activated_regions = {}

    patch_size = 10 * patch_size_factor  # Actual patch size in image coordinates (e.g., 20, 30)

    for proto_idx in range(num_prototypes):
        top_latent_indices = np.argsort(-H_array[proto_idx])[:top_k]

        # Map 1D indices to (channel, y, x) coordinates
        top_regions = [(idx // (H * W), (idx % (H * W)) // W, (idx % (H * W)) % W) for idx in top_latent_indices]
        activated_regions[proto_idx] = top_regions

    return activated_regions

def map_latent_to_image(latent_coords, image_size=160, feature_map_size=16, patch_size_factor=2):
    patch_size = 10 * patch_size_factor  # Actual patch size
    scale_factor = image_size // feature_map_size  # Mapping latent space to image space
    mapped_positions = [(y * scale_factor, x * scale_factor) for _, y, x in latent_coords]

    # Ensure patches are within bounds
    adjusted_positions = [
        (max(0, min(y, image_size - patch_size)), max(0, min(x, image_size - patch_size))) for y, x in mapped_positions
    ]
    return adjusted_positions

def extract_prototype_patches(image_filename, patch_positions, patch_size=20):
    if not os.path.exists(image_filename):
        print(f" ERROR: Image file not found: {image_filename}")
        return []

    image = cv2.imread(image_filename)
    
    if image is None:
        print(f" ERROR: Failed to read image: {image_filename}")
        return []

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)  # Convert to RGB

    patches = []
    for y, x in patch_positions:
        if 0 <= y < image.shape[0] - patch_size and 0 <= x < image.shape[1] - patch_size:
            patches.append(image[y:y+patch_size, x:x+patch_size])
        else:
            print(f" WARNING: Patch ({y},{x}) is out of bounds for image {image_filename}")

    return patches