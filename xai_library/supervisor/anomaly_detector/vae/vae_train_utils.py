import torch.nn.functional as F
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from PIL import Image
import os
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np

# Define constants
RESIZED_WIDTH = 160
RESIZED_HEIGHT = 160
LATENT_DIM = 256
BATCH_SIZE = 100
NUM_WORKERS = 8
LEARNING_RATE = 0.001
STEP_SIZE = 5
GAMMA = 0.5
NUM_EPOCHS = 100
INPUT_DATASET_PATH = '../../../../datasets/Camera/toy_model_v1_inference/dev/images'
CROP_PREDICTION_DATASET_PATH = '../../../../datasets/Camera/toy_model_v1_inference/pred_cropped'
MODEL_SAVE_PATH = './input_img_vae_weight'
LATENT_SAVE_PATH = './latent_vectors'
TEST_DATA_PATH = '../vaetest'
INPUT_MODEL_WEIGHT = 'vae_model_img20250214.pth'
OUTPUT_MODEL_WEIGHT = 'vae_model_output20250214.pth'

# Set device (GPU if available, otherwise CPU)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# Create a custom dataset class for image loading
class CustomDataset(Dataset):
    def __init__(self, folder, transform=None, img_extension='.png'):
        self.image_files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith(img_extension)]
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
    def __init__(self, input_shape=(3, 120, 160), latent_dim=1024):
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
    from tqdm import tqdm
    model.train()
    for epoch in range(1, num_epochs + 1):
        train_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs}", leave=True)

        for batch_idx, data in enumerate(progress_bar):
            data = data.to(device)
            optimizer.zero_grad()
            recon_batch, mu, log_var = model(data)
            loss = vae_loss(recon_batch, data, mu, log_var)
            loss.backward()
            train_loss += loss.item()
            optimizer.step()

            # Update progress bar with loss info
            progress_bar.set_postfix(loss=loss.item())

        average_loss = train_loss / len(train_loader.dataset)
        scheduler.step(average_loss)  # Update learning rate
        print(f'Epoch {epoch}/{num_epochs} - Average loss: {average_loss:.6f}')

def load_vae_model(model, model_path):
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def detect_anomaly(model, image_path):
    transform = transforms.Compose([
        transforms.Resize((RESIZED_HEIGHT, RESIZED_WIDTH)),
        transforms.ToTensor()
    ])
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    with torch.no_grad():
        reconstructed_image, _, _ = model(image_tensor)
    
    mse_loss = nn.functional.mse_loss(reconstructed_image, image_tensor)
    return reconstructed_image.cpu().squeeze(0), mse_loss.item()

def visualize_anomalies(model, data_folder):
    img_extension = ('.jpg', '.png')
    image_files = [f for f in os.listdir(data_folder) if f.endswith(img_extension)]
    
    plt.figure(figsize=(12, 3 * len(image_files[:6])))

    for i, image_file in enumerate(image_files[:6]):
        image_path = os.path.join(data_folder, image_file)
        reconstructed_image, anomaly_score = detect_anomaly(model, image_path)
        original_image = Image.open(image_path).convert("RGB")

        plt.subplot(len(image_files), 3, i * 3 + 1)
        plt.title(f"Original Image ({image_file})")
        plt.imshow(original_image)
        plt.axis("off")

        plt.subplot(len(image_files), 3, i * 3 + 2)
        plt.title(f"Reconstructed Image ({image_file})")
        plt.imshow(reconstructed_image.permute(1, 2, 0))
        plt.axis("off")

        plt.subplot(len(image_files), 3, i * 3 + 3)
        plt.title(f"Anomaly Score: {anomaly_score:.4f}")
        plt.axis("off")

    plt.show()

def compute_latent_vectors(model, data_folder, transform, latent_path=LATENT_SAVE_PATH, 
                           mu_filename="mu_array.npy", log_var_filename="log_var_array.npy"):
    """
    Computes and saves latent vectors (mu, log_var) for images in a folder.

    Parameters:
    - model: Trained VAE model.
    - data_folder: Path to the image folder.
    - transform: Preprocessing transformations for images.
    - latent_path: Directory to save latent vector files .
    - mu_filename: Name of the saved mean latent vector file (default: 'mu_array.npy').
    - log_var_filename: Name of the saved log variance latent vector file (default: 'log_var_array.npy').
    """
    
    os.makedirs(latent_path, exist_ok=True)  # Ensure save directory exists
    
    img_extension = ('.jpg', '.png')
    image_files = [f for f in os.listdir(data_folder) if f.endswith(img_extension)]
    
    mu_list, log_var_list = [], []
    
    for image_file in tqdm(image_files, desc="Processing images"):
        image_path = os.path.join(data_folder, image_file)
        image = Image.open(image_path).convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(device)
        _, mu, log_var = model(image_tensor)
        
        mu_list.append(mu.cpu().detach().numpy().flatten())
        log_var_list.append(log_var.cpu().detach().numpy().flatten())
    
    np.save(os.path.join(latent_path, mu_filename), np.array(mu_list))
    np.save(os.path.join(latent_path, log_var_filename), np.array(log_var_list))

    print(f"Saved latent vectors to {latent_path}/{mu_filename} and {latent_path}/{log_var_filename}")

def main():
    is_input_dataset = True
    train_mode = True
    compute_latent = False # Compute laten only when train_mode is False
    
    transform = transforms.Compose([
        transforms.Resize((RESIZED_HEIGHT, RESIZED_WIDTH)),
        transforms.ToTensor()
    ])

    dataset_folder = INPUT_DATASET_PATH if is_input_dataset else CROP_PREDICTION_DATASET_PATH
    custom_dataset = CustomDataset(dataset_folder, transform=transform)

    train_loader = DataLoader(
        custom_dataset, 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS,
        pin_memory=True,  
        persistent_workers=True,  
        prefetch_factor=4
    )

    # Initialize VAE model
    vae_model = VAE(input_shape=(3, RESIZED_HEIGHT, RESIZED_WIDTH), latent_dim=LATENT_DIM).to(device)
    vae_optimizer, vae_scheduler = create_optimizer_and_scheduler(vae_model, lr=LEARNING_RATE, step_size=STEP_SIZE, gamma=GAMMA)
    model_path =  os.path.join(MODEL_SAVE_PATH, INPUT_MODEL_WEIGHT if is_input_dataset else OUTPUT_MODEL_WEIGHT)

    if train_mode:
        train_vae(vae_model, train_loader, vae_optimizer, vae_scheduler, num_epochs=NUM_EPOCHS)
        torch.save(vae_model.state_dict(), model_path)
    else:
        loaded_model = load_vae_model(vae_model, model_path)
        visualize_anomalies(loaded_model, TEST_DATA_PATH)
        if compute_latent:
            compute_latent_vectors(loaded_model, INPUT_DATASET_PATH)

if __name__ == "__main__":
    main()