import math
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms, models
from EXPLib.xai_library.data_explainers.prototypes.mmd_critic import Dataset, select_prototypes, select_criticisms

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Helper Functions
def prepare_data_loader(image_folder, batch_size=64):
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    ds = datasets.ImageFolder(root=image_folder, transform=transform)
    return torch.utils.data.DataLoader(ds, batch_size=batch_size, shuffle=False)

def initialize_model():
    model = models.mobilenet_v3_large(weights=models.MobileNet_V3_Large_Weights.DEFAULT)
    model.fc = torch.nn.Identity()  # Replace the final layer with identity for embeddings
    return model.to(device).eval()

def extract_patches(image, patch_size=20):
    patches = image.unfold(1, patch_size, patch_size).unfold(2, patch_size, patch_size)
    patches = patches.contiguous().view(3, -1, patch_size, patch_size)
    return patches.permute(1, 0, 2, 3)  # Shape: (num_patches, channels, patch_size, patch_size)

def generate_patches_and_embeddings(dl, model, patch_size=20):
    all_patches, all_patch_embeddings = [], []

    for i, (batch, _) in enumerate(dl, 1):
        if i % 10 == 0:
            print(f"Processed {i} batches…")

        batch = batch.to(device)
        patches_batch = []

        for img in batch:
            patches = extract_patches(img, patch_size)
            patches_batch.append(patches)
            all_patches.append(patches.cpu())

        patches_batch = torch.cat(patches_batch).to(device)
        with torch.no_grad():
            patch_embeddings_batch = model(patches_batch).cpu()

        all_patch_embeddings.append(patch_embeddings_batch)

    return torch.cat(all_patches), torch.cat(all_patch_embeddings)


def compute_kernel(d, kernel_type='local', gamma=None):
    if kernel_type == 'global':
        d.compute_rbf_kernel(gamma)
    elif kernel_type == 'local':
        d.compute_local_rbf_kernel(gamma)
    else:
        raise KeyError('kernel_type must be either "global" or "local"')

def select_and_visualize_prototypes(d, all_patches, num_prototypes, output_dir, make_plots=True):
    if num_prototypes > 0:
        print('Computing prototypes...', end='', flush=True)
        prototype_indices = select_prototypes(d.K, num_prototypes)
        prototypes = all_patches[prototype_indices]
        print('Done.', flush=True)
        print(prototype_indices.sort()[0].tolist())
        if make_plots:
            visualize_patches(prototypes, num_prototypes, 'Prototypes', output_dir + '/' + f'{num_prototypes}_prototypes_patches.svg')

def select_and_visualize_criticisms(d, all_patches, prototype_indices, num_criticisms, output_dir, regularizer='logdet', make_plots=True):
    if num_criticisms > 0:
        print('Computing criticisms...', end='', flush=True)
        criticism_indices = select_criticisms(d.K, prototype_indices, num_criticisms, regularizer)
        criticisms = all_patches[criticism_indices]
        print('Done.', flush=True)
        print(criticism_indices.sort()[0].tolist())
        if make_plots:
            visualize_patches(criticisms, num_criticisms, 'Criticisms', output_dir +'/' + f'{num_criticisms}_criticisms_patches.svg')

def visualize_patches(patches, num_items, title, output_path):
    num_cols = 8
    num_rows = math.ceil(num_items / num_cols)
    fig, axes = plt.subplots(num_rows, num_cols, figsize=(6, num_rows * 0.75))
    for i, axis in enumerate(axes.ravel()):
        if i >= num_items:
            axis.axis('off')
            continue
        img = patches[i][0].numpy()
        axis.imshow(img, cmap='gray')
        axis.axis('off')
    fig.suptitle(f'{num_items} {title}')
    plt.savefig(output_path)
    plt.show(fig)
    plt.close(fig)

def run_analysis_pipeline(image_folder, output_dir, patch_size=20, batch_size=64,
                          num_prototypes=32, num_criticisms=10, kernel_type='local',
                          regularizer='logdet', gamma=None, make_plots=True):
    # Prepare data loader
    dl = prepare_data_loader(image_folder, batch_size)
    
    # Initialize model for embeddings
    model = initialize_model()
    
    # Generate patches and embeddings
    all_patches, all_patch_embeddings = generate_patches_and_embeddings(dl, model, patch_size)
    
    # Setup dummy labels and dataset
    X = all_patch_embeddings
    y = torch.zeros((X.shape[0],), dtype=torch.long)
    d = Dataset(X, y)
    
    # Compute kernel for MMD-critic
    compute_kernel(d, kernel_type, gamma)
    
    # Select and visualize prototypes
    select_and_visualize_prototypes(d, all_patches, num_prototypes, output_dir, make_plots)
    
    # Select and visualize criticisms
    prototype_indices = select_prototypes(d.K, num_prototypes)
    select_and_visualize_criticisms(d, all_patches, prototype_indices, num_criticisms, output_dir, regularizer, make_plots)

# Example usage
# run_analysis_pipeline(Path('./images2'), Path('./output'))
