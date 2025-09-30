# Example usage with toy model:
# extractor=FeatureExtractor(model,[model.backbone.features[-1][3][0]])
# activations=extractor(image)[0].cpu()
# activation_np = activations[0][0].squeeze().detach().numpy()

# num_filters = activations.shape[1]  # Number of filters in the layer

# plt.figure(figsize=(15, 10))
# for i in range(min(32, num_filters)):
#     plt.subplot(4, 8, i+1)  # 32 slots
#     sns.heatmap(activations[0, i].detach().numpy(), cmap='viridis', annot=False, cbar=False)
#     plt.title(f'Filter {i+1}')

# plt.suptitle('Activation Heatmaps', fontsize=16)
# plt.show()
# Note that the target layer format can be found by playing around with the model structure

import matplotlib as plt
import numpy as np

class FeatureExtractor():
    # Class for extracting activations
    # The target layer specified directly
    
    def __init__(self, model, target_layers):
        self.model = model
        self.target_layers = target_layers
        self.activations = []

    def save_activation(self, module, input, output):
        self.activations.append(output)

    def clear_activations(self):
        self.activations = []

    def __call__(self, x):  
        self.clear_activations()

        hooks = []
        for layer in self.target_layers:
            hook = layer.register_forward_hook(self.save_activation)
            hooks.append(hook)

        x = self.model(x)

        # Remove hooks after forward pass
        for hook in hooks:
            hook.remove()

        return self.activations
    def plot_and_save_features(self, features, save_path):
        # Convert features to numpy array
        features_np = features.detach().cpu().numpy()
        
        # Plot the features
        plt.imshow(np.squeeze(features_np))
        plt.axis('off')
        plt.savefig(save_path)
        plt.close()

