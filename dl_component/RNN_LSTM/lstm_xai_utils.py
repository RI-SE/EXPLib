import torch
from torch import nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn as sns


def integrated_gradients(inputs, model, target_index=None, baseline=None, steps=50):
    """
    Compute integrated gradients for a given input and model.

    """
    if baseline is None:
        baseline = torch.zeros_like(inputs).to(inputs.device)
    
    scaled_inputs = [baseline + (float(i) / steps) * (inputs - baseline) for i in range(steps + 1)]
    grads = []

    for scaled_input in scaled_inputs:
        scaled_input = Variable(scaled_input, requires_grad=True)
        output = model(scaled_input)  # model should return only output tensor
        if target_index is not None:
            output = output[:, :, target_index]  # select the target output
        output.sum().backward()
        grads.append(scaled_input.grad.detach().clone())
    
    avg_grads = torch.mean(torch.stack(grads), dim=0)
    integrated_grads = (inputs - baseline) * avg_grads
    return integrated_grads

class MockLSTM(nn.Module):
    def __init__(self, input_size=10, hidden_size=20, num_layers=2, output_size=5):
        super(MockLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        # x: [seq_len, batch, input_size]
        out, _ = self.lstm(x)           # out: [seq_len, batch, hidden_size]
        out = self.fc(out)              # out: [seq_len, batch, output_size]
        return out

def mockup_input_for_lstm(seq_len=5, batch=3, features=10):
    return torch.randn(seq_len, batch, features)



def plot_integrated_gradients_over_time(ig_tensor, batch_index=0, feature_names=None, title=None):
    """
    Visualize Integrated Gradients over time for a single sequence in the batch.

    """
    # Select the batch element
    ig_sample = ig_tensor[:, batch_index, :]  # [seq_len, input_size]
    seq_len, input_size = ig_sample.shape

    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(input_size)]

    plt.figure(figsize=(10, 6))
    for i in range(input_size):
        plt.plot(ig_sample[:, i].detach().cpu().numpy(), label=feature_names[i])

    plt.title(title or 'Integrated Gradients for Input Features Over Time')
    plt.xlabel('Time Step (Sequence Length)')
    plt.ylabel('Integrated Gradient Value')
    plt.legend(loc='upper right', fontsize=8)
    plt.grid(True)
    plt.show()


def compute_saliency_map(model, input_tensor, hidden_states=None, target_index=None):
    """
    Compute saliency map for RNN/LSTM input.

    """
    input_var = Variable(input_tensor, requires_grad=True)

    # Forward pass
    if hidden_states is not None:
        output = model(input_var, hidden_states)
    else:
        output = model(input_var)

    # If model returns tuple (e.g., LSTM returns (output, (hn, cn)))
    if isinstance(output, tuple):
        output = output[0]  # Take the actual output tensor

    # Select target output
    if target_index is not None:
        target = output[:, :, target_index].sum()
    else:
        target = output.sum()

    # Backward pass
    model.zero_grad()
    target.backward()

    saliency = input_var.grad.detach()
    return saliency


def plot_saliency_map(saliency_tensor, batch_index=0, feature_names=None, title=None):
    """
    Plot saliency map as a heatmap over sequence and features.

    """
    saliency_sample = saliency_tensor[:, batch_index, :]  # [seq_len, input_size]
    seq_len, input_size = saliency_sample.shape

    if feature_names is None:
        feature_names = [f'Feature {i+1}' for i in range(input_size)]

    plt.figure(figsize=(10, 6))
    sns.heatmap(saliency_sample.cpu().numpy(), annot=True, cmap='viridis', xticklabels=feature_names)
    plt.title(title or 'Saliency Map for Input Features Over Time')
    plt.xlabel('Input Feature Dimension')
    plt.ylabel('Time Step (Sequence Length)')
    plt.show()
