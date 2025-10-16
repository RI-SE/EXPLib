import torch
import torch.nn as nn

class NeuronCoverageBase:
    def __init__(self, model):
        self.model = model.eval()
        self.neuron_activations = {}
        self.hooks = []
        self._register_hooks()

    def _register_hooks(self):
        for name, layer in self.model.named_modules():
            # Register only on convolutional or linear layers for simplicity
            if isinstance(layer, (nn.Conv2d, nn.Linear)):
                hook = layer.register_forward_hook(self._capture_activation(name))
                self.hooks.append(hook)

    def _capture_activation(self, layer_name):
        def hook_fn(module, input, output):
            if isinstance(output, torch.Tensor):
                self.neuron_activations[layer_name] = output.detach().cpu()
        return hook_fn

    def clear_hooks(self):
        for h in self.hooks:
            h.remove()


class NeuronCoverage(NeuronCoverageBase):
    def compute_coverage(self, input_tensor, threshold=0.5):
        self.neuron_activations = {}
        _ = self.model(input_tensor)

        total_neurons = 0
        activated = 0
        for act in self.neuron_activations.values():
            flat = act.view(-1)
            total_neurons += flat.numel()
            activated += (torch.sigmoid(flat) > threshold).sum().item()

        coverage = activated / total_neurons * 100 if total_neurons > 0 else 0
        return coverage


class TopKNeuronCoverage(NeuronCoverageBase):
    def __init__(self, model, top_k=1000):
        super().__init__(model)
        self.top_k = top_k

    def compute_coverage(self, input_tensor):
        self.neuron_activations = {}
        _ = self.model(input_tensor)

        all_activations = torch.cat([act.flatten() for act in self.neuron_activations.values()])
        topk_vals, _ = torch.topk(all_activations, min(self.top_k, all_activations.numel()))
        coverage = len(topk_vals.unique()) / all_activations.numel() * 100
        return coverage


class NeuronBoundaryCoverage(NeuronCoverageBase):
    def compute_boundary_coverage(self, input1, input2):
        self.neuron_activations = {}
        _ = self.model(input1)
        act1 = {k: v.clone() for k, v in self.neuron_activations.items()}

        self.neuron_activations = {}
        _ = self.model(input2)
        act2 = self.neuron_activations

        diff_count = 0
        total = 0
        for k in act1.keys():
            if k in act2:
                total += act1[k].numel()
                diff_count += torch.sum(act1[k] != act2[k]).item()

        coverage = diff_count / total * 100 if total > 0 else 0
        return coverage