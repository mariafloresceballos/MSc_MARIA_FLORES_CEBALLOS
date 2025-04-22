# gradcam.py

import torch
import torch.nn.functional as F

class GradCAM1D:
    """
    Implements Grad-CAM for 1D convolutional models in PyTorch.
    """

    def __init__(self, model, target_layer):
        """
        Initialize Grad-CAM with a model and the target convolutional layer.

        Args:
            model (torch.nn.Module): The model to inspect.
            target_layer (torch.nn.Module): The convolutional layer to track.
        """
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None
        self.hook_handles = []

        self._register_hooks()

    def _register_hooks(self):
        """
        Internal: Register forward and backward hooks on the target layer.
        """
        def forward_hook(module, input, output):
            self.activations = output.detach()

        def backward_hook(module, grad_input, grad_output):
            self.gradients = grad_output[0].detach()

        self.hook_handles.append(self.target_layer.register_forward_hook(forward_hook))
        self.hook_handles.append(self.target_layer.register_backward_hook(backward_hook))

    def remove_hooks(self):
        """
        Remove hooks to free memory and avoid side effects.
        """
        for handle in self.hook_handles:
            handle.remove()

    def generate(self, input_tensor, target_class):
        """
        Generate the Grad-CAM heatmap for a given input and class.

        Args:
            input_tensor (torch.Tensor): Input of shape (1, C, L).
            target_class (int): Target class index.

        Returns:
            cam (np.ndarray): Grad-CAM vector of shape (L,).
            probabilities (np.ndarray): Softmax probabilities for all classes.
        """
        self.model.zero_grad()
        output = self.model(input_tensor, raw=True)  # raw=True avoids softmax inside model
        target = output[:, target_class]
        target.backward()

        # Global average pooling of gradients
        weights = self.gradients.mean(dim=2, keepdim=True)
        cam = (weights * self.activations).sum(dim=1)
        cam = F.relu(cam)

        # Normalize CAM to [0, 1]
        cam -= cam.min()
        cam /= cam.max() + 1e-6

        return cam.squeeze().cpu().numpy(), output.softmax(dim=1).cpu().numpy()
