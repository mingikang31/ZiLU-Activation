# =============================================================================
# Cell 1: Imports
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import os

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage
from typing import OrderedDict


# =============================================================================
# Cell 2: Utility Functions
# =============================================================================
def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1."""
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors, indexing='ij'), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i+1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def PSNR(x, y):
    mse = F.mse_loss(x, y)
    psnr = 10 * torch.log10(1 / mse)
    return psnr


# =============================================================================
# Cell 3: Image Loading
# =============================================================================
def get_cameraman_tensor(sidelength):
    img = Image.fromarray(skimage.data.camera())
    transform = Compose([
        Resize((sidelength, sidelength)),
        ToTensor(),
        Normalize((0.5,), (0.5,))
    ])
    img = transform(img)
    return img


img = get_cameraman_tensor(256)
print(img.shape)
plt.imshow(img.squeeze().numpy(), cmap='gray')
plt.title("Cameraman (256x256)")
plt.show()

img_noise = img + 0.1 * torch.randn_like(img)
psnr = PSNR(img, img_noise)
print(f"PSNR of noisy image: {psnr.item():.2f} dB")


# =============================================================================
# Cell 4: Dataset & DataLoader
# =============================================================================
class ImageFitting(Dataset):
    def __init__(self, sidelength):
        self.img = get_cameraman_tensor(sidelength)
        self.pixels = self.img.permute(1, 2, 0).reshape(-1, 1)
        self.coords = get_mgrid(sidelength)

    def __len__(self):
        return 1

    def __getitem__(self, idx):
        if idx > 0:
            raise IndexError("Dataset only contains one item.")
        return self.coords, self.pixels


cameraman = ImageFitting(256)
dataloader = DataLoader(cameraman, batch_size=1, pin_memory=True)


# =============================================================================
# Cell 5: Fourier Feature Encoding (shared by ALL models)
# =============================================================================
class FourierFeatureEncoding(nn.Module):
    """Random Fourier feature mapping for positional encoding.

    From 'Fourier Features Let Networks Learn High Frequency Functions in
    Low Dimensional Domains' (Tancik et al., NeurIPS 2020).

    Uses a fixed random matrix B so all models share the same encoding.
    """
    def __init__(self, in_features, num_frequencies=256, scale=10.0, seed=42):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        B = torch.randn(in_features, num_frequencies, generator=generator) * scale
        self.register_buffer('B', B)

    def forward(self, x):
        x_proj = 2 * np.pi * x @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

    @property
    def out_features(self):
        return self.B.shape[1] * 2


# Create ONE shared encoding — every model will reference this same instance
SHARED_ENCODING = FourierFeatureEncoding(
    in_features=2, num_frequencies=256, scale=10.0, seed=42
)


# =============================================================================
# Cell 6: Training Function
# =============================================================================
def train(model, image, dataloader, device, output_dir, lr=1e-4, total_steps=500):
    os.makedirs(output_dir, exist_ok=True)

    steps_til_summary = 50
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    input, target = next(iter(dataloader))
    input, target = input.to(device), target.to(device)

    psnr_history = []

    for step in range(total_steps):
        optimizer.zero_grad()
        output, coords = model(input)
        loss = F.mse_loss(output, target)

        # Compute gradient/laplace BEFORE backward frees the graph
        if step % steps_til_summary == 0:
            img_grad = gradient(output, coords)
            img_laplace = laplace(output, coords)

        loss.backward()
        optimizer.step()

        if step % steps_til_summary == 0:
            with torch.no_grad():
                psnr = PSNR(output, target)
            psnr_history.append(psnr.item())
            print(f"Step {step}: Loss = {loss.item():.6f}, PSNR = {psnr.item():.2f} dB")

            fig, axes = plt.subplots(1, 4, figsize=(18, 6))
            axes[0].imshow(target.detach().cpu().numpy().reshape(image.shape[1], image.shape[2]), cmap='gray')
            axes[1].imshow(output.detach().cpu().numpy().reshape(image.shape[1], image.shape[2]), cmap='gray')
            axes[2].imshow(img_grad.norm(dim=-1).detach().cpu().numpy().reshape(image.shape[1], image.shape[2]), cmap='gray')
            axes[3].imshow(img_laplace.detach().cpu().numpy().reshape(image.shape[1], image.shape[2]), cmap='gray')

            axes[0].set_title("Target Image")
            axes[1].set_title("Output Image")
            axes[2].set_title("Gradient Magnitude")
            axes[3].set_title("Laplacian")

            fig.suptitle(f"Step {step}: Loss = {loss.item():.6f}, PSNR = {psnr.item():.2f} dB")
            plt.savefig(f"{output_dir}/training_step_{step}.png")
            plt.close(fig)

    # Final PSNR
    with torch.no_grad():
        output, _ = model(input)
        final_psnr = PSNR(output, target)
    print(f"\nFinal PSNR: {final_psnr.item():.2f} dB")
    return psnr_history


# =============================================================================
# Cell 7: Generic INR Layer & Model (used by ALL models including Sine)
# =============================================================================
class INRLayer(nn.Module):
    """Generic layer for Implicit Neural Representations with any activation."""

    def __init__(self, in_features, out_features, activation_fn, bias=True, init='xavier'):
        super().__init__()
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.activation = activation_fn
        self.init = init
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.init == 'kaiming':
                nn.init.kaiming_normal_(self.linear.weight, nonlinearity='relu')
            elif self.init == 'xavier':
                nn.init.xavier_uniform_(self.linear.weight)
            if self.linear.bias is not None:
                nn.init.zeros_(self.linear.bias)

    def forward(self, x):
        return self.activation(self.linear(x))

    def forward_with_intermediate(self, x):
        intermediate = self.linear(x)
        return self.activation(intermediate), intermediate


class INR(nn.Module):
    """
    Generic Implicit Neural Representation model.
    All models use the same shared Fourier feature encoding for fair comparison.
    The ONLY variable is the activation function.
    """

    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 activation_fn_factory, encoding, outermost_linear=True, init='xavier'):
        super().__init__()

        self.encoding = encoding
        actual_in = encoding.out_features if encoding is not None else in_features

        self.net = []
        self.net.append(INRLayer(actual_in, hidden_features, activation_fn_factory(), init=init))

        for _ in range(hidden_layers):
            self.net.append(INRLayer(hidden_features, hidden_features, activation_fn_factory(), init=init))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            nn.init.xavier_uniform_(final_linear.weight)
            nn.init.zeros_(final_linear.bias)
            self.net.append(final_linear)
        else:
            self.net.append(INRLayer(hidden_features, out_features, activation_fn_factory(), init=init))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        x = coords
        if self.encoding is not None:
            x = self.encoding(x)
        output = self.net(x)
        return output, coords

    def forward_with_activations(self, coords, retain_grad=False):
        activations = OrderedDict()
        activation_count = 0
        x = coords.clone().detach().requires_grad_(True)
        activations['input'] = x

        if self.encoding is not None:
            x = self.encoding(x)
            activations['encoding'] = x

        for i, layer in enumerate(self.net):
            if isinstance(layer, INRLayer):
                x, intermed = layer.forward_with_intermediate(x)
                if retain_grad:
                    x.retain_grad()
                    intermed.retain_grad()
                activations[f'{layer.__class__.__name__}_{activation_count}'] = intermed
                activation_count += 1
            else:
                x = layer(x)
                if retain_grad:
                    x.retain_grad()
            activations[f'{layer.__class__.__name__}_{activation_count}'] = x
            activation_count += 1

        return activations


# =============================================================================
# Cell 8: Activation Definitions
# =============================================================================

# --- Sine (drop-in, no omega_0 tricks — just plain sin) ---
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


# --- ZiLU ---
class ArcTan(nn.Module):
    def __init__(self, sigma=None):
        super().__init__()
        self.sigma = sigma if sigma is not None else nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        return 0.5 + (1.0 / torch.pi) * torch.arctan(self.sigma * x)


class ZiLU(nn.Module):
    def __init__(self, sigma=None):
        super().__init__()
        self.arctan = ArcTan(sigma)

    def forward(self, x):
        return x * self.arctan(x)


# --- ZiLU Approx ---
class ArcTan_Approx(nn.Module):
    def __init__(self, sigma=None):
        super().__init__()
        self.sigma = sigma if sigma is not None else nn.Parameter(torch.tensor(5.0))

    def forward(self, x):
        z = self.sigma * x
        return (0.5 + torch.clamp(z, min=0)) / (1.0 + torch.abs(z))


class ZiLU_Approx(nn.Module):
    def __init__(self, sigma=None):
        super().__init__()
        self.arctan_approx = ArcTan_Approx(sigma)

    def forward(self, x):
        return x * self.arctan_approx(x)


# =============================================================================
# Cell 9: Device Setup
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Move shared encoding to device
SHARED_ENCODING.to(device)


# =============================================================================
# Cell 10: Train Sine (fair SIREN replacement — same encoding, plain sin)
# =============================================================================
model_sine = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: Sine(),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)

psnr_sine = train(model_sine, img, dataloader, device=device, output_dir="./sine/", lr=1e-3)


# =============================================================================
# Cell 11: Train ReLU
# =============================================================================
model_relu = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: nn.ReLU(),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)

psnr_relu = train(model_relu, img, dataloader, device=device, output_dir="./relu/", lr=1e-3)


# =============================================================================
# Cell 12: Train GELU
# =============================================================================
model_gelu = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: nn.GELU(),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)

psnr_gelu = train(model_gelu, img, dataloader, device=device, output_dir="./gelu/", lr=1e-3)


# =============================================================================
# Cell 13: Train SiLU
# =============================================================================
model_silu = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: nn.SiLU(),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)

psnr_silu = train(model_silu, img, dataloader, device=device, output_dir="./silu/", lr=1e-3)


# =============================================================================
# Cell 14: Train ZiLU
# =============================================================================
model_zailu = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: ZiLU(sigma=5.0),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)

psnr_zailu = train(model_zailu, img, dataloader, device=device, output_dir="./zailu/", lr=1e-3)


# =============================================================================
# Cell 15: Train ZiLU Approx
# =============================================================================
model_zailu_approx = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: ZiLU_Approx(sigma=5.0),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)

psnr_zailu_approx = train(model_zailu_approx, img, dataloader, device=device, output_dir="./zailu_approx/", lr=1e-3)


# =============================================================================
# Cell 16: Comparison Plot
# =============================================================================
fig, ax = plt.subplots(1, 1, figsize=(10, 6))

labels = ['Sine', 'ReLU', 'GELU', 'SiLU', 'ZiLU', 'ZiLU-Approx']
histories = [psnr_sine, psnr_relu, psnr_gelu, psnr_silu, psnr_zailu, psnr_zailu_approx]
colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']

for label, hist, color in zip(labels, histories, colors):
    steps = [i * 50 for i in range(len(hist))]
    ax.plot(steps, hist, label=label, color=color, linewidth=2)

ax.set_xlabel("Training Step", fontsize=14)
ax.set_ylabel("PSNR (dB)", fontsize=14)
ax.set_title("Image Fitting: PSNR Comparison (Same Fourier Encoding)", fontsize=16)
ax.legend(fontsize=12)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("./psnr_comparison.png", dpi=150)
plt.show()


# =============================================================================
# Cell 17: Side-by-Side Final Reconstructions
# =============================================================================
fig, axes = plt.subplots(2, 4, figsize=(24, 12))

axes[0, 0].imshow(img.squeeze().numpy(), cmap='gray')
axes[0, 0].set_title("Ground Truth", fontsize=14)

models = {
    'Sine': model_sine,
    'ReLU': model_relu,
    'GELU': model_gelu,
    'SiLU': model_silu,
    'ZiLU': model_zailu,
    'ZiLU-Approx': model_zailu_approx,
}

input_coords = cameraman.coords.unsqueeze(0).to(device)
target_pixels = cameraman.pixels.unsqueeze(0).to(device)

positions = [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)]

for (name, model), (r, c) in zip(models.items(), positions):
    model.eval()
    with torch.no_grad():
        output, _ = model(input_coords)
        psnr_val = PSNR(output, target_pixels)

    recon = output.detach().cpu().numpy().reshape(256, 256)
    axes[r, c].imshow(recon, cmap='gray')
    axes[r, c].set_title(f"{name} ({psnr_val.item():.2f} dB)", fontsize=14)

axes[1, 3].axis('off')

for ax_row in axes:
    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("Final Reconstructions — Same Fourier Encoding, 500 Steps", fontsize=18)
plt.tight_layout()
plt.savefig("./final_reconstructions.png", dpi=150)
plt.show()