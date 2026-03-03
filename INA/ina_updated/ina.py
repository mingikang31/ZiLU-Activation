# =============================================================================
# Cell 1: Imports
# =============================================================================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import os
import time

from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import skimage
from typing import OrderedDict


# =============================================================================
# Cell 2: Reproducibility
# =============================================================================
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# =============================================================================
# Cell 3: Utility Functions
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


def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# =============================================================================
# Cell 4: Image Loading
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


# =============================================================================
# Cell 5: Dataset & DataLoader
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
# Cell 6: Fourier Feature Encoding
# =============================================================================
class FourierFeatureEncoding(nn.Module):
    """Random Fourier feature mapping for positional encoding.

    From 'Fourier Features Let Networks Learn High Frequency Functions in
    Low Dimensional Domains' (Tancik et al., NeurIPS 2020).
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


# =============================================================================
# Cell 7: Training Function
# =============================================================================
def train(model, image, dataloader, device, output_dir, lr=1e-4, total_steps=2000):
    os.makedirs(output_dir, exist_ok=True)

    steps_til_summary = 100
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    input, target = next(iter(dataloader))
    input, target = input.to(device), target.to(device)

    psnr_history = []
    loss_history = []
    start_time = time.time()

    for step in range(total_steps):
        optimizer.zero_grad()
        output, coords = model(input)
        loss = F.mse_loss(output, target)

        if step % steps_til_summary == 0:
            img_grad = gradient(output, coords)
            img_laplace = laplace(output, coords)

        loss.backward()
        optimizer.step()

        if step % steps_til_summary == 0:
            with torch.no_grad():
                psnr = PSNR(output, target)
            psnr_history.append(psnr.item())
            loss_history.append(loss.item())
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

    elapsed = time.time() - start_time

    with torch.no_grad():
        output, _ = model(input)
        final_psnr = PSNR(output, target)
    print(f"\nFinal PSNR: {final_psnr.item():.2f} dB | Time: {elapsed:.1f}s")
    return {
        'psnr_history': psnr_history,
        'loss_history': loss_history,
        'final_psnr': final_psnr.item(),
        'time': elapsed,
    }


# =============================================================================
# Cell 8: SIREN Model (original paper implementation)
# =============================================================================
class SineLayer(nn.Module):
    def __init__(self, in_features, out_features, bias=True,
                 is_first=False, omega_0=30):
        super().__init__()
        self.omega_0 = omega_0
        self.is_first = is_first
        self.in_features = in_features
        self.linear = nn.Linear(in_features, out_features, bias=bias)
        self.init_weights()

    def init_weights(self):
        with torch.no_grad():
            if self.is_first:
                self.linear.weight.uniform_(-1 / self.in_features,
                                             1 / self.in_features)
            else:
                self.linear.weight.uniform_(-np.sqrt(6 / self.in_features) / self.omega_0,
                                             np.sqrt(6 / self.in_features) / self.omega_0)

    def forward(self, input):
        return torch.sin(self.omega_0 * self.linear(input))

    def forward_with_intermediate(self, input):
        intermediate = self.omega_0 * self.linear(input)
        return torch.sin(intermediate), intermediate


class Siren(nn.Module):
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 outermost_linear=False, first_omega_0=30, hidden_omega_0=30.):
        super().__init__()

        self.net = []
        self.net.append(SineLayer(in_features, hidden_features,
                                  is_first=True, omega_0=first_omega_0))

        for i in range(hidden_layers):
            self.net.append(SineLayer(hidden_features, hidden_features,
                                      is_first=False, omega_0=hidden_omega_0))

        if outermost_linear:
            final_linear = nn.Linear(hidden_features, out_features)
            with torch.no_grad():
                final_linear.weight.uniform_(-np.sqrt(6 / hidden_features) / hidden_omega_0,
                                              np.sqrt(6 / hidden_features) / hidden_omega_0)
            self.net.append(final_linear)
        else:
            self.net.append(SineLayer(hidden_features, out_features,
                                      is_first=False, omega_0=hidden_omega_0))

        self.net = nn.Sequential(*self.net)

    def forward(self, coords):
        coords = coords.clone().detach().requires_grad_(True)
        output = self.net(coords)
        return output, coords


# =============================================================================
# Cell 9: Generic INR Layer & Model
# =============================================================================
class INRLayer(nn.Module):
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
    def __init__(self, in_features, hidden_features, hidden_layers, out_features,
                 activation_fn_factory, encoding=None, outermost_linear=True, init='xavier'):
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


# =============================================================================
# Cell 10: Activation Definitions
# =============================================================================
class Sine(nn.Module):
    def forward(self, x):
        return torch.sin(x)


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
# Cell 11: Device Setup
# =============================================================================
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")


# =============================================================================
# =============================================================================
#
#   EXPERIMENT A: No Positional Encoding (SIREN paper setup)
#   - Raw (x, y) coordinates → network → pixel value
#   - This is where SIREN shines due to omega_0
#   - ReLU/GELU should perform poorly here (spectral bias)
#
# =============================================================================
# =============================================================================
print("=" * 70)
print("EXPERIMENT A: No Positional Encoding (SIREN paper conditions)")
print("=" * 70)

results_A = {}

# --- SIREN (original, with omega_0) ---
set_seed(42)
model_siren = Siren(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    outermost_linear=True
).to(device)
print(f"SIREN params: {count_params(model_siren):,}")
results_A['SIREN'] = train(model_siren, img, dataloader, device=device,
                           output_dir="./expA_siren/", lr=1e-4, total_steps=2000)

# --- ReLU (no encoding) ---
set_seed(42)
model_relu_A = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: nn.ReLU(),
    encoding=None,
    outermost_linear=True, init='kaiming'
).to(device)
print(f"ReLU params: {count_params(model_relu_A):,}")
results_A['ReLU'] = train(model_relu_A, img, dataloader, device=device,
                          output_dir="./expA_relu/", lr=1e-4, total_steps=2000)

# --- GELU (no encoding) ---
set_seed(42)
model_gelu_A = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: nn.GELU(),
    encoding=None,
    outermost_linear=True, init='xavier'
).to(device)
print(f"GELU params: {count_params(model_gelu_A):,}")
results_A['GELU'] = train(model_gelu_A, img, dataloader, device=device,
                          output_dir="./expA_gelu/", lr=1e-4, total_steps=2000)

# --- SiLU (no encoding) ---
set_seed(42)
model_silu_A = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: nn.SiLU(),
    encoding=None,
    outermost_linear=True, init='xavier'
).to(device)
print(f"SiLU params: {count_params(model_silu_A):,}")
results_A['SiLU'] = train(model_silu_A, img, dataloader, device=device,
                          output_dir="./expA_silu/", lr=1e-4, total_steps=2000)

# --- ZiLU (no encoding) ---
set_seed(42)
model_zailu_A = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: ZiLU(sigma=1.0),
    encoding=None,
    outermost_linear=True, init='xavier'
).to(device)
print(f"ZiLU params: {count_params(model_zailu_A):,}")
results_A['ZiLU'] = train(model_zailu_A, img, dataloader, device=device,
                          output_dir="./expA_zailu/", lr=1e-4, total_steps=2000)

# --- ZiLU-Approx (no encoding) ---
set_seed(42)
model_zailu_approx_A = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: ZiLU_Approx(sigma=1.0),
    encoding=None,
    outermost_linear=True, init='xavier'
).to(device)
print(f"ZiLU-Approx params: {count_params(model_zailu_approx_A):,}")
results_A['ZiLU-Approx'] = train(model_zailu_approx_A, img, dataloader, device=device,
                                  output_dir="./expA_zailu_approx/", lr=1e-4, total_steps=2000)


# =============================================================================
# =============================================================================
#
#   EXPERIMENT B: Shared Fourier Feature Encoding
#   - All models get the SAME Fourier features
#   - Isolates the activation function from frequency representation
#   - This is the fair "which activation learns best given equal info" test
#
# =============================================================================
# =============================================================================
print("\n" + "=" * 70)
print("EXPERIMENT B: Shared Fourier Feature Encoding (isolate activation)")
print("=" * 70)

SHARED_ENCODING = FourierFeatureEncoding(
    in_features=2, num_frequencies=256, scale=10.0, seed=42
).to(device)

results_B = {}

# --- Sine (plain sin, no omega_0) ---
set_seed(42)
model_sine_B = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: Sine(),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)
print(f"Sine params: {count_params(model_sine_B):,}")
results_B['Sine'] = train(model_sine_B, img, dataloader, device=device,
                          output_dir="./expB_sine/", lr=1e-3, total_steps=2000)

# --- ReLU ---
set_seed(42)
model_relu_B = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: nn.ReLU(),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)
print(f"ReLU params: {count_params(model_relu_B):,}")
results_B['ReLU'] = train(model_relu_B, img, dataloader, device=device,
                          output_dir="./expB_relu/", lr=1e-3, total_steps=2000)

# --- GELU ---
set_seed(42)
model_gelu_B = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: nn.GELU(),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)
print(f"GELU params: {count_params(model_gelu_B):,}")
results_B['GELU'] = train(model_gelu_B, img, dataloader, device=device,
                          output_dir="./expB_gelu/", lr=1e-3, total_steps=2000)

# --- SiLU ---
set_seed(42)
model_silu_B = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: nn.SiLU(),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)
print(f"SiLU params: {count_params(model_silu_B):,}")
results_B['SiLU'] = train(model_silu_B, img, dataloader, device=device,
                          output_dir="./expB_silu/", lr=1e-3, total_steps=2000)

# --- ZiLU ---
set_seed(42)
model_zailu_B = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: ZiLU(sigma=1.0),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)
print(f"ZiLU params: {count_params(model_zailu_B):,}")
results_B['ZiLU'] = train(model_zailu_B, img, dataloader, device=device,
                          output_dir="./expB_zailu/", lr=1e-3, total_steps=2000)

# --- ZiLU-Approx ---
set_seed(42)
model_zailu_approx_B = INR(
    in_features=2, out_features=1,
    hidden_features=256, hidden_layers=3,
    activation_fn_factory=lambda: ZiLU_Approx(sigma=1.0),
    encoding=SHARED_ENCODING,
    outermost_linear=True, init='xavier'
).to(device)
print(f"ZiLU-Approx params: {count_params(model_zailu_approx_B):,}")
results_B['ZiLU-Approx'] = train(model_zailu_approx_B, img, dataloader, device=device,
                                  output_dir="./expB_zailu_approx/", lr=1e-3, total_steps=2000)


# =============================================================================
# Cell 12: Comparison Plots
# =============================================================================
colors = {
    'SIREN': '#1f77b4',
    'Sine': '#1f77b4',
    'ReLU': '#ff7f0e',
    'GELU': '#2ca02c',
    'SiLU': '#d62728',
    'ZiLU': '#9467bd',
    'ZiLU-Approx': '#8c564b',
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(20, 7))

# --- Experiment A ---
for name, res in results_A.items():
    steps = [i * 100 for i in range(len(res['psnr_history']))]
    ax1.plot(steps, res['psnr_history'], label=f"{name} ({res['final_psnr']:.1f} dB)",
             color=colors[name], linewidth=2)

ax1.set_xlabel("Training Step", fontsize=13)
ax1.set_ylabel("PSNR (dB)", fontsize=13)
ax1.set_title("Experiment A: No Encoding\n(SIREN paper conditions)", fontsize=14)
ax1.legend(fontsize=11)
ax1.grid(True, alpha=0.3)

# --- Experiment B ---
for name, res in results_B.items():
    steps = [i * 100 for i in range(len(res['psnr_history']))]
    ax2.plot(steps, res['psnr_history'], label=f"{name} ({res['final_psnr']:.1f} dB)",
             color=colors[name], linewidth=2)

ax2.set_xlabel("Training Step", fontsize=13)
ax2.set_ylabel("PSNR (dB)", fontsize=13)
ax2.set_title("Experiment B: Shared Fourier Encoding\n(Isolate activation function)", fontsize=14)
ax2.legend(fontsize=11)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("./psnr_comparison_both_experiments.png", dpi=150)
plt.show()


# =============================================================================
# Cell 13: Summary Table
# =============================================================================
print("\n" + "=" * 80)
print(f"{'':>16} | {'Exp A (No Encoding)':>22} | {'Exp B (Fourier Enc.)':>22}")
print(f"{'Activation':>16} | {'PSNR (dB)':>10} {'Time (s)':>11} | {'PSNR (dB)':>10} {'Time (s)':>11}")
print("-" * 80)

all_names = ['SIREN', 'ReLU', 'GELU', 'SiLU', 'ZiLU', 'ZiLU-Approx']
exp_B_names = ['Sine', 'ReLU', 'GELU', 'SiLU', 'ZiLU', 'ZiLU-Approx']

for name_a, name_b in zip(all_names, exp_B_names):
    psnr_a = results_A[name_a]['final_psnr'] if name_a in results_A else float('nan')
    time_a = results_A[name_a]['time'] if name_a in results_A else float('nan')
    psnr_b = results_B[name_b]['final_psnr'] if name_b in results_B else float('nan')
    time_b = results_B[name_b]['time'] if name_b in results_B else float('nan')
    print(f"{name_a:>16} | {psnr_a:>10.2f} {time_a:>10.1f}s | {psnr_b:>10.2f} {time_b:>10.1f}s")

print("=" * 80)


# =============================================================================
# Cell 14: Side-by-Side Reconstructions (both experiments)
# =============================================================================
fig, axes = plt.subplots(3, 7, figsize=(28, 12))

# Row labels
axes[0, 0].set_ylabel("Ground Truth", fontsize=13, rotation=0, labelpad=80, va='center')
axes[1, 0].set_ylabel("Exp A\n(No Encoding)", fontsize=12, rotation=0, labelpad=80, va='center')
axes[2, 0].set_ylabel("Exp B\n(Fourier Enc.)", fontsize=12, rotation=0, labelpad=80, va='center')

input_coords = cameraman.coords.unsqueeze(0).to(device)
target_pixels = cameraman.pixels.unsqueeze(0).to(device)

# Ground truth in first row, first column
for col in range(7):
    if col == 0:
        axes[0, col].imshow(img.squeeze().numpy(), cmap='gray')
        axes[0, col].set_title("Ground Truth", fontsize=12)
    else:
        axes[0, col].axis('off')

# Experiment A models
models_A = {
    'SIREN': model_siren,
    'ReLU': model_relu_A,
    'GELU': model_gelu_A,
    'SiLU': model_silu_A,
    'ZiLU': model_zailu_A,
    'ZiLU-Approx': model_zailu_approx_A,
}

for col, (name, model) in enumerate(models_A.items(), start=1):
    model.eval()
    with torch.no_grad():
        output, _ = model(input_coords)
        psnr_val = PSNR(output, target_pixels)
    recon = output.detach().cpu().numpy().reshape(256, 256)
    axes[1, col].imshow(recon, cmap='gray')
    axes[1, col].set_title(f"{name}\n{psnr_val.item():.2f} dB", fontsize=11)

axes[1, 0].imshow(img.squeeze().numpy(), cmap='gray')
axes[1, 0].set_title("Target", fontsize=11)

# Experiment B models
models_B = {
    'Sine': model_sine_B,
    'ReLU': model_relu_B,
    'GELU': model_gelu_B,
    'SiLU': model_silu_B,
    'ZiLU': model_zailu_B,
    'ZiLU-Approx': model_zailu_approx_B,
}

for col, (name, model) in enumerate(models_B.items(), start=1):
    model.eval()
    with torch.no_grad():
        output, _ = model(input_coords)
        psnr_val = PSNR(output, target_pixels)
    recon = output.detach().cpu().numpy().reshape(256, 256)
    axes[2, col].imshow(recon, cmap='gray')
    axes[2, col].set_title(f"{name}\n{psnr_val.item():.2f} dB", fontsize=11)

axes[2, 0].imshow(img.squeeze().numpy(), cmap='gray')
axes[2, 0].set_title("Target", fontsize=11)

for ax_row in axes:
    for ax in ax_row:
        ax.set_xticks([])
        ax.set_yticks([])

plt.suptitle("Image Fitting: Activation Function Comparison (2000 steps)", fontsize=16)
plt.tight_layout()
plt.savefig("./final_reconstructions_both.png", dpi=150)
plt.show()
