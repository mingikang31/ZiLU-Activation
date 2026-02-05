import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import pandas as pd

"""
Activation Functions
"""
def SquarePlus(x, beta=4):
    """Squareplus activation function."""
    return 0.5 * (x + torch.sqrt(x**2 + beta))

def ReLU(x):
    return F.relu(x)

def SiLU(x):
    return F.silu(x)

def GELU(x):
    return F.gelu(x)

def Sigmoid(x):
    return F.sigmoid(x)

def LeakyReLU(x, negative_slope=0.01):
    return F.leaky_relu(x, negative_slope=negative_slope)

def PReLU(x, weight=0.25):
    return F.prelu(x, torch.tensor(weight, device=x.device, dtype=x.dtype))

def ELU(x, alpha=1.0):
    return F.elu(x, alpha=alpha)

def Hardshrink(x, lambd=0.5):
    return F.hardshrink(x, lambd=lambd)

def Softshrink(x, lambd=0.5):
    return F.softshrink(x, lambd=lambd)

def Tanhshrink(x):
    return F.tanhshrink(x)

def Hardtanh(x, min_val=-1.0, max_val=1.0):
    return F.hardtanh(x, min_val=min_val, max_val=max_val)

def Softplus(x, beta=1, threshold=20):
    return F.softplus(x, beta=beta, threshold=threshold)

def Softsign(x):
    return F.softsign(x)

def Tanh(x):
    return F.tanh(x)

def CELU(x, alpha=1.0):
    return F.celu(x, alpha=alpha)

def Swish(x):
    return F.silu(x)  # Swish is equivalent to SiLU

def Mish(x):
    return F.mish(x)

def HardSwish(x):
    return F.hardswish(x)

def HardSigmoid(x):
    return F.hardsigmoid(x)

def ArcTan(x):
    return 0.5 + (1.0 / torch.pi) * torch.arctan(x)

def ArcTan_Approx(x):
    return (0.5 + torch.clamp(x, min=0)) / (1.0 + torch.abs(x))

def ZiLU(x):
    return x * (0.5 + (1.0 / torch.pi) * torch.arctan(x))

def ZiLU_Approx(x):
    return x * ((0.5 + torch.clamp(x, min=0)) / (1.0 + torch.abs(x)))

def SeLU(x):
  return F.selu(x)

"""
Time benchmarking for various activation functions on different devices (forward and backward passes).
"""
# ...existing code...

def benchmark_activation(activation_fn, input_tensor, device, num_warmup=10, num_iterations=100, compile=False):
    """
    Benchmark forward and backward pass times for an activation function.
    """
    input_tensor = input_tensor.to(device)

    if compile:
        activation_fn = torch.compile(activation_fn)

    # Warmup
    for _ in range(num_warmup):
        test_input = input_tensor.clone().detach().requires_grad_(True)
        output = activation_fn(test_input)
        if test_input.requires_grad:
            output.sum().backward()

    # Synchronize device
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()

    # Benchmark forward pass
    forward_times = []
    for _ in range(num_iterations):
        test_input = input_tensor.clone().detach().requires_grad_(True)

        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()

        start = time.perf_counter()
        output = activation_fn(test_input)

        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()

        end = time.perf_counter()
        forward_times.append((end - start) * 1000)

        # Keep output alive to prevent optimization
        del output

    # Benchmark backward pass
    backward_times = []
    for _ in range(num_iterations):
        test_input = input_tensor.clone().detach().requires_grad_(True)
        output = activation_fn(test_input)
        loss = output.sum()

        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()

        start = time.perf_counter()
        loss.backward()

        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()

        end = time.perf_counter()
        backward_times.append((end - start) * 1000)

    return {
        'forward_mean': sum(forward_times) / len(forward_times),
        'forward_std': torch.tensor(forward_times).std().item(),
        'backward_mean': sum(backward_times) / len(backward_times),
        'backward_std': torch.tensor(backward_times).std().item()
    }

def run_benchmarks():
    """Run benchmarks for all activation functions on all available devices."""

    # Test configuration
    input_size = 1000000  # 1 million elements

    # Create test input
    input_tensor = torch.randn(input_size, requires_grad=True)

    # Activation functions to test
    activations = {
        "ReLU": ReLU,
        "SiLU": SiLU,
        "GELU": GELU,
        "Sigmoid": Sigmoid,
        "LeakyReLU": LeakyReLU,
        "PReLU": PReLU,
        "ELU": ELU,
        "Hardshrink": Hardshrink,
        "Softshrink": Softshrink,
        "Tanhshrink": Tanhshrink,
        "Hardtanh": Hardtanh,
        "Softplus": Softplus,
        "Softsign": Softsign,
        "Tanh": Tanh,
        "CELU": CELU,
        "Swish": Swish,
        "Mish": Mish,
        "HardSwish": HardSwish,
        "HardSigmoid": HardSigmoid,
        "SquarePlus": SquarePlus,
        "ArcTan": ArcTan,
        "ArcTan_Approx": ArcTan_Approx,
        "ZiLU": ZiLU,
        "ZiLU_Approx": ZiLU_Approx, 
        "SeLU": SeLU
    }

    # Determine available devices
    devices = ['cpu']
    if torch.cuda.is_available():
        devices.append('cuda')
    if torch.backends.mps.is_available():
        devices.append('mps')

    print(f"Available devices: {devices}")
    print(f"Input shape: {input_tensor.shape}\n")

    # Run benchmarks
    results = {}
    for device in devices:
        print(f"\n{'='*60}")
        print(f"Device: {device.upper()}")
        print(f"{'='*60}")

        results[device] = {}

        for name, activation_fn in activations.items():
            print(f"\nBenchmarking {name}...")

            try:
                # Create fresh input for each test
                test_input = input_tensor.clone().detach().requires_grad_(True)

                # Run benchmark
                timing = benchmark_activation(
                    activation_fn,
                    test_input,
                    device,
                    num_warmup=10,
                    num_iterations=100
                )

                results[device][name] = timing

                print(f"  Forward:  {timing['forward_mean']:.4f} ± {timing['forward_std']:.4f} ms")
                print(f"  Backward: {timing['backward_mean']:.4f} ± {timing['backward_std']:.4f} ms")

            except Exception as e:
                print(f"  Error: {e}")
                results[device][name] = None

    # Print summary table
    print(f"\n{'='*60}")
    print("SUMMARY TABLE")
    print(f"{'='*60}")

    for device in devices:
        print(f"\n{device.upper()}:")
        print(f"{'Activation':<20} {'Forward (ms)':<20} {'Backward (ms)':<20}")
        print("-" * 60)

        for name in activations.keys():
            if results[device].get(name):
                timing = results[device][name]
                fwd = f"{timing['forward_mean']:.4f} ± {timing['forward_std']:.4f}"
                bwd = f"{timing['backward_mean']:.4f} ± {timing['backward_std']:.4f}"
                print(f"{name:<20} {fwd:<20} {bwd:<20}")
            else:
                print(f"{name:<20} {'N/A':<20} {'N/A':<20}")

    return results

def results_to_dataframe(results):
    """
    Convert benchmark results to a pandas DataFrame.

    Args:
        results: Nested dict with structure {device: {activation: {metric: value}}}

    Returns:
        pandas DataFrame with columns: Device, Activation, Forward_Mean, Forward_Std, Backward_Mean, Backward_Std
    """
    data = []

    for device, activations in results.items():
        for activation_name, timing in activations.items():
            if timing is not None:
                data.append({
                    'Device': device,
                    'Activation': activation_name,
                    'Forward_Mean (ms)': timing['forward_mean'],
                    'Forward_Std (ms)': timing['forward_std'],
                    'Backward_Mean (ms)': timing['backward_mean'],
                    'Backward_Std (ms)': timing['backward_std']
                })
            else:
                data.append({
                    'Device': device,
                    'Activation': activation_name,
                    'Forward_Mean (ms)': None,
                    'Forward_Std (ms)': None,
                    'Backward_Mean (ms)': None,
                    'Backward_Std (ms)': None
                })

    df = pd.DataFrame(data)
    return df

if __name__ == "__main__":
    results = run_benchmarks()

    # Convert to DataFrame
    df = results_to_dataframe(results)

    # Save to CSV
    output_file = './Output/benchmark_results.csv'
    df.to_csv(output_file, index=False)
