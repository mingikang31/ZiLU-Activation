import torch
import torch.nn as nn
import time
import pandas as pd
from Models.activation import *

"""
Time benchmarking for various activation functions on different devices (forward and backward passes). 
"""
def benchmark_activation(activation_fn, input_tensor, device, num_warmup=10, num_iterations=100, compile=True):
    """
    Benchmark forward and backward pass times for an activation function.
    
    Args:
        activation_fn: The activation function module
        input_tensor: Input tensor for testing
        device: Device to run on ('cpu', 'mps', 'cuda')
        num_warmup: Number of warmup iterations
        num_iterations: Number of timed iterations
    
    Returns:
        dict with forward_time and backward_time in milliseconds
    """
    activation_fn = activation_fn.to(device)
    input_tensor = input_tensor.to(device)

    if compile: 
        activation_fn = torch.compile(activation_fn)
        
    # Warmup
    for _ in range(num_warmup):
        output = activation_fn(input_tensor)
        if input_tensor.requires_grad:
            output.sum().backward()
            input_tensor.grad = None
    
    # Synchronize device
    if device == 'cuda':
        torch.cuda.synchronize()
    elif device == 'mps':
        torch.mps.synchronize()
    
    # Benchmark forward pass
    forward_times = []
    for _ in range(num_iterations):
        if device == 'cuda':
            torch.cuda.synchronize()
            start = time.perf_counter()
        elif device == 'mps':
            torch.mps.synchronize()
            start = time.perf_counter()
        else:
            start = time.perf_counter()
        
        output = activation_fn(input_tensor)
        
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
        
        end = time.perf_counter()
        forward_times.append((end - start) * 1000)  # Convert to ms
    
    # Benchmark backward pass
    backward_times = []
    for _ in range(num_iterations):
        output = activation_fn(input_tensor)
        
        if device == 'cuda':
            torch.cuda.synchronize()
            start = time.perf_counter()
        elif device == 'mps':
            torch.mps.synchronize()
            start = time.perf_counter()
        else:
            start = time.perf_counter()
        
        output.sum().backward()
        
        if device == 'cuda':
            torch.cuda.synchronize()
        elif device == 'mps':
            torch.mps.synchronize()
        
        end = time.perf_counter()
        backward_times.append((end - start) * 1000)  # Convert to ms
        
        input_tensor.grad = None
    
    return {
        'forward_mean': sum(forward_times) / len(forward_times),
        'forward_std': torch.tensor(forward_times).std().item(),
        'backward_mean': sum(backward_times) / len(backward_times),
        'backward_std': torch.tensor(backward_times).std().item()
    }

def run_benchmarks():
    """Run benchmarks for all activation functions on all available devices."""
    
    # Test configuration
    batch_size = 64
    input_size = 1024
    sigma = 5.0
    
    # Create test input
    input_tensor = torch.randn(batch_size, input_size, requires_grad=True)
    
    # Activation functions to test
    activations = {
        "ReLU": nn.ReLU(),
        "SiLU": nn.SiLU(),
        "GELU": nn.GELU(),
        "Sigmoid": nn.Sigmoid(),
        "LeakyReLU": nn.LeakyReLU(),
        "PReLU": nn.PReLU(),
        "ELU": nn.ELU(),
        "Hardshrink": nn.Hardshrink(),
        "Softshrink": nn.Softshrink(),
        "Tanhshrink": nn.Tanhshrink(),
        "Hardtanh": nn.Hardtanh(),
        "Softplus": nn.Softplus(),
        "Softsign": nn.Softsign(),
        "Tanh": nn.Tanh(),
        "CELU": nn.CELU(),
        "Swish": nn.SiLU(),  # Swish is equivalent to SiLU
        "Mish": nn.Mish(),
        "HardSwish": nn.Hardswish(),
        "HardSigmoid": nn.Hardsigmoid(),
        "GELU_s": GELU_s(sigma=sigma),
        "SiLU_s": SiLU_s(sigma=sigma),
        "ZiLU_Old": ZiLU_Old(sigma=sigma),
        "ArcTan": ArcTan(sigma=sigma),
        "ArcTan_Approx": ArcTan_Approx(sigma=sigma),
        "ZiLU": ZiLU(sigma=sigma),
        "ZiLU_Approx": ZiLU_Approx(sigma=sigma)
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
   