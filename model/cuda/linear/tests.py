import torch
import time
from mini_llama.cuda import linear_forward, linear_backward_weights, linear_backward_inputs
import statistics

def test_linear_layer_forward(batch_size=16, sequence_length=512, in_features=512, out_features=1024, num_warmup=10, num_runs=25):
    """
    Comprehensive test suite for our custom linear layer implementation.
    Tests for the forward pass!
    """
    
    # Presentation matters :)
    print("=" * 80)
    print(f"Testing CUDA implementation for the Linear layer...")
    print("=" * 80)
    
    print("\nConfiguration:")
    print(f"Sequence Length: {sequence_length}")
    print(f"In features: {in_features}")
    print(f"Out features: {out_features}")
    print(f"Batch size: {batch_size}")
    
    cuda_times = []
    pytorch_times = []
    
    device = torch.device('cuda')
    
    for run in range(num_runs):
        
        print(f"\n{'-' * 80}")
        print(f"Beginning run {run}...")
        print(f"{'-' * 80}")
        
        # Create input tensors that require gradients
        X = torch.randn(
            batch_size, 
            sequence_length, 
            in_features, 
            device=device, 
            dtype=torch.float32,
            requires_grad=True
        )
        
        weights = torch.randn(
            in_features, 
            out_features, 
            device=device, 
            dtype=torch.float32,
            requires_grad=True
        )
        
        # Create copies for PyTorch implementation
        X_torch = X.detach().clone().requires_grad_(True)
        weights_torch = weights.detach().clone().requires_grad_(True)
    
        # Test forward pass correctness
        print("\nTesting forward pass correctness...")
        
        # Time CUDA implementation
        start_time = time.perf_counter()
        
        output_custom = linear_forward(X, weights)
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Appending the time taken
        cuda_times.append(end_time - start_time)
        
        start_time = time.perf_counter()
        
        output_torch = torch.matmul(X_torch, weights_torch)
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Appending the torch implementation time
        pytorch_times.append(end_time - start_time)
        
        max_diff_forward = (output_custom - output_torch).abs().max().item()
        print(f'Forward pass maximum difference: {max_diff_forward}')
        
        if max_diff_forward < 1e-3:
            print('✓ Forward pass outputs match within tolerance')
        else:
            print('✗ Forward pass outputs differ significantly!')

        # Printing out time differences
        print("\nTiming Results:")
        print(f"CUDA implementation time: {cuda_times[-1] * 1000:.3f} ms")
        print(f"PyTorch implementation time: {pytorch_times[-1] * 1000:.3f} ms")
        print(f"Speedup: {pytorch_times[-1]/cuda_times[-1]:.2f}x")

        print(f"{'-' * 80}")
    
    # Giving out final time speedup values!
    cuda_time_mean = statistics.mean(cuda_times)    
    pytorch_time_mean = statistics.mean(pytorch_times)
    
    print(f"\n{'=' * 80}")
    print("Final Timing Results:")
    print(f"{'-' * 80}")
    
    print(f"CUDA implementation time: {cuda_time_mean * 1000:.3f} ms")
    print(f"PyTorch implementation time: {pytorch_time_mean * 1000:.3f} ms")
    print(f"Speedup: {pytorch_time_mean/cuda_time_mean:.2f}x")
    
def test_linear_layer_backward(batch_size=16, sequence_length=320, in_features=512, out_features=1024, num_runs=25):
    
    # Presentation matters :)
    print("=" * 80)
    print(f"Testing CUDA implementation for the Linear layer backward pass...")
    print("=" * 80)
    
    print("\nConfiguration:")
    print(f"Sequence Length: {sequence_length}")
    print(f"In features: {in_features}")
    print(f"Out features: {out_features}")
    print(f"Batch size: {batch_size}")
    
    cuda_times = []
    pytorch_times = []
    
    device = torch.device('cuda')
    
    for run in range(num_runs):
        
        print(f"\n{'-' * 80}")
        print(f"Beginning run {run}...")
        print(f"{'-' * 80}")
        
        # Create input tensors that require gradients
        X = torch.randn(
            batch_size, 
            sequence_length, 
            in_features, 
            device=device, 
            dtype=torch.float32,
            requires_grad=True
        )
        
        weights = torch.randn(
            in_features, 
            out_features, 
            device=device, 
            dtype=torch.float32,
            requires_grad=True
        )
        
        # Create copies for PyTorch implementation
        X_torch = X.detach().clone().requires_grad_(True)
        weights_torch = weights.detach().clone().requires_grad_(True)
        
        # Define upstream gradients for backward pass
        # Using random gradients to test general case
        upstream_grad = torch.randn(
            batch_size,
            sequence_length,
            out_features,
            device=device,
            dtype=torch.float32
        )
    
        # Test forward pass correctness
        print("\nTesting backward pass correctness...")
        
        # Time CUDA implementation
        start_time = time.perf_counter()
        
        output_custom = linear_forward(X, weights)
        d_weights_cuda = linear_backward_weights(upstream_grad, X.transpose(1, 2).contiguous())
        
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Appending the time taken
        cuda_times.append(end_time - start_time)
        
        start_time = time.perf_counter()
        
        output_torch = torch.matmul(X_torch, weights_torch)
        output_torch.retain_grad()
        
        output_torch.backward(upstream_grad)
        
        torch.cuda.synchronize()
        
        end_time = time.perf_counter()
        
        # Appending the torch implementation time
        pytorch_times.append(end_time - start_time)
        
        max_diff_forward = (d_weights_cuda - weights_torch.grad).abs().max().item()
        print(f'Backward pass maximum difference: {max_diff_forward}')
        
        if max_diff_forward < 1e-3:
            print('✓ Backward pass outputs match within tolerance')
        else:
            print('✗ Backward pass outputs differ significantly!')

        # Printing out time differences
        print("\nTiming Results:")
        print(f"CUDA implementation time: {cuda_times[-1] * 1000:.3f} ms")
        print(f"PyTorch implementation time: {pytorch_times[-1] * 1000:.3f} ms")
        print(f"Speedup: {pytorch_times[-1]/cuda_times[-1]:.2f}x")

        print(f"{'-' * 80}")
    
    # Giving out final time speedup values!
    cuda_time_mean = statistics.mean(cuda_times)    
    pytorch_time_mean = statistics.mean(pytorch_times)
    
    print(f"\n{'=' * 80}")
    print("Final Timing Results:")
    print(f"{'-' * 80}")
    
    print(f"CUDA implementation time: {cuda_time_mean * 1000:.3f} ms")
    print(f"PyTorch implementation time: {pytorch_time_mean * 1000:.3f} ms")
    print(f"Speedup: {pytorch_time_mean/cuda_time_mean:.2f}x")


if __name__ == "__main__":

    test_linear_layer_backward(batch_size=64, sequence_length=320, in_features=512, out_features=1024, num_runs=10)