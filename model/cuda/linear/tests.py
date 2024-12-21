import torch
import time
from mini_llama.cuda import linear_forward, linear_backward_weights, linear_backward_inputs

def test_linear_layer(batch_size=16, sequence_length=512, in_features=512, out_features=1024, num_warmup=10, num_runs=25):
    """
    Comprehensive test suite for our custom linear layer implementation.
    Tests both forward and backward passes for correctness and performance.
    """
    device = torch.device('cuda')
    
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

    print("\n=== Testing Forward Pass ===")
    
    # Warm up GPU
    print("Warming up GPU...")
    for _ in range(num_warmup):
        _ = torch.matmul(X_torch, weights_torch)
        _ = linear_forward(X, weights)
    
    # Test forward pass correctness
    print("\nTesting forward pass correctness...")
    with torch.no_grad():
        output_custom = linear_forward(X, weights)
        output_torch = torch.matmul(X_torch, weights_torch)
        
        max_diff_forward = (output_custom - output_torch).abs().max().item()
        print(f'Forward pass maximum difference: {max_diff_forward}')
        
        if max_diff_forward < 1e-3:
            print('✓ Forward pass outputs match within tolerance')
        else:
            print('✗ Forward pass outputs differ significantly!')
    
    print("\n=== Testing Backward Pass ===")
    
    # PyTorch backward pass
    output_torch = torch.matmul(X_torch, weights_torch)
    output_torch.backward(upstream_grad)
    
    # Our custom backward pass
    output_custom = linear_forward(X, weights)
    
    # Get transposed weights for input gradient computation
    weights_T = weights.t().contiguous()
    
    # Compute gradients using our custom implementation
    grad_input_custom = linear_backward_inputs(upstream_grad, weights_T)
    
    # For weight gradients, we need to transpose the input
    # [batch, seq, in_feat] -> [batch, in_feat, seq]
    input_T = X.transpose(1, 2).contiguous()
    grad_weights_custom = linear_backward_weights(upstream_grad, input_T)
    
    # Compare gradients
    print("\nComparing gradients...")
    
    # Check input gradients
    max_diff_grad_input = (grad_input_custom - X_torch.grad).abs().max().item()
    print(f'Input gradient maximum difference: {max_diff_grad_input}')
    
    if max_diff_grad_input < 1e-3:
        print('✓ Input gradients match within tolerance')
    else:
        print('✗ Input gradients differ significantly!')
        print(f'Custom grad input mean: {grad_input_custom.mean().item():.6f}')
        print(f'PyTorch grad input mean: {X_torch.grad.mean().item():.6f}')
    
    # Check weight gradients
    max_diff_grad_weights = (grad_weights_custom - weights_torch.grad).abs().max().item()
    print(f'Weight gradient maximum difference: {max_diff_grad_weights}')
    
    if max_diff_grad_weights < 1e-3:
        print('✓ Weight gradients match within tolerance')
    else:
        print('✗ Weight gradients differ significantly!')
        print(f'Custom grad weights mean: {grad_weights_custom.mean().item():.6f}')
        print(f'PyTorch grad weights mean: {weights_torch.grad.mean().item():.6f}')
    
    print("\n=== Performance Benchmarking ===")
    
    # Benchmark forward pass
    print("\nForward Pass Timing:")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        output_custom = linear_forward(X, weights)
    torch.cuda.synchronize()
    custom_forward_time = (time.time() - start_time) / num_runs
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        output_torch = torch.matmul(X_torch, weights_torch)
    torch.cuda.synchronize()
    torch_forward_time = (time.time() - start_time) / num_runs
    
    # Benchmark backward passes
    print("\nBackward Pass Timing:")
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        grad_input_custom = linear_backward_inputs(upstream_grad, weights_T)
        grad_weights_custom = linear_backward_weights(upstream_grad, input_T)
    torch.cuda.synchronize()
    custom_backward_time = (time.time() - start_time) / num_runs
    
    torch.cuda.synchronize()
    start_time = time.time()
    for _ in range(num_runs):
        output_torch = torch.matmul(X_torch, weights_torch)
        output_torch.backward(upstream_grad)
        X_torch.grad.zero_()
        weights_torch.grad.zero_()
    torch.cuda.synchronize()
    torch_backward_time = (time.time() - start_time) / num_runs
    
    # Print performance results
    print(f'\nPerformance Results:')
    print(f'Forward Pass:')
    print(f'  Custom implementation: {custom_forward_time * 1000:.3f} ms')
    print(f'  PyTorch implementation: {torch_forward_time * 1000:.3f} ms')
    print(f'  Speedup: {torch_forward_time/custom_forward_time:.2f}x')
    
    print(f'\nBackward Pass:')
    print(f'  Custom implementation: {custom_backward_time * 1000:.3f} ms')
    print(f'  PyTorch implementation: {torch_backward_time * 1000:.3f} ms')
    print(f'  Speedup: {torch_backward_time/custom_backward_time:.2f}x')


if __name__ == "__main__":
    # Run tests with default parameters
    test_linear_layer()
    
    # Optionally, run tests with different batch sizes to test scalability
    print("\nTesting with larger batch size...")
    test_linear_layer(batch_size=32)
    
    print("\nTesting with larger sequence length...")
    test_linear_layer(sequence_length=1024)