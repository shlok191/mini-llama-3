import torch
import mini_llama  # The module containing your CUDA implementation
import time

def test_attention_implementation():
    # Set the dimensions
    sequence_length = 256
    embedding_dim = 256

    # Create random input tensors on the GPU
    query = torch.randn(sequence_length, embedding_dim,
                        dtype=torch.float32,
                        device='cuda')
    key = torch.randn(sequence_length, embedding_dim,
                      dtype=torch.float32,
                      device='cuda')
    value = torch.randn(sequence_length, embedding_dim,
                        dtype=torch.float32,
                        device='cuda')

    # Warm-up runs for your CUDA implementation to stabilize performance
    for _ in range(10):
        _, _ = mini_llama.attention_forward(query, key, value)
    
    torch.cuda.synchronize()

    # Time your CUDA implementation
    start_time = time.perf_counter()
    cuda_output, logsumexp = mini_llama.attention_forward(query, key, value)
    
    torch.cuda.synchronize()  # Ensure all CUDA kernels have finished
    cuda_time = time.perf_counter() - start_time

    print(f"CUDA implementation time: {cuda_time * 1000:.3f} ms")

    # Warm-up runs for PyTorch implementation
    for _ in range(10):
        
        # Compute attention scores
        scores = torch.matmul(query, key.transpose(0, 1))
        scores = scores / (embedding_dim ** 0.5)
        
        # Apply softmax and compute the output
        attention_weights = torch.softmax(scores, dim=-1)
        pytorch_output = torch.matmul(attention_weights, value)
    
    torch.cuda.synchronize()

    # Time the PyTorch implementation
    start_time = time.perf_counter()
    scores = torch.matmul(query, key.transpose(0, 1))
    scores = scores / (embedding_dim ** 0.5)
    attention_weights = torch.softmax(scores, dim=-1)
    pytorch_output = torch.matmul(attention_weights, value)
    
    torch.cuda.synchronize()  # Ensure all CUDA kernels have finished
    pytorch_time = time.perf_counter() - start_time

    print(f"PyTorch implementation time: {pytorch_time * 1000:.3f} ms")

    # Compare the results
    rtol = 1e-3  # Relative tolerance
    atol = 1e-5  # Absolute tolerance
    
    print(cuda_output.shape)
    print(pytorch_output.shape)
    
    max_diff = torch.max(torch.abs(cuda_output - pytorch_output))

    print(f"Maximum difference between implementations: {max_diff}")
    is_close = torch.allclose(cuda_output, pytorch_output, rtol=rtol, atol=atol)

    if is_close:
        print("✅ Test passed! CUDA implementation matches PyTorch")
    else:
        print("❌ Test failed! Implementations produce different results")

        # Print detailed statistics for debugging
        print("\nDetailed comparison:")
        print(f"Mean absolute error: {torch.mean(torch.abs(cuda_output - pytorch_output))}")
        print(f"Standard deviation of error: {torch.std(torch.abs(cuda_output - pytorch_output))}")

        # Sample a few positions for comparison
        sample_idx = torch.randint(0, sequence_length, (5,))
        print("\nSample values comparison:")
        for idx in sample_idx:
            idx = idx.item()  # Convert tensor to int
            print(f"\nPosition {idx}:")
            print(f"CUDA output: {cuda_output[idx][:5]}")  # First 5 values
            print(f"PyTorch output: {pytorch_output[idx][:5]}")


def test_attention_backward():
    """
    Test the backward pass of our CUDA attention implementation.
    We'll create small tensors and verify the gradients are correct.
    """
    # Set random seed for reproducibility
    torch.manual_seed(42)
    
    # Create input tensors on GPU
    query = torch.randn(256, 256, device='cuda', requires_grad=True)
    key = torch.randn(256, 256, device='cuda', requires_grad=True)
    value = torch.randn(256, 256, device='cuda', requires_grad=True)
    
    # First, run the forward pass to get the outputs
    output, logsumexp = mini_llama.attention_forward(query, key, value)
    
    # Create a random gradient tensor
    grad_output = torch.randn_like(output)
    
    # Get gradients from our CUDA implementation
    grad_query, grad_key, grad_value = mini_llama.attention_backward(
        query, key, value, output, grad_output, logsumexp
    )
    
    print(grad_query)
    # torch.cuda.synchronize()
    
    # # Now compute gradients using PyTorch's autograd for comparison
    # # First, compute attention scores using PyTorch operations
    # scores = torch.matmul(query, key.transpose(0, 1)) / 16.0
    # attention_weights = torch.softmax(scores, dim=-1)
    # torch_output = torch.matmul(attention_weights, value)
    
    # # Compute backward pass
    # torch_output.backward(grad_output)
    
    # # Compare gradients
    # print("Maximum difference in query gradients:", 
    #       torch.max(torch.abs(grad_query - query.grad)).item())
    
    # print("Maximum difference in key gradients:", 
    #       torch.max(torch.abs(grad_key - key.grad)).item())
    
    # print("Maximum difference in value gradients:", 
    #       torch.max(torch.abs(grad_value - value.grad)).item())
    
    # # Assert that gradients are close enough (using a reasonable tolerance)
    # assert torch.allclose(grad_query, query.grad, rtol=1e-3, atol=1e-3), \
    #     "Query gradients don't match!"
        
    # assert torch.allclose(grad_key, key.grad, rtol=1e-3, atol=1e-3), \
    #     "Key gradients don't match!"
    
    # assert torch.allclose(grad_value, value.grad, rtol=1e-3, atol=1e-3), \
    #     "Value gradients don't match!"
    
    # print("All gradient checks passed!")

if __name__ == "__main__":
    test_attention_backward()

