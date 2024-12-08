import torch
import mini_llama # The module containing your CUDA implementation

def test_attention_implementation():
    # First, let's create some sample input data
    # We'll use small dimensions to make it easier to debug
    sequence_length = 128
    embedding_dim = 256
    
    # Create random input tensors
    # Using float32 since that's what your CUDA implementation expects
    query = torch.randn(sequence_length, embedding_dim, 
                       dtype=torch.float32, 
                       device='cuda')
    key = torch.randn(sequence_length, embedding_dim, 
                      dtype=torch.float32, 
                      device='cuda')
    value = torch.randn(sequence_length, embedding_dim, 
                       dtype=torch.float32, 
                       device='cuda')
    
    # Run your CUDA implementation
    cuda_output = mini_llama.attention_forward(query, key, value)
    print(cuda_output)
    
    # Compute the reference PyTorch implementation
    # First, compute attention scores
    scores = torch.matmul(query, key.transpose(0, 1))
    scores = scores / (embedding_dim ** 0.5)  # Scale by sqrt(d_k)
    
    # Apply softmax
    attention_weights = torch.softmax(scores, dim=-1)
    
    # Compute weighted sum with values
    pytorch_output = torch.matmul(attention_weights, value)
    
    # Compare the results
    # Using relative tolerance since we're dealing with floating point
    rtol = 1e-3  # Relative tolerance
    atol = 1e-5  # Absolute tolerance
    max_diff = torch.max(torch.abs(cuda_output - pytorch_output))
    
    print(f"Maximum difference between implementations: {max_diff}")
    is_close = torch.allclose(cuda_output, pytorch_output, rtol=rtol, atol=atol)
    
    if is_close:
        print("✅ Test passed! CUDA implementation matches PyTorch")
    else:
        print("❌ Test failed! Implementations produce different results")
        
        # Print some detailed statistics to help with debugging
        print("\nDetailed comparison:")
        print(f"Mean absolute error: {torch.mean(torch.abs(cuda_output - pytorch_output))}")
        print(f"Standard deviation of error: {torch.std(torch.abs(cuda_output - pytorch_output))}")
        
        # Sample a few values for comparison
        sample_idx = torch.randint(0, sequence_length, (5,))
        print("\nSample values comparison:")
        for idx in sample_idx:
            print(f"\nPosition {idx}:")
            print(f"CUDA output: {cuda_output[idx][:5]}")  # First 5 values
            print(f"PyTorch output: {pytorch_output[idx][:5]}")

if __name__ == "__main__":
    test_attention_implementation()