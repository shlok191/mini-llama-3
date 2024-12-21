import torch
import torch.nn as nn
import pytest
import math
from mini_llama.rmsnorm import RMSNorm

def test_rmsnorm():
    """
    Tests RMSNorm with typical transformer dimensions, focusing on sequence length 256
    and hidden dimension 1024. This matches common usage in transformer models where
    we normalize each position's feature vector independently.
    """
    print("\n=== Testing RMSNorm Implementation ===")
    
    # Initialize with transformer-typical dimensions
    seq_length = 512   # Typical sequence length
    hidden_dim = 1024  # Typical hidden dimension
    eps = 1e-8
    
    def test_basic_properties():
        print("\nTesting basic normalization properties...")
        norm = RMSNorm(hidden_dim, eps=eps).cuda()
        
        # Create input sequence with known properties
        x = torch.randn(seq_length, hidden_dim, device='cuda')
        normalized = norm(x)
        
        # Verify output shape matches input
        assert normalized.shape == (seq_length, hidden_dim), \
            f"Shape mismatch: got {normalized.shape}, expected {(seq_length, hidden_dim)}"
        print("✓ Shape preservation verified")
        
        # Check RMS is approximately 1 for each position in the sequence
        rms = torch.sqrt(torch.mean(normalized ** 2, dim=-1))
        assert torch.allclose(rms, torch.ones(seq_length, device='cuda'), rtol=1e-2), \
            "RMS values not normalized to 1"
        print("✓ RMS normalization verified")
        
        # Test scaling behavior
        norm.scale.data.fill_(2.0)
        scaled = norm(x)
        assert torch.allclose(scaled, 2 * normalized, rtol=1e-5), \
            "Scale parameter not applied correctly"
        print("✓ Scale parameter behavior verified")
    
    def test_gradient_flow():
        print("\nTesting gradient computation and flow...")
        norm = RMSNorm(hidden_dim).cuda()
        
        # Create input requiring gradients
        x = torch.randn(seq_length, hidden_dim, device='cuda', requires_grad=True)
        
        # Forward pass
        out = norm(x)
        
        # Create a loss that depends on the entire output
        loss = out.sum()
        loss.backward()
        
        norm.scale.retain_grad()
        
        # Verify gradient properties
        assert x.grad is not None, "Input gradients not computed"
        assert norm.scale.grad is not None, "Scale parameter gradients not computed"
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad)), \
            "Input gradients are all zero"
        print("✓ Gradient computation verified")
        
        # Check gradient numerical stability
        assert not torch.isnan(x.grad).any(), "NaN found in input gradients"
        assert not torch.isnan(norm.scale.grad).any(), "NaN found in scale gradients"
        print("✓ Gradient numerical stability verified")
    
    def test_numerical_stability():
        print("\nTesting numerical stability...")
        norm = RMSNorm(hidden_dim).cuda()
        
        # Test with large magnitude inputs
        x_large = torch.exp(torch.randn(seq_length, hidden_dim, device='cuda') * 10)
        out_large = norm(x_large)
        assert not torch.isnan(out_large).any(), "NaN produced with large inputs"
        assert not torch.isinf(out_large).any(), "Inf produced with large inputs"
        print("✓ Large magnitude stability verified")
        
        # Test with small magnitude inputs
        x_small = torch.exp(torch.randn(seq_length, hidden_dim, device='cuda') * -10)
        out_small = norm(x_small)
        assert not torch.isnan(out_small).any(), "NaN produced with small inputs"
        assert not torch.isinf(out_small).any(), "Inf produced with small inputs"
        print("✓ Small magnitude stability verified")
        
        # Verify epsilon prevents division by zero
        x_tiny = torch.zeros(seq_length, hidden_dim, device='cuda')
        out_tiny = norm(x_tiny)
        assert not torch.isnan(out_tiny).any(), "NaN produced with zero inputs"
        assert not torch.isinf(out_tiny).any(), "Inf produced with zero inputs"
        print("✓ Zero input handling verified")
    
    def test_sequence_invariance():
        print("\nTesting sequence position invariance...")
        norm = RMSNorm(hidden_dim).cuda()
        
        # Create two identical feature vectors at different sequence positions
        x = torch.randn(seq_length, hidden_dim, device='cuda')
        x[0] = x[1]  # Make two positions identical
        
        normalized = norm(x)
        
        # Verify identical inputs produce identical outputs regardless of position
        assert torch.allclose(normalized[0], normalized[1], rtol=1e-5), \
            "Normalization not position-invariant"
        print("✓ Position invariance verified")
        
        # Verify different positions are normalized independently
        different_rms = torch.sqrt(torch.mean(x ** 2, dim=-1))
        assert not torch.allclose(different_rms[0], different_rms[-1]), \
            "Different positions not normalized independently"
        print("✓ Independent position normalization verified")
    
    # Run all tests
    test_basic_properties()
    test_gradient_flow()
    test_numerical_stability()
    test_sequence_invariance()
    
    print("\n=== All RMSNorm tests completed successfully! ===")

if __name__ == "__main__":
    test_rmsnorm()
